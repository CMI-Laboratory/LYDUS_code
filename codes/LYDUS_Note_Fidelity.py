import re
import openai
import argparse
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Union
from matplotlib.axes import Axes
from tqdm import tqdm

UNSTRUCTURED_FIDELITY_MAP = {
    "ACT": "CT abdomen",
    "BCT": "CT brain",
    "CCT": "CT chest",
    "SCT": "CT spine",
    "CXR": "X-ray chest",
    "AXR": "X-ray abdomen",
    "SXR": "X-ray spine",
    "ECH": "Echocardiography",
    "ADM": "Admission note",
    "DIS": "Discharge summary",
    "SUR": "Surgery note",
    "EME": "Emergency note"
}

def draw_unst_fidelity_box_plot(ax: Axes, result_df: pd.DataFrame):
    ax.clear()
    valid_categories = list(UNSTRUCTURED_FIDELITY_MAP.keys())
    filtered_data = result_df[result_df['Mapping_info_2'].isin(valid_categories)].copy()
    filtered_data['Mapping_info_2'] = filtered_data['Mapping_info_2'].map(UNSTRUCTURED_FIDELITY_MAP)
    #filtered_data['Fidelity_results'] *= 100
    sns.boxplot(x='Mapping_info_2', y='Fidelity_results', data=filtered_data, ax=ax)
    sns.stripplot(x='Mapping_info_2', y='Fidelity_results', data=filtered_data, color='blue', jitter=True, alpha=0.5, size=10, ax=ax)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Fidelity Results (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(filtered_data['Mapping_info_2'].unique())))
    ax.set_xticklabels(filtered_data['Mapping_info_2'].unique(), rotation=30)
    ax.grid()

def _run_clinical(df: pd.DataFrame, client: openai.OpenAI, model: str) -> Tuple[pd.DataFrame, float, float]:
    tqdm.pandas(desc="Running fidelity (clinical)")
    df = df[df['Mapping_info_1'] == 'note_clinical'].copy()
    mapping = {"ADM": "Hospitalization Record", "DIS": "Discharge Record", "SUR": "Surgical Record", "EME": "Emergency Treatment Record"}

    report_templates = {
        "Hospitalization Record": """Admission department:\nChief complaint:\nPresent ilness:\nPast medical history:\nSocial & Familty history:\nPhysical exammination:\nReview of systems:\nDiagnosis:\nTreatment plan:""",
        "Discharge Record": """Admission department:\nDischarge department:\nDischarge reason:\nDiagnosis:\nSummary of progression:\nMedical prescription:\nSurgery or procedure:\nTreatment result:\nDischarge plan:\nDischarge form:""",
        "Surgical Record": """Preoperative diagnosis:\nPostoperative diagnosis:\nPreoperative procedure:\nPostoperative procedure:\nType of anesthesia:\nOperative findings:\nSurgical procedure""",
        "Emergency Treatment Record": """Visit information:\nPast medical history:\nMedication history:\nPresent illness:\nExamination findings:\nPresumptive diagnosis:\nTreatment plan:"""
    }

    system_prompt = """Input is the template and the corresponding medical record.\nTask: Look at the items in the report template and determine whether each item is present in the medical record. Then classify them as "mentioned" or "not mentioned".\nOutput format\n- (Item from the template): mentioned/not mentioned"""
    
    def process_row(row):
        template = report_templates.get(mapping.get(row['Mapping_info_2'], ""), "")
        if not template:
            return "Invalid mapping or template missing"
        query = template + "\n" + row['Value']
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            response = client.chat.completions.create(model=model, messages=messages, temperature=0)
            return response.choices[0].message.content
        except Exception:
            return None

    def retry_process_row(row, attempts=3):
        for _ in range(attempts):
            result = process_row(row)
            if result is not None:
                return result
        return "Failed after retries"

    df['Fidelity_clinical'] = df.progress_apply(retry_process_row, axis=1)

    def clinical_fidelity_score(report_text):
        if report_text is None:
            return 0
        findings_items = re.findall(r':\s*("[^"]+"|[^:\n]+)', report_text)
        total_items = len(findings_items)
        if total_items == 0:
            return 0
        not_mentioned = sum('not mentioned' in item.lower() for item in findings_items)
        return round((1 - (not_mentioned / total_items)) * 100, 2)

    df_filtered = df[df['Fidelity_clinical'] != 'Invalid mapping or template missing'].copy()
    df_filtered['Fidelity_clinical_results'] = df_filtered['Fidelity_clinical'].apply(clinical_fidelity_score)

    return df_filtered, round(df_filtered['Fidelity_clinical_results'].mean(), 2), round(df_filtered['Fidelity_clinical_results'].std(), 2)

def _run_radiology(df: pd.DataFrame, client: openai.OpenAI, model: str) -> Tuple[pd.DataFrame, float, float]:
    tqdm.pandas(desc="Running fidelity (radiology)")
    df = df[df['Mapping_info_1'] == 'note_rad'].copy()
    mapping = {k: v for k, v in UNSTRUCTURED_FIDELITY_MAP.items() if k in ["ACT", "BCT", "CCT", "SCT", "CXR", "AXR", "SXR", "ECH"]}

    report_templates = {
        "CT abdomen": """Liver :\nGallbladder :\nSpleen :\nPancreas :\nAdrenals :\nKidneys :\nBowel :\nMesentery/Peritoneum :\nNodes :\nPelvis :\nBone windows :\nVasculature :\nSoft Tissues :""",
        "CT brain": """Extra-axial spaces:\nVentricular system:\nBasal cisterns:\nCerebral parenchyma:\nCerebellum:\nBrainstem:\nVascular system:\nParanasal sinuses and mastoid air cells:\nVisualized orbits:\nBone:""",
        "CT chest": """Pulmonary Parenchyma and Airways:\nPleural Space:\nHeart and Pericardium:\nMediastinum and Hila:\nThoracic Vessels:\nOsseous Structures and Chest Wall:\nUpper Abdomen:\nAdditional findings:""",
        "CT spine": """Alignment :\nBones :\nIntervertebral Discs :\nSpinal canal :\nParaspinal soft tissues :\nOthers :""",
        "X-ray chest": """Lungs :\nHeart :\nMediastinum :\nPleural Spaces :\nOsseous Structures :""",
        "X-ray abdomen": """Bowel gas pattern :\nAbnormal calcifications :\nBones :\nOthers :""",
        "X-ray spine": """Alignment :\nVertebral bodies :\nIntervertebral spaces :\nSoft Tissues :\nOthers:""",
        "Echocardiography": """Left ventricle :\nLeft atrium :\nRight atrium :\nRight ventricle :\nAortic valve :\nMitral valve :\nTricuspid valve :\nPulmonic valve :\nPericardium :\nAorta :\nPulmonary artery :\nInferior vena cava and pulmonary veins :"""
    }

    system_prompt = """Input is the template and the corresponding medical record.\nTask: Look at the items in the report template and determine whether each item is present in the medical record. Then classify them as "mentioned" or "not mentioned".\nOutput format\n- (Item from the template): mentioned/not mentioned"""

    def process_row(row):
        template = report_templates.get(mapping.get(row['Mapping_info_2'], ""), "")
        if not template:
            return "Invalid mapping or template missing"
        query = template + "\n" + str(row['Value'])
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            response = client.chat.completions.create(model=model, messages=messages, temperature=0)
            return response.choices[0].message.content
        except Exception:
            return None

    def retry_process_row(row, attempts=3):
        for _ in range(attempts):
            result = process_row(row)
            if result is not None:
                return result
        return "Failed after retries"

    df['Fidelity_radiology'] = df.progress_apply(retry_process_row, axis=1)

    def radiology_fidelity_score(text):
        if text is None or text == 'Invalid mapping or template missing':
            return 0
        items = [line for line in text.lower().split('\n') if line.strip()]
        total = len(items)
        not_mentioned = sum('not mentioned' in line for line in items)
        return round((1 - (not_mentioned / total)) *100, 2) if total else 0

    df_filtered = df[df['Fidelity_radiology'] != 'Invalid mapping or template missing'].copy()
    df_filtered['Fidelity_radiology_results'] = df_filtered['Fidelity_radiology'].apply(radiology_fidelity_score)

    return df_filtered, round(df_filtered['Fidelity_radiology_results'].mean(), 2), round(df_filtered['Fidelity_radiology_results'].std(), 2)


def get_unstructured_fidelity(
    quiq: pd.DataFrame,
    model: str,
    api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float, float]:

    assert len(quiq[(quiq['Mapping_info_1'] == 'note_rad') | (quiq['Mapping_info_1'] == 'note_clinical')]) > 0, 'FAIL : No note_rad or note_clinical values.'

    client = openai.OpenAI(api_key=api_key)
    df_clinical, mean_clinical, std_clinical = _run_clinical(quiq.copy(), client, model)
    df_radiology, mean_radiology, std_radiology = _run_radiology(quiq.copy(), client, model)

    df1 = df_clinical.rename(columns={
        'Fidelity_clinical': 'Fidelity',
        'Fidelity_clinical_results': 'Fidelity_results'
    })
    df2 = df_radiology.rename(columns={
        'Fidelity_radiology': 'Fidelity',
        'Fidelity_radiology_results': 'Fidelity_results'
    })
    result_df = pd.concat([df1, df2], ignore_index=True)

    summary_df = result_df.groupby('Mapping_info_2')['Fidelity_results'].agg(
        Count='count',
        Fidelity_score_mean=lambda x: round(x.mean(), 2),
        Fidelity_score_std=lambda x: round(x.std(), 2)
    ).reset_index()
    summary_df['Mapping_info_2'] = summary_df['Mapping_info_2'].map(UNSTRUCTURED_FIDELITY_MAP)

    return df_clinical, df_radiology, result_df, summary_df, mean_clinical, std_clinical, mean_radiology, std_radiology


if __name__ == '__main__':
    print('<LYDUS - Note Fidelity>\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    quiq_path = config.get('quiq_path')
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')
    api_key = config.get('api_key')

    quiq = pd.read_csv(quiq_path)

    df_clinical, df_radiology, result_df, summary_df, \
    mean_clinical, std_clinical, mean_radiology, std_radiology = get_unstructured_fidelity(
        quiq=quiq,
        model=model_ver,
        api_key=api_key,
    )

    df_clinical.to_csv(f"{save_path}/note_fidelity_clinical_results.csv", index=False)
    df_radiology.to_csv(f"{save_path}/note_fidelity_radiology_results.csv", index=False)
    result_df.to_csv(f"{save_path}/note_fidelity_total_results.csv", index=False)
    summary_df.to_csv(f"{save_path}/note_fidelity_summary_detail.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_unst_fidelity_box_plot(ax, result_df)
    fig.tight_layout()
    fig.savefig(f"{save_path}/note_fidelity_plot.png")

    fidelity_stats_df = pd.DataFrame([
        {"Category": "Clinical", "Fidelity_mean": mean_clinical, "Fidelity_std": std_clinical},
        {"Category": "Radiology", "Fidelity_mean": mean_radiology, "Fidelity_std": std_radiology}
    ])
    fidelity_stats_df.to_csv(f"{save_path}/note_fidelity_summary.csv", index=False)

    print('\n<SUCCESS>')
