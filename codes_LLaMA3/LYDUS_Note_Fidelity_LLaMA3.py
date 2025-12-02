import os, re, argparse, yaml, ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.axes import Axes
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Local Llama client (OpenAI 대체)
# =========================
class LocalLlamaClient:
    def __init__(self, model_path: str, dtype: str = "bfloat16"):
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(str(dtype).lower(), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    def chat(self, model: str, messages: list, temperature: float = 0, max_tokens: int = 800, top_p: float = 1.0):
        # OpenAI chat 포맷과 유사한 messages를 받아 템플릿 생성
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # deterministic
                pad_token_id=self.tokenizer.pad_token_id
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # OpenAI-ish response mock
        class _Msg:    # noqa
            def __init__(self, c): self.content = c
        class _Choice: # noqa
            def __init__(self, c): self.message = _Msg(c)
        class _Resp:   # noqa
            def __init__(self, c): self.choices = [_Choice(c)]
        return _Resp(text)

# =========================
# 상수/그림 함수 (원본 로직 유지)
# =========================
MAPPING_INFO_2 = {
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
    valid_categories = list(MAPPING_INFO_2.keys())
    filtered_data = result_df[result_df['Mapping_info_2'].isin(valid_categories)].copy()
    filtered_data['Mapping_info_2'] = filtered_data['Mapping_info_2'].map(MAPPING_INFO_2)
    sns.boxplot(x='Mapping_info_2', y='Fidelity_results', data=filtered_data, ax=ax)
    sns.stripplot(x='Mapping_info_2', y='Fidelity_results', data=filtered_data, color='blue', jitter=True, alpha=0.5, size=10, ax=ax)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Fidelity Results (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(filtered_data['Mapping_info_2'].unique())))
    ax.set_xticklabels(filtered_data['Mapping_info_2'].unique(), rotation=30)
    ax.grid()

# =========================
# Clinical fidelity (원본 로직 최대 유지)
# =========================
def _run_clinical(df: pd.DataFrame, client: LocalLlamaClient, model: str) -> Tuple[pd.DataFrame, float, float]:
    tqdm.pandas(desc="Running fidelity (clinical)")
    df = df[df['Mapping_info_1'] == 'note_clinical'].copy()

    mapping = {
        "ADM": "Hospitalization Record",
        "DIS": "Discharge Record",
        "SUR": "Surgical Record",
        "EME": "Emergency Treatment Record"
    }

    report_templates = {
        "Hospitalization Record": """Admission department:
Chief complaint:
Present ilness:
Past medical history:
Social & Familty history:
Physical exammination:
Review of systems:
Diagnosis:
Treatment plan:""",
        "Discharge Record": """Admission department:
Discharge department:
Discharge reason:
Diagnosis:
Summary of progression:
Medical prescription:
Surgery or procedure:
Treatment result:
Discharge plan:
Discharge form:""",
        "Surgical Record": """Preoperative diagnosis:
Postoperative diagnosis:
Preoperative procedure:
Postoperative procedure:
Type of anesthesia:
Operative findings:
Surgical procedure""",
        "Emergency Treatment Record": """Visit information:
Past medical history:
Medication history:
Present illness:
Examination findings:
Presumptive diagnosis:
Treatment plan:"""
    }

    system_prompt = (
        "Input is the template and the corresponding medical record.\n"
        "Task: Look at the items in the report template and determine whether each item is present in the medical record. "
        'Then classify them as "mentioned" or "not mentioned".\n'
        "Output format\n- (Item from the template): mentioned/not mentioned"
    )

    def process_row(row):
        template = report_templates.get(mapping.get(row['Mapping_info_2'], ""), "")
        if not template:
            return "Invalid mapping or template missing"
        query = template + "\n" + str(row['Value'])
        try:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}]
            response = client.chat(model=model, messages=messages, temperature=0, max_tokens=800)
            return response.choices[0].message.content
        except Exception:
            return None

    def retry_process_row(row, attempts=3):
        for _ in range(attempts):
            res = process_row(row)
            if res is not None:
                return res
        return "Failed after retries"

    df['Fidelity_clinical'] = df.progress_apply(retry_process_row, axis=1)

    def clinical_fidelity_score(report_text: str):
        if report_text is None:
            return 0
        # 각 라인에서 : 뒤 부분을 모으고 'not mentioned' 비율로 점수 계산
        lines = [ln.strip() for ln in str(report_text).split('\n') if ln.strip()]
        pairs = [ln for ln in lines if ':' in ln]
        total = len(pairs)
        if total == 0:
            return 0
        not_mentioned = 0
        for ln in pairs:
            rhs = ln.split(':', 1)[1].strip().lower()
            if 'not mentioned' in rhs:
                not_mentioned += 1
        return round((1 - (not_mentioned / total)) * 100, 2)

    df_filtered = df[df['Fidelity_clinical'] != 'Invalid mapping or template missing'].copy()
    df_filtered['Fidelity_clinical_results'] = df_filtered['Fidelity_clinical'].apply(clinical_fidelity_score)

    return df_filtered, round(df_filtered['Fidelity_clinical_results'].mean(), 2), round(df_filtered['Fidelity_clinical_results'].std(), 2)

# =========================
# Radiology fidelity (원본 로직 최대 유지)
# =========================
def _run_radiology(df: pd.DataFrame, client: LocalLlamaClient, model: str) -> Tuple[pd.DataFrame, float, float]:
    tqdm.pandas(desc="Running fidelity (radiology)")
    df = df[df['Mapping_info_1'] == 'note_rad'].copy()

    mapping = {k: v for k, v in MAPPING_INFO_2.items() if k in ["ACT", "BCT", "CCT", "SCT", "CXR", "AXR", "SXR", "ECH"]}

    report_templates = {
        "CT abdomen": """Liver :
Gallbladder :
Spleen :
Pancreas :
Adrenals :
Kidneys :
Bowel :
Mesentery/Peritoneum :
Nodes :
Pelvis :
Bone windows :
Vasculature :
Soft Tissues :""",
        "CT brain": """Extra-axial spaces:
Ventricular system:
Basal cisterns:
Cerebral parenchyma:
Cerebellum:
Brainstem:
Vascular system:
Paranasal sinuses and mastoid air cells:
Visualized orbits:
Bone:""",
        "CT chest": """Pulmonary Parenchyma and Airways:
Pleural Space:
Heart and Pericardium:
Mediastinum and Hila:
Thoracic Vessels:
Osseous Structures and Chest Wall:
Upper Abdomen:
Additional findings:""",
        "CT spine": """Alignment :
Bones :
Intervertebral Discs :
Spinal canal :
Paraspinal soft tissues :
Others :""",
        "X-ray chest": """Lungs :
Heart :
Mediastinum :
Pleural Spaces :
Osseous Structures :""",
        "X-ray abdomen": """Bowel gas pattern :
Abnormal calcifications :
Bones :
Others :""",
        "X-ray spine": """Alignment :
Vertebral bodies :
Intervertebral spaces :
Soft Tissues :
Others:""",
        "Echocardiography": """Left ventricle :
Left atrium :
Right atrium :
Right ventricle :
Aortic valve :
Mitral valve :
Tricuspid valve :
Pulmonic valve :
Pericardium :
Aorta :
Pulmonary artery :
Inferior vena cava and pulmonary veins :"""
    }

    system_prompt = (
        "Input is the template and the corresponding medical record.\n"
        "Task: Look at the items in the report template and determine whether each item is present in the medical record. "
        'Then classify them as "mentioned" or "not mentioned".\n'
        "Output format\n- (Item from the template): mentioned/not mentioned"
    )

    def process_row(row):
        template = report_templates.get(mapping.get(row['Mapping_info_2'], ""), "")
        if not template:
            return "Invalid mapping or template missing"
        query = template + "\n" + str(row['Value'])
        try:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}]
            response = client.chat(model=model, messages=messages, temperature=0, max_tokens=600)
            return response.choices[0].message.content
        except Exception:
            return None

    def retry_process_row(row, attempts=3):
        for _ in range(attempts):
            res = process_row(row)
            if res is not None:
                return res
        return "Failed after retries"

    df['Fidelity_radiology'] = df.progress_apply(retry_process_row, axis=1)

    def radiology_fidelity_score(text: str):
        if text is None or text == 'Invalid mapping or template missing':
            return 0
        items = [line for line in str(text).lower().split('\n') if line.strip()]
        total = len(items)
        not_mentioned = sum('not mentioned' in line for line in items)
        return round((1 - (not_mentioned / total)) * 100, 2) if total else 0

    df_filtered = df[df['Fidelity_radiology'] != 'Invalid mapping or template missing'].copy()
    df_filtered['Fidelity_radiology_results'] = df_filtered['Fidelity_radiology'].apply(radiology_fidelity_score)

    return df_filtered, round(df_filtered['Fidelity_radiology_results'].mean(), 2), round(df_filtered['Fidelity_radiology_results'].std(), 2)

# =========================
# Orchestrator (원본 집계 로직 유지)
# =========================
def get_unstructured_fidelity(
    quiq: pd.DataFrame,
    model: str,
    api_key: str   # 호환용 (미사용)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float, float]:

    assert len(quiq[(quiq['Mapping_info_1'] == 'note_rad') | (quiq['Mapping_info_1'] == 'note_clinical')]) > 0, 'FAIL : No note_rad or note_clinical values.'

    llama = LocalLlamaClient(model_path=model, dtype="bfloat16")

    df_clinical, mean_clinical, std_clinical = _run_clinical(quiq.copy(), llama, model)
    df_radiology, mean_radiology, std_radiology = _run_radiology(quiq.copy(), llama, model)

    df1 = df_clinical.rename(columns={
        'Fidelity_clinical': 'Fidelity',
        'Fidelity_clinical_results': 'Fidelity_results'
    })
    df2 = df_radiology.rename(columns={
        'Fidelity_radiology': 'Fidelity',
        'Fidelity_radiology_results': 'Fidelity_results'
    })
    result_df = pd.concat([df1, df2], ignore_index=True)

    summary_df = result_df.groupby(['Mapping_info_1', 'Mapping_info_2'])['Fidelity_results'].agg(
        Count='count',
        Fidelity_score_mean=lambda x: round(x.mean(), 2),
        Fidelity_score_std=lambda x: round(x.std(), 2)
    ).reset_index()
    summary_df['Mapping_info_2_'] = summary_df['Mapping_info_2'].map(MAPPING_INFO_2)
    summary_df = summary_df[['Mapping_info_1', 'Mapping_info_2', 'Mapping_info_2_', 'Count', 'Fidelity_score_mean', 'Fidelity_score_std']]
    summary_df = summary_df.sort_values(by='Fidelity_score_mean', ascending=False)

    return df_clinical, df_radiology, result_df, summary_df, mean_clinical, std_clinical, mean_radiology, std_radiology

# =========================
# CLI
# =========================
if __name__ == '__main__':
    print('<LYDUS - Note Fidelity>\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    quiq_path = config.get('quiq_path')
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')   # HF repo or local model path
    api_key = config.get('api_key')       # dummy (호환용)

    os.makedirs(save_path, exist_ok=True)

    quiq = pd.read_csv(quiq_path)

    df_clinical, df_radiology, result_df, summary_df, \
    mean_clinical, std_clinical, mean_radiology, std_radiology = get_unstructured_fidelity(
        quiq=quiq,
        model=model_ver,
        api_key=api_key,
    )

    result_df.to_csv(f"{save_path}/note_fidelity_total_detail.csv", index=False,
                    columns=['Mapping_info_1', 'Mapping_info_2', 'Primary_key', 'Original_table_name', 'Variable_name',
                             'Event_date', 'Value', 'Fidelity', 'Fidelity_results'])
    summary_df.to_csv(f"{save_path}/note_fidelity_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_unst_fidelity_box_plot(ax, result_df)
    fig.tight_layout()
    fig.savefig(f"{save_path}/note_fidelity_plot.png")

    fidelity_scores = result_df['Fidelity_results']
    mean_fidelity = round(fidelity_scores.mean(), 2)

    with open(os.path.join(save_path, 'note_fidelity_total.txt'), 'w', encoding='utf-8') as file:
        file.write(f'Note Fidelity (%) = {mean_fidelity}\n')

    print('\n<SUCCESS>')
