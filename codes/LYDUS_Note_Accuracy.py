import yaml
import argparse
import ast
import time
import threading
import pandas as pd
import openai
import seaborn as sns
from tqdm import tqdm
from typing import Tuple, Union
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

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

def draw_unstructured_accuracy_box_plot(ax:Axes, result_df:pd.DataFrame):
    ax.clear()
    valid_categories = list(MAPPING_INFO_2.keys())
    filtered_data = result_df[result_df['Mapping_info_2'].isin(valid_categories)].copy()
    filtered_data['Mapping_info_2'] = filtered_data['Mapping_info_2'].map(MAPPING_INFO_2)
    #filtered_data['Fidelity_results'] *= 100
    sns.boxplot(x='Mapping_info_2', y='Accuracy_results', data=filtered_data, ax=ax)
    sns.stripplot(x='Mapping_info_2', y='Accuracy_results', data=filtered_data, color='blue', jitter=True, alpha=0.5, size=10, ax=ax)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Accuracy Results (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(filtered_data['Mapping_info_2'].unique())))
    ax.set_xticklabels(filtered_data['Mapping_info_2'].unique(), rotation=30)
    ax.grid()

def _run_clinical(quiq: pd.DataFrame, openai_client: openai.OpenAI, model: str) :
    client = openai_client
    df = quiq.copy()
    df = df[df['Mapping_info_1'] == 'note_clinical']

    system_content = """Please review the medical record provided and identify errors in the following specific categories:

    1) Spelling or grammatical error
    2) Diagnostic Information Errors: 
    - Incorrect or missing disease diagnosis (eg. DM type 1 —> DM type 2), inaccuracies in the anatomic locations mentioned (eg. ascending colon —> rectum), discrepancies in locations (eg. right —> left).
    - For example, right/left side errors.
    3) Drug Information Errors
    - Incorrect or missing in prescribed drugs in report.
    4) Procedure Information Errors: 
    - Incorrect or missing procedure names, inaccuracies in the anatomic locations mentioned, discrepancies in locations.
    5)  Demographic Information Errors:
    - Incorrect or missing patient details such as name, age, or sex.
    6) Date Information Errors
    - Incorrect or missing dates, chronological errors.

    Format your response exactly as follows (JSON):
    - Spelling or Grammatical Errors: Yes/No (Brief Reason)
    - Diagnostic Information Error: Yes/No (Brief Reason)
    - Drug Information Error: Yes/No (Brief Reason)
    - Procedure Information Error: Yes/No (Brief Reason)
    - Demographic Information Error: Yes/No (Brief Reason)
    - Date Information Error: Yes/No (Brief Reason)

    Note: 
    -Limit your explanation for each error to fewer than 5 words. 
    -Only report errors that fall into these 6 specified categories. 
    -If multiple errors occur within a single category, number them.
    -Medication instruction or treatment plan or may change between admission and discharge, but the diagnosis and treatment names should remain the same.
    """

    def api_call(row, value_col='Value', results=[]):
        report = "<value note> " + row[value_col]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": report}
                ],
                temperature=0
            )
            results[0] = response.choices[0].message.content
        except Exception as e:
            results[1] = e

    def process_rows(value_col='Value', result_col='result'):
        start_time = time.time()
        errors_to_retry = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing rows for {result_col}"):
            result = [None, None]
            thread = threading.Thread(target=api_call, args=(row, value_col, result))
            thread.start()
            thread.join(timeout=60)

            if thread.is_alive() or result[1]:
                print(f"Error at row {i}: {str(result[1]) if result[1] else 'Timeout'}")
                errors_to_retry.append(i)
                thread.join()
            else:
                df.loc[i, result_col] = result[0]

        for i in errors_to_retry:
            row = df.loc[i]
            result = [None, None]
            thread = threading.Thread(target=api_call, args=(row, value_col, result))
            thread.start()
            thread.join(timeout=60)
            if not thread.is_alive() and not result[1]:
                df.loc[i, result_col] = result[0]

        print(f"Total elapsed time for {result_col}: {time.time() - start_time} seconds\n")

    process_rows(value_col='Value', result_col='Accuracy_clinical')

    def check_error1(cell_content):
        try : 
            lines = cell_content.split('\n')
            columns_to_check = ['Diagnostic Information Error', 'Procedure Information Error']
            total_count = 0
            no_error_count = 0
            for line in lines:
                for col in columns_to_check:
                    if col in line:
                        total_count += 1
                        if 'No' in line:
                            no_error_count += 1
            if total_count == 0:
                return None
            return no_error_count / total_count * 100
        except :
            return None

    if 'Accuracy_clinical' not in df.columns:
        df.insert(len(df.columns), 'Accuracy_clinical', None)
        df.insert(len(df.columns), 'Accuracy_clinical_result', None)
        return df, None, None

    df['Accuracy_clinical_result'] = df['Accuracy_clinical'].apply(check_error1)
    return df, df['Accuracy_clinical_result'].mean(), df['Accuracy_clinical_result'].std()

def _run_radiology(quiq: pd.DataFrame, openai_client: openai.OpenAI, model: str):
    client = openai_client
    df = quiq.copy()
    df = df[df['Mapping_info_1'] == 'note_rad']

    system_content = """
    Task:
    Assess the \"Impression\" section of a radiology report for critical errors that may have significant clinical implications.

    Output Format (JSON):
    {
        "error 1": "{Specify the identified error clearly or state 'no error'}",
        "error 1 reason": "{Specify the reason of error if applicable, or state 'N/A'}"
    }     
    """

    def api_call(row, value_col='Value', results=[]):
        report = "<value note> " + row[value_col]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": report}
                ],
                temperature=0
            )
            results[0] = response.choices[0].message.content
        except Exception as e:
            results[1] = e

    def process_rows(value_col='Value', result_col='result'):
        start_time = time.time()
        errors_to_retry = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing rows for {result_col}"):
            result = [None, None]
            thread = threading.Thread(target=api_call, args=(row, value_col, result))
            thread.start()
            thread.join(timeout=60)

            if thread.is_alive() or result[1]:
                print(f"Error at row {i}: {str(result[1]) if result[1] else 'Timeout'}")
                errors_to_retry.append(i)
                thread.join()
            else:
                df.loc[i, result_col] = result[0]

        for i in errors_to_retry:
            row = df.loc[i]
            result = [None, None]
            thread = threading.Thread(target=api_call, args=(row, value_col, result))
            thread.start()
            thread.join(timeout=60)
            if not thread.is_alive() and not result[1]:
                df.loc[i, result_col] = result[0]

        print(f"Total elapsed time for {result_col}: {time.time() - start_time} seconds\n")

    process_rows(value_col='Value', result_col='Accuracy_radiology')

    def check_error1(row):
        try:
            row_dict = ast.literal_eval(row)
            if 'error 1' in row_dict:
                return 0 if row_dict['error 1'].lower() != 'no error' else 100
            return None
        except (ValueError, SyntaxError):
            return None

    if 'Accuracy_radiology' not in df.columns:
        df.insert(len(df.columns), 'Accuracy_radiology', None)
        df.insert(len(df.columns), 'Accuracy_radiology_result', None)
        return df, None, None

    df['Accuracy_radiology_result'] = df['Accuracy_radiology'].apply(check_error1)
    return df, df['Accuracy_radiology_result'].mean(), df['Accuracy_radiology_result'].std()

def get_unstructured_accuracy(quiq: pd.DataFrame, model: str, api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert len(quiq[(quiq['Mapping_info_1'] == 'note_rad') | (quiq['Mapping_info_1'] == 'note_clinical')]) > 0, 'FAIL : No note_rad or note_clinical values.'

    client = openai.OpenAI(api_key=api_key)

    df_clinical, mean_clinical, std_clinical = _run_clinical(quiq, client, model)
    df_radiology, mean_radiology, std_radiology = _run_radiology(quiq, client, model)

    df_clinical_renamed = df_clinical.rename(columns={
        'Accuracy_clinical': 'Accuracy',
        'Accuracy_clinical_result': 'Accuracy_results'
    })
    df_radiology_renamed = df_radiology.rename(columns={
        'Accuracy_radiology': 'Accuracy',
        'Accuracy_radiology_result': 'Accuracy_results'
    })

    result_df = pd.concat([df_clinical_renamed, df_radiology_renamed], ignore_index=True)
    result_df = result_df.dropna(subset=['Accuracy_results'])

    summary_df = result_df.groupby(['Mapping_info_1', 'Mapping_info_2])['Accuracy_results'].agg(
        Count='count',
        Accuracy_score_mean=lambda x: round(x.mean(), 2),
        Accuracy_score_std=lambda x: round(x.std(), 2)
    ).reset_index()
    summary_df['Mapping_info_2_'] = summary_df['Mapping_info_2'].map(MAPPING_INFO_2)
    summary_df = summary_df[['Mapping_info_1', 'Mapping_info_2', 'Mapping_info_2_', 'Count', 'Accuracy_score_mean', 'Accuracy_score_std']]
    summary_df = summary_df.sort_values(by = 'Accuracy_score_mean', ascending = False)


    return df_clinical, df_radiology, result_df, summary_df

if __name__ == '__main__':
    print('<LYDUS - Note Accuracy>\n')

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

    df_clinical, df_radiology, result_df, summary_df = get_unstructured_accuracy(
        quiq=quiq,
        model=model_ver,
        api_key=api_key,
    )

    result_df.to_csv(f"{save_path}/note_accuracy_total_detail.csv", index=False,
                    columns = ['Mapping_info_1', 'Mapping_info_2', 'Primary_key', 'Original_table_name', 'Variable_name',\
                               'Event_date', 'Value', 'Accuracy', 'Accuracy_results'])
    summary_df.to_csv(f"{save_path}/note_accuracy_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_unstructured_accuracy_box_plot(ax, result_df)
    fig.tight_layout()
    fig.savefig(f"{save_path}/note_accuracy_plot.png")

    accuracy_scores = result_df['Accuracy_results']
    mean_accuracy = round(accuracy_scores.mean(), 2)

    with open(save_path + '/note_accuracy_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Note Accuracy (%) = {mean_accuracy}\n')

    print('<SUCCESS>')
