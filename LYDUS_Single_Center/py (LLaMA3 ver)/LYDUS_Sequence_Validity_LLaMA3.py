# (아래 코드 전체 복붙)
import yaml
import argparse
import numpy as np
import pandas as pd
import ast

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Llama(Local) client: OpenAI 대체 ----------
class LocalLlamaClient:
    def __init__(self, model_path: str, dtype: str = "bfloat16"):
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }.get(str(dtype).lower(), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )

    def chat(
        self,
        model: str,
        messages: list,
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ):
        # OpenAI-style messages → chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # deterministic
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # OpenAI-ish response mock
        class _Msg:
            def __init__(self, c): self.content = c

        class _Choice:
            def __init__(self, c): self.message = _Msg(c)

        class _Resp:
            def __init__(self, c): self.choices = [_Choice(c)]

        return _Resp(text)

# ---------- 원래 함수들 ----------
def _validate_sequence(dataset, timepoint_pairs):
    df_total = pd.DataFrame()

    for start, end in timepoint_pairs:
        print(f"\nCompare between '{start}' & '{end}'")
        start_table, start_var = start.split(' - ')
        end_table, end_var = end.split(' - ')

        if not (start_table == end_table):
            print('FAIL - Accurate comparison is difficult between different tables.')
            continue

        start_dates = dataset[dataset['Variable_name'] == start_var][
            ['Patient_id', 'Primary_key', 'Original_table_name', 'Variable_name', 'Value']
        ].dropna()
        start_dates = start_dates.rename(columns={'Variable_name': 'Start_var', 'Value': 'Start_date'})

        end_dates = dataset[dataset['Variable_name'] == end_var][
            ['Patient_id', 'Primary_key', 'Original_table_name', 'Variable_name', 'Value']
        ].dropna()
        end_dates = end_dates.rename(columns={'Variable_name': 'End_var', 'Value': 'End_date'})

        df_merged = pd.merge(
            start_dates,
            end_dates,
            on=['Patient_id', 'Original_table_name', 'Primary_key']
        )

        df_merged['Is_valid'] = df_merged['Start_date'] <= df_merged['End_date']

        df_total = pd.concat([df_total, df_merged])

    return df_total

def get_sequence_validity(quiq: pd.DataFrame,
                          model: str,
                          api_key: str):

    data = quiq.copy()
    data['Mapping_info_1'] = data['Mapping_info_1'].astype(str)

    combined_time_df = data.loc[data['Mapping_info_1'].str.contains('date', case=False, na=False)]

    assert len(combined_time_df) > 0, 'FAIL : No date values.'

    combined_time_df['Value'] = pd.to_datetime(combined_time_df['Value'], errors='coerce')

    combined_time_df['Identifier'] = combined_time_df['Original_table_name'] + ' - ' + combined_time_df['Variable_name']
    unique_names = combined_time_df['Identifier'].unique().tolist()
    unique_variables_string = ', '.join(unique_names)

    system_prompt = """
As a data analyst in a healthcare setting, your task is to validate the sequential ordering of time-related fields within medical datasets. The aim is to uphold the data quality by ensuring the chronological order of medical events as recorded across various hospitals.

Procedure:
- CSV File Analysis: Review CSV files from medical data, noting that variable names may vary in terms of explicitness and abbreviation.
- Chronological Understanding: Establish the chronological order of events based on the time-related variables within the medical data.
- Exclude Non-Time Variables: Any variable that does not represent a time point should be excluded (e.g., location, status codes, categorical data).
- Exclude Unpaired Time Variables: If a variable representing a start time does not have a corresponding end time (or vice versa), it should be excluded from the pairing process.
- Exclude Sensitive or Complex Time Variables: Variables that are sensitive in nature (like 'death_time' or 'year_of_birth') or that have complex relationships (like 'diagnosis_date' preceding 'admission_date') that require additional context should be excluded from direct pairing.
- Exclude additional context variables such as ('insertion date' preceding 'tubing change') that require additional context.
- Create Time Variable Pairs: Only after exclusions have been identified, formulate pairs of time-related variables that logically represent the beginning and end of an event or process.
- Validate and Finalize Sequence: Confirm that the pairs are in a universally applicable sequence, according to the dataset's context.
- Preserve Original Formatting: Maintain the exact case and naming of the variable names from the dataset.
- Universal Sequences: Focus on sequences with broad applicability, like medication start and end dates.
- Avoid Assumptions: Steer clear of inferring fixed orders where they may not universally apply.
- Circumvent Presumptive Sequences: Refrain from deducing sequences where the order is not consistently established.
- Case Sensitivity: Retain the exact case (uppercase or lowercase) of the original column names in the output.
Once exclusions are identified, proceed to create the output without including any of the excluded variables.

Input or output variable format has to be 'Original_table_name - Variable_name'.

Example :
Input Format: ADMISSION - admission_time, ADMISSION - discharge_time, PATIENT - death_time, EMERGENCY - emergencyreg_time, EMERGENCY - emergencyout_time, PATIENT - dateofbirth
Output Format: timepoint_pairs = [
    ('ADMISSION - admission_time', 'ADMISSION - discharge_time'), 
    ('EMERGENCY - emergencyreg_time', 'EMERGENCY - emergencyout_time'),  
]
"""

    # ---------- 여기서부터 OpenAI → LocalLlama ----------
    llama = LocalLlamaClient(model_path=model, dtype="bfloat16")

    response = llama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": unique_variables_string}
        ],
        temperature=0,
        max_tokens=1200,
        top_p=1.0
    )

    gpt_output = response.choices[0].message.content
    # 모델이 "timepoint_pairs = [...]" 형태로 준다고 가정
    timepoint_pairs = ast.literal_eval(gpt_output.split('=', 1)[1].strip())

    print(f'\nIdentified Pairs : {timepoint_pairs}')

    df_total = _validate_sequence(combined_time_df, timepoint_pairs)

    df_summary = df_total.groupby(['Original_table_name', 'Start_var', 'End_var']).agg({
        'Is_valid': 'sum',
        'Primary_key': 'count'
    }).reset_index()
    df_summary = df_summary.rename(columns={'Is_valid': 'Valid_num', 'Primary_key': 'Total_num'})
    df_summary['Sequence_Validity (%)'] = np.round(df_summary['Valid_num'] / df_summary['Total_num'] * 100, 2)
    df_summary['Invalid_num'] = df_summary['Total_num'] - df_summary['Valid_num']

    df_summary = df_summary[
        ['Original_table_name', 'Start_var', 'End_var', 'Total_num', 'Invalid_num', 'Sequence_Validity (%)']
    ]

    return df_total, df_summary


if __name__ == '__main__':
    print('<LYDUS - Sequence Validity (Llama3 local)>')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')  # ex) "/content/Meta-Llama-3.1-8B-Instruct"
    api_key = config.get('api_key')      # local에서는 사용 안 함

    df_total, df_summary = get_sequence_validity(quiq, model_ver, api_key)

    total_num = df_summary['Total_num'].sum()
    invalid_num = df_summary['Invalid_num'].sum()
    valid_num = total_num - invalid_num
    sequence_validity = valid_num / total_num * 100
    sequence_validity = round(sequence_validity, 2)

    with open(save_path + '/sequence_validity_total.txt', 'w', encoding='utf-8') as file:
        file.write(f'Sequence Validity (%) = {sequence_validity}\n')
        file.write(f'Total Num = {total_num}\n')
        file.write(f'Invalid Num = {invalid_num}\n')

    df_total.to_csv(save_path + '/sequence_validity_detail.csv', index=False)
    df_summary.to_csv(save_path + '/sequence_validity_summary.csv', index=False)

    print('\n<SUCCESS>')
