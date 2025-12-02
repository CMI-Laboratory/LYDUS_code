import os, re, gc, yaml, argparse
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Local Llama client (OpenAI 대체)
# -----------------------------
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

    def chat(self, messages: list, max_tokens: int = 200, temperature: float = 0.0):
        # OpenAI chat 포맷을 Hugging Face chat template로 변환
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,               # temperature/top_p 무시(결정적)
                pad_token_id=self.tokenizer.pad_token_id
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return text


# -----------------------------
# 원본 로직 (핵심 흐름 유지)
# -----------------------------
def match_code_regex(target_name, target_desc):
    regex_cache = {
        'ICD-9'   : r'^[0-9]{3}(\.[0-9]{1,2})?$',
        'ICD-10'  : r'^[A-Z]{1}[0-9]{2}(\.[0-9]{1,2})?$',
        'ICD-11'  : r'^[A-Z0-9]{1}[A-Z]{1}[0-9]{1}[A-Z0-9]{1}(\.[A-Z0-9]{1,2})?$',
        'SNOMED-CT': r'^[0-9]{6,18}$',
        'RxNorm'  : r'^[0-9]{5,9}$',
        'LOINC'   : r'^[0-9]{1,6}\-[0-9]{1}$',
        'ATC'     : r'^[A-Z]{1}[0-9]{2}[A-Z]{2}[0-9]{2}$'
    }

    # 원본 코드의 .item() 사용을 유지 (식별자별 1개 행이라는 전제)
    target_name = target_name.item().lower()
    target_desc = target_desc.item().lower()

    if (('icd' in target_name) and ('9' in target_name)) or (('icd' in target_desc) and ('9' in target_desc)):
        code_name = 'ICD-9'
    elif (('icd' in target_name) and ('10' in target_name)) or (('icd' in target_desc) and ('10' in target_desc)):
        code_name = 'ICD-10'
    elif (('icd' in target_name) and ('11' in target_name)) or (('icd' in target_desc) and ('11' in target_desc)):
        code_name = 'ICD-11'
    elif (('snomed' in target_name) and ('ct' in target_name)) or (('snomed' in target_desc) and ('ct' in target_desc)):
        code_name = 'SNOMED-CT'
    elif ('rxnorm' in target_name) or ('rxnorm' in target_desc):
        code_name = 'RxNorm'
    elif ('loinc' in target_name) or ('loinc' in target_desc):
        code_name = 'LOINC'
    elif ('atc' in target_name) or ('atc' in target_desc):
        code_name = 'ATC'
    else:
        code_name = None

    if code_name is not None:
        regex_for_target = regex_cache.get(code_name)
        return code_name, regex_for_target
    else:
        return None, None


def llm_define_regex(client: LocalLlamaClient, model: str, target_name, target_description):
    # ---------- 1) 코드 카테고리 식별 ----------
    system_prompt_1 = (
        "You are a medical coding assistant.\n"
        "You will be given a name and a description of a variable.\n"
        "Your task is to identify and return **exactly one standardized medical code category** "
        "(e.g. ICD-9, SNOMED-CT) that best corresponds to the description of a variable.\n"
        "Respond with only the name of the code category, no additional explanation.\n"
        "If the description does not clearly correspond to any known code category, respond with 'None'"
    )
    user_prompt_1 = (
        f"Name of a variable : {target_name.item()}\n"
        f"Description of a variable : {target_description.item()}"
    )

    code_name = client.chat(
        messages=[
            {"role": "system", "content": system_prompt_1},
            {"role": "user", "content": user_prompt_1}
        ],
        max_tokens=16, temperature=0.0
    ).strip()

    if code_name == 'None':
        return None, None

    # ---------- 2) 정규식 생성 ----------
    system_prompt_2 = (
        "You are medical coding expert.\n"
        "You will be given the name of a standardized medical code category (e.g. ICD-9, SNOMED-CT).\n"
        "Your task is to return a regular expression that accurately captures the **typical format** of codes "
        "within the specified category.\n"
        "Respond with only the regular expression, no additional explanation.\n"
        "If the format of the given category is unknown or cannot be generalized, respond with 'None'"
    )
    user_prompt_2 = f"Medical code category : {code_name}"

    regex_for_target = client.chat(
        messages=[
            {"role": "system", "content": system_prompt_2},
            {"role": "user", "content": user_prompt_2}
        ],
        max_tokens=64, temperature=0.0
    ).strip()

    if regex_for_target == 'None':
        return code_name, None

    # 공백/백틱 등 주변 문자가 섞였을 수 있어 최소 정리
    regex_for_target = regex_for_target.strip().strip('`').strip()

    return code_name, regex_for_target


def llm_validate_code(validation_target, regex_for_target):
    validation_target['Is_valid'] = validation_target['Value'].str.match(regex_for_target)
    return validation_target


def get_code_validity(quiq: pd.DataFrame, via: pd.DataFrame, model: str, api_key: str):
    # api_key는 로컬모드에서 사용하지 않지만, 원본 시그니처 유지
    llama = LocalLlamaClient(model_path=model, dtype="bfloat16")

    quiq_df = quiq.copy()
    via_df = via.copy()
    quiq_df['Mapping_info_1'] = quiq_df['Mapping_info_1'].astype(str)

    # medical_code에 대해서만 검증
    filtered_quiq_df = quiq_df[quiq_df['Mapping_info_1'].str.contains('medical_code', case=False, na=False)]
    filtered_quiq_df = filtered_quiq_df.dropna(subset=['Value'])
    # 모든 코드 문자열화 (원본 로직 유지)
    filtered_quiq_df['Value'] = filtered_quiq_df['Value'].apply(
        lambda x: str(int(x)) if (isinstance(x, float) and x.is_integer()) else str(x)
    )
    filtered_quiq_df = filtered_quiq_df[['Original_table_name', 'Variable_name', 'Value']]

    assert len(filtered_quiq_df) > 0, 'FAIL : No value related to medical code'

    gc.collect()

    validation_df = pd.DataFrame()

    filtered_quiq_df['Identifier'] = filtered_quiq_df['Original_table_name'] + ' - ' + filtered_quiq_df['Variable_name']
    via_df['Identifier'] = via_df['Original_table_name'] + ' - ' + via_df['Variable_name']

    unique_identifiers = filtered_quiq_df['Identifier'].unique()

    for current_identifier in unique_identifiers:
        validation_target = filtered_quiq_df.loc[filtered_quiq_df['Identifier'] == current_identifier].copy()
        validation_target['Regex'] = ''

        target_name = via_df.loc[via_df['Identifier'] == current_identifier]['Variable_name']
        target_description = via_df.loc[via_df['Identifier'] == current_identifier]['Description']

        if len(target_name) == 0:
            print(f"FAIL - '{current_identifier}' could not be found in the VIA table.")
            print(f'       Please check the QUIQ and VIA tables.\n')
            validation_target['Is_valid'] = False
            validation_df = pd.concat([validation_df, validation_target], axis=0, ignore_index=True)
            continue

        # 1) VIA 문자열에서 규칙 추정 시도
        code_name, regex_for_target = match_code_regex(target_name, target_description)

        # 2) 못 찾으면 LLM으로 코드카테고리/정규식 생성
        if regex_for_target is None:
            code_name, regex_for_target = llm_define_regex(llama, model, target_name, target_description)

        # 3) 여전히 없다면 실패 표기 (원본 흐름 유지)
        if regex_for_target is None:
            if code_name is None:
                print(f"FAIL - Unable to identify an appropriate code for '{current_identifier}'")
                print(f'       Please provide a more detailed VIA description.\n')
            else:
                print(f"FAIL - A medical code category ({code_name}) was detected.")
                print(f'       But an appropriate regular rexpression could not be defined.\n')

            validation_target['Is_valid'] = False
            validation_df = pd.concat([validation_df, validation_target], axis=0, ignore_index=True)
            continue

        # 4) 정규식으로 검증
        print(f"SUCCESS - Identified '{code_name}' with respect to '{current_identifier}'")
        print(f"The following regular expression is used : {regex_for_target}\n")

        validation_target['Regex'] = regex_for_target
        validation_target = llm_validate_code(validation_target, regex_for_target)
        validation_df = pd.concat([validation_df, validation_target], axis=0, ignore_index=True)

        gc.collect()

    # 요약/반환
    if validation_df.empty:
        error_summary = pd.DataFrame(columns=['Original_table_name','Variable_name','Total_code','Invalid_code','Format Validity (%)','Regular_Expression'])
        variable_vs_cases = {}
        return validation_df[['Original_table_name','Variable_name','Value','Is_valid']] if 'Is_valid' in validation_df.columns else validation_df, error_summary, variable_vs_cases

    error_summary = validation_df.groupby(['Original_table_name', 'Variable_name']).agg(
        Total_code = ('Value', 'count'),
        Invalid_code = ('Is_valid', 'sum'),          # 여기서 sum은 True=1, False=0 합(=유효 개수)
        Format_Validity = ('Value', lambda x: None),
        Regular_Expression = ('Regex', 'first')
    ).reset_index()
    error_summary = error_summary.rename(columns = {'Format_Validity' : 'Format Validity (%)'})
    # 유효 비율(=Format Validity) 먼저 계산
    error_summary['Format Validity (%)'] = (error_summary['Invalid_code'] / error_summary['Total_code'] * 100).round(2)
    # 'Invalid_code'를 실제 'Invalid' 개수로 바꾸기 (총 - 유효)
    error_summary['Invalid_code'] = error_summary['Total_code'] - error_summary['Invalid_code']

    variable_vs_cases = {}
    for current_identifier in unique_identifiers:
        temp = validation_df.loc[
            validation_df['Identifier'] == current_identifier,
            ['Original_table_name', 'Variable_name', 'Value', 'Is_valid']
        ].copy()
        variable_vs_cases[current_identifier] = temp

    validation_df = validation_df[['Original_table_name', 'Variable_name', 'Value', 'Is_valid']]

    # ✅ 빠졌던 반환 추가
    return validation_df, error_summary, variable_vs_cases


if __name__ == '__main__':
    print('<LYDUS - Format Validity>\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    quiq_path = config.get('quiq_path')
    via_path = config.get('via_path')
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')   # HF repo id 또는 로컬 폴더 경로
    api_key = config.get('api_key')       # 사용하지 않음(호환용)

    os.makedirs(save_path, exist_ok=True)

    quiq = pd.read_csv(quiq_path)
    via = pd.read_csv(via_path)

    validation_df, error_summary, variable_vs_cases = get_code_validity(quiq, via, model_ver, api_key)

    error_summary.to_csv(os.path.join(save_path, 'format_validity_summary.csv'), index=False)
    validation_df.to_csv(os.path.join(save_path, 'format_validity_detail.csv'), index=False)

    total_code = int(error_summary['Total_code'].sum()) if not error_summary.empty else 0
    invalid_code = int(error_summary['Invalid_code'].sum()) if not error_summary.empty else 0
    format_validity = round(((total_code - invalid_code) / total_code * 100), 2) if total_code > 0 else 0.0

    with open(os.path.join(save_path, 'format_validity_total.txt'), 'w', encoding='utf-8') as file:
        file.write(f'Format Validity (%) = {format_validity}\n')
        file.write(f'Total Code = {total_code}\n')
        file.write(f'Invalid Code = {invalid_code}\n')

    print('<SUCCESS>')
