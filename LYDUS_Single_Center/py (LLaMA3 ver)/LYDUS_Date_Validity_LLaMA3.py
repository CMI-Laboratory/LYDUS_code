# (아래 코드 전체 복붙)

import yaml
import argparse
import datetime
import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm
import gc
import os

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

        # tokenizer & model 로드
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
        # OpenAI-style messages를 chat template으로 변환
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

        # 프롬프트 이후 생성된 부분만 decode
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # OpenAI-ish mock 객체
        class _Msg:
            def __init__(self, c): self.content = c

        class _Choice:
            def __init__(self, c): self.message = _Msg(c)

        class _Resp:
            def __init__(self, c): self.choices = [_Choice(c)]

        return _Resp(text)

# ---------- 원래 상수 / 함수들 ----------
SYSTEM_CONTENT = """We will evaluate the quality of the medical data
I want to verify that the date given is valid.
Please answer with 'yes' or 'no'.
No other answer than 'yes' or 'no'.
"""

def _gpt_chat(client: LocalLlamaClient, model: str, user_content: str,
              temperature: float = 0, max_tokens: int = 1000) -> str:
    """
    OpenAI 버전의 _gpt_chat을 LocalLlamaClient용으로 재구현.
    문자열 하나(모델의 답변)만 리턴하도록 단순화.
    """
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": user_content}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def _valid_date_custom(date_string):
    formats = [
        "%Y년 %m월 %d일",
        "%Y %m월 %d일",
        "%Y년 %m %d일",
        "%Y년 %m월 %d",
        "%Y년 %m월 %d",
        "%Y %m월 %d",
        "%Y %m %일",
    ]
    for date_format in formats:
        try:
            datetime.datetime.strptime(str(date_string), date_format)
            return True
        except ValueError:
            pass
    return False

def _is_valid_date(date_string):
    if isinstance(date_string, pd.Timestamp):
        return True
    try:
        parse(str(date_string))
        return True
    except Exception:
        return False

def _extract_date_data_mapping(df: pd.DataFrame) -> pd.DataFrame:
    record_dates_df = df[['Original_table_name', 'Event_date']].dropna()
    record_dates_df['Variable_name'] = 'Event_date'
    record_dates_df = record_dates_df.rename(columns={'Event_date': 'Date_value'})

    date_mapping_df = df[df['Mapping_info_1'].str.contains('date', case=False, na=False)]
    date_mapping_df = date_mapping_df[['Original_table_name', 'Variable_name', 'Value']].dropna()
    date_mapping_df = date_mapping_df.rename(columns={'Value': 'Date_value'})

    combined_date_df = pd.concat([record_dates_df, date_mapping_df])

    return combined_date_df[['Original_table_name', 'Variable_name', 'Date_value']]

def _validate_date_entry(date_string, client: LocalLlamaClient, model: str):
    # 1) 파이썬/판다스 수준에서 먼저 검사
    if _is_valid_date(date_string):
        return True
    elif _valid_date_custom(date_string):
        return True
    else:
        # 2) 그래도 안 되면 LLM에게 yes/no 판단 요청
        print('gpt (llama)', date_string)
        user_content = f"date : {date_string}"
        try:
            result = _gpt_chat(client, model, user_content).strip().lower()
            # 'yes', 'no'만 오도록 프롬프트 했지만, 안전하게 in 체크
            if "no" in result:
                return False
            elif "yes" in result:
                return True
            else:
                # 애매한 답변이면 None 처리 (필요하면 False로 바꿔도 됨)
                return None
        except Exception as e:
            print("Failed to call Llama service:", e)
            return None

def get_date_validity(
    quiq: pd.DataFrame,
    model: str,
    api_key: str   # 호환성을 위해 남겨두지만 사용하지 않음 (local llama)
):
    # OpenAI 대신 local Llama 로드
    llama_client = LocalLlamaClient(model_path=model, dtype="bfloat16")

    df = quiq.copy()
    df['Mapping_info_1'] = df['Mapping_info_1'].astype(str)

    final_date_df = _extract_date_data_mapping(df)
    assert len(final_date_df) > 0, 'FAIL : No date values.'

    grouped_df_idx = final_date_df.groupby(['Original_table_name', 'Variable_name']).count().index

    summary_df = pd.DataFrame(
        columns=['Original_table_name', 'Variable_name', 'Total_date', 'Invalid_date', 'Date_Validity_(%)']
    )

    tqdm.pandas()
    valid_results_df = pd.DataFrame()

    for n, idx in enumerate(grouped_df_idx):
        table_name = idx[0]
        variable_name = idx[1]

        print(f'\n{table_name} - {variable_name}')
        temp = final_date_df[
            (final_date_df['Original_table_name'] == table_name) &
            (final_date_df['Variable_name'] == variable_name)
        ].copy()

        # Llama 사용해서 각 Date_value 검증
        temp['Is_valid'] = temp['Date_value'].progress_apply(
            lambda x: _validate_date_entry(x, llama_client, model)
        )

        total_date = len(temp)
        # None 값은 유효/무효 판단 불가라서 제외하고 계산할지 여부는 선택
        # 여기서는 True/False만 합산 대상으로 사용
        valid_mask = temp['Is_valid'].isin([True, False])
        valid_known = temp[valid_mask]

        valid_date = (valid_known['Is_valid'] == True).sum()
        invalid_date = (valid_known['Is_valid'] == False).sum()

        # 총 데이터 기준 / 아니면 known 기준 중 택일
        # 여기서는 원래 코드에 맞춰 total_date로 나눔
        date_validity = valid_date / total_date * 100 if total_date > 0 else 0
        date_validity = round(date_validity, 2)

        summary_df.at[n, 'Original_table_name'] = table_name
        summary_df.at[n, 'Variable_name'] = variable_name
        summary_df.at[n, 'Total_date'] = total_date
        summary_df.at[n, 'Invalid_date'] = invalid_date
        summary_df.at[n, 'Date_Validity_(%)'] = date_validity

        valid_results_df = pd.concat([valid_results_df, temp], axis=0)

        gc.collect()

    return valid_results_df, summary_df

if __name__ == '__main__':
    print('<LYDUS - Date Validity (Llama3 local)>')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    quiq_path = config.get('quiq_path')
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')  # ex) "/content/Meta-Llama-3.1-8B-Instruct"
    api_key = config.get('api_key')      # local에서는 사용 안 함

    os.makedirs(save_path, exist_ok=True)

    quiq = pd.read_csv(quiq_path)

    valid_results_df, summary_df = get_date_validity(
        quiq=quiq,
        model=model_ver,
        api_key=api_key
    )

    total_date = summary_df['Total_date'].sum()
    invalid_date = summary_df['Invalid_date'].sum()
    valid_date = total_date - invalid_date
    date_validity = valid_date / total_date * 100 if total_date > 0 else 0
    date_validity = round(date_validity, 2)

    with open(os.path.join(save_path, 'date_validity_total.txt'), 'w', encoding='utf-8') as file:
        file.write(f'Date Validity (%) = {date_validity}\n')
        file.write(f'Total dates = {total_date}\n')
        file.write(f'Invalid dates = {invalid_date}\n')

    valid_results_df.to_csv(os.path.join(save_path, 'date_validity_detail.csv'), index=False)
    summary_df.to_csv(os.path.join(save_path, 'date_validity_summary.csv'), index=False)

    print('\n<SUCCESS>')
