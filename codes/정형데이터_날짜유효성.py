#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install openai')
get_ipython().system('pip install langchain')

import os

#@markdown https://platform.openai.com/account/api-keys
OPENAI_API_KEY = "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" #@param {type:"string"}
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import pandas as pd
import datetime
from dateutil.parser import parse
import langchain
from langchain.llms import OpenAI

llm = OpenAI(model_name='text-davinci-003', temperature=0.9)

date_formats = ["%Y년 %m %d일", "%Y년 %m월 %d일", "%Y년 %m월 %d", "%Y %m월 %d", "%Y %m %일", "%Y년 %B %d"]

def is_valid_date(date_string):
    if isinstance(date_string, pd.Timestamp):
        return True
    try:
        parse(date_string)
        return True
    except ValueError:
        return False

def valid_date_custom(date_string, formats):
    for date_format in formats:
        try:
            datetime.datetime.strptime(date_string, date_format)
            return True
        except ValueError:
            pass
    return False

def validate_date_entry(df, date_string, date_col_name):
    if pd.isna(date_string):
        return
    #유효한 날짜 필터링
    if is_valid_date(date_string) or valid_date_custom(date_string, date_formats):
        #print(f"{date_string}은(는) 유효한 날짜입니다.")
        df.loc[df[date_col_name] == date_string, 'date_valid'] = 0
    #유효하지 않은 날짜 GPT에 검증
    else:
        #print(f"{date_string}이(가) 유효한 날짜인지 검증하겠습니다.")
        user_response = llm(f"Is {date_string} a valid date? Please answer with 'yes' or 'no'.").strip()
        if "no" in user_response.lower():
            df.loc[df[date_col_name] == date_string, 'date_valid'] = 1
            #print(f"{date_string}은(는) 유효하지 않은 날짜입니다.")
        else:
            df.loc[df[date_col_name] == date_string, 'date_valid'] = 0
            #print(f"{date_string}은(는) 유효한 날짜입니다.")

def validate_dates(csv_file_name):
    df = pd.read_csv(csv_file_name)
    date_col_name = '기록날짜'
    if 'percentage' in df.columns:
        df['percentage'] = None
    df['date_valid'] = -1  # 초기값 -1로 설정. NA 항목은 이 값 그대로 유지됩니다.

    for date_string in df[date_col_name]:
        validate_date_entry(df, date_string, date_col_name)
    
    valid_entries = df[df['date_valid'] != -1]
    date_validation = (valid_entries['date_valid'] == 0).mean() 
    date_validation = round(date_validation, 2)
    print("날짜 유효성:",date_validation)
    
    # 유효하지 않은 날짜들을 CSV 파일로 저장
    invalid_dates_df = df[df['date_valid'] == 1]
    invalid_dates_df = invalid_dates_df.drop('date_valid', axis=1)
    invalid_dates_df.to_csv('invalid_dates.csv', index=False)
    
    

