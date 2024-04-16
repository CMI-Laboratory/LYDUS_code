#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime
import openai
import os
from dateutil.parser import parse
import langchain
from langchain.llms import OpenAI
import warnings
import yaml
import sys

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  
        return yaml.safe_load(file)

# Check if the script is provided with the correct number of command-line arguments
if len(sys.argv) < 2:
    print("Usage: python script_name.py <config_path>")
    sys.exit(1)

# Get the config file path from the command-line arguments
config_path = sys.argv[1]

# Read configuration from config.yml
config_data = read_yaml(config_path)

openai_api_key = config_data.get('open_api_key')
if not openai_api_key:
    print("OpenAI API key not found in the configuration file.")
    sys.exit(1)

csv_path = config_data.get('csv_path')
if not csv_path:
    print("CSV path not found in the configuration file.")
    sys.exit(1)

OPENAI_API_KEY = openai_api_key

# Setting Client
client = openai.Client(api_key=OPENAI_API_KEY)
os.environ["OPENAI_API_TYPE"] = "azure"
warnings.filterwarnings('ignore')


# target 변수 추출 함수
def extract_date_data(df):
    #변수명에 'date 관련 단어' 포함
    date_words = 'date, time, 날짜, 시간, 일시, 일자, 생년월일, 생일'
    date_pattern = '|'.join(date_words.split(', '))
    
    #정규표현식
    date_regex = (
        r"\d{4}(년|[-./\s])\d+(월|[-./\s])\d+(일|st|nd|rd|th|[-./\s])|"  
        r"\d+(일|st|nd|rd|th|[-./\s])\d+(월|[-./\s])\d{4}(년|)|"  
        r"\d+(월|[-./\s])\d+(일|st|nd|rd|th|[-./\s])\d{4}(년|)|"  
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s?"
    )
    
    #1) 'Record_datetime' 데이터와 인덱스 추출
    record_dates_df = df[['Record_datetime']].dropna().copy()
    record_dates_df['Original_index'] = record_dates_df.index
    record_dates_df.rename(columns={'Record_datetime': 'Date_value'}, inplace=True)
    
    #2) 'Variable_type'이 'time' 또는 'date' 포함한 데이터와 인덱스 추출
    time_or_date_type_df = df[df['Variable_type'].str.contains('time|date', case=False, na=False) & df['Value'].notna() & df['Variable_type'].notna()][['Value']].copy()
    #time_or_date_type_df = df[df['Variable_type'].str.contains('time|date', case=False) & df['Value'].notna()][['Value']].copy()
    time_or_date_type_df['Original_index'] = time_or_date_type_df.index
    time_or_date_type_df.rename(columns={'Value': 'Date_value'}, inplace=True)
        
    #단위가 없고, Value에 문자열 갯수 10 이하, digit 6개 이상인 행 선택
    df_filtered = df[df['Variable_type'].str.contains('string|str', case=False, na = False)]
    df_filtered = df_filtered[df_filtered['Unit'].isnull()]
    df_filtered = df_filtered[df_filtered['Value'].str.replace('[\d\W]+', '', regex=True).str.len() < 10]
    df_filtered = df_filtered[df_filtered['Value'].str.contains(r'(\d\D*){6,}', case=False, na=False)]

    #3) 'Value'에 정규표현식 포함
    date_regex_df = df_filtered[df_filtered['Value'].str.contains(date_regex, case=False, regex=True) & df['Value'].notna()][['Value']].copy()
    date_regex_df['Original_index'] = date_regex_df.index
    date_regex_df.rename(columns={'Value': 'Date_value'}, inplace=True)

    #4) 변수명에 'date 관련 단어 포함'인 것 중, Value에 특수기호가 하나만 있는 것 제외
    string_values_df = df_filtered[df_filtered['Variable_name'].str.contains(date_pattern, case=False, na=False) & df_filtered['Value'].notna()].copy()
    string_values_df = string_values_df[~string_values_df['Value'].astype(str).str.contains(r'^[^\W\s]*\W[^\W\s]*$', na=False, regex=True)]
    string_values_df['Original_index'] = string_values_df.index
    string_values_df.rename(columns={'Value': 'Date_value'}, inplace=True)
    
    # 네 데이터프레임 결합
    combined_date_df = pd.concat([record_dates_df, time_or_date_type_df, string_values_df, date_regex_df])
    
    # 'original index'와 'date value' 열을 가진 새로운 데이터프레임 생성
    final_date_df = combined_date_df[['Original_index', 'Date_value']].reset_index(drop=True)

    #최종 필터링
    final_date_df = final_date_df.dropna(subset=['Date_value'])
    final_date_df = final_date_df.drop_duplicates()
    
    return final_date_df

def extract_date_data_mapping(df):
    
    #1) 'Record_datetime' 데이터와 인덱스 추출
    record_dates_df = df[['Record_datetime']].dropna().copy()
    record_dates_df['Original_index'] = record_dates_df.index
    record_dates_df.rename(columns={'Record_datetime': 'Date_value'}, inplace=True)
    
    
    #2) Mapping_info_1 이 date 인 것
    date_mapping_df = df[df['Mapping_info_1'].str.contains('date', case = False, na = False)]
    date_mapping_df['Original_index'] = date_mapping_df.index
    date_mapping_df.rename(columns={'Value': 'Date_value'}, inplace=True)
    
    combined_date_df = pd.concat([record_dates_df, date_mapping_df])
    
    # 'original index'와 'date value' 열을 가진 새로운 데이터프레임 생성
    final_date_mapping_df = combined_date_df[['Original_index', 'Date_value']].reset_index(drop=True)

    #최종 필터링
    final_date_mapping_df = final_date_mapping_df.dropna(subset=['Date_value'])
    final_date_mapping_df = final_date_mapping_df.drop_duplicates()
    final_date_mapping_df = final_date_mapping_df.reset_index(drop=True)
    
    return final_date_mapping_df

system_content = f"""We will evaluate the quality of the medical data.I want to verify that the date given is valid.
    Please answer with 'yes' or 'no'.
    No other answer than 'yes' or 'no'.
    """
def gpt_chat(client, system_content, user_content, temperature=0, max_tokens=1000, n=1):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=n
    )
    return [choice.message.content for choice in response.choices]

#Timestamp 형식 or date로 parising 가능한지
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

def validate_date_entry(row):
    formats = ["%Y년 %m %d일", "%Y년 %m월 %d일", "%Y년 %m월 %d", "%Y %m월 %d", "%Y %m %일", "%Y년 %B %d"]
    date_string = row['Date_value']
    #유효한 날짜 필터링
    if pd.isna(date_string):
        return None  # NA Value은 무시
    if is_valid_date(date_string) or valid_date_custom(date_string, formats):
        return None  # 유효한 날짜는 skip
    #유효하지 않은 날짜
    else:
        user_content = f"{date_string}"
        try:
            result = gpt_chat(client, system_content, user_content)
            if "no" in result:
                return row
            else:
                return None
        except Exception as e:
            print("Failed to call GPT service:", e)
            return None
        
    
def validate_dates(csv_file_name):
    

    #날짜 변수명만 추출
    df = pd.read_csv(csv_file_name) 
    #final_date_df = extract_date_data(df)
    final_date_df = extract_date_data_mapping(df)
    
    if final_date_df.empty:
        print("No date data available.")
        return None
    
    invalid_indexes = []
    
    # 각 행에 대해 날짜 유효성 검증 수행
    for index, row in final_date_df.iterrows():
        invalid_row = validate_date_entry(row)
        if invalid_row is not None:
            invalid_indexes.append(row['Original_index'])
    
    
    # 원본 df에서 유효하지 않은 날짜를 가진 행들만 필터링하여 새로운 데이터프레임 생성
    invalid_date_df = df.loc[df.index.isin(invalid_indexes)]
    
    # 유효하지 않은 날짜 데이터프레임 저장
    invalid_date_df.to_csv('invalid_dates.csv', index=False)

    #유효성 계산
    date_validation = (len(invalid_date_df))/(len(final_date_df))
    
    #유효하지 않은 데이터 수 , 전체 데이터 수 출력
    print('Number of invalid dates / Total number of dates : ', len(invalid_date_df),'/', len(final_date_df))
    date_validation_percentage = (1 - round(date_validation, 4)) * 100
    print(f"Date_validation: {date_validation_percentage:.2f}%")
    

    return invalid_date_df
    
        
validate_dates(csv_path)

