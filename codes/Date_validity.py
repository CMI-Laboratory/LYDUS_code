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
    
save_path = config_data.get('save_path')
if not save_path:
    print("Save path not found in the configuration file.")
    sys.exit(1)
    


OPENAI_API_KEY = openai_api_key

# Setting Client
client = openai.Client(api_key=OPENAI_API_KEY)
os.environ["OPENAI_API_TYPE"] = "azure"
warnings.filterwarnings('ignore')



# Setting Client
client = openai.Client(api_key=OPENAI_API_KEY)
os.environ["OPENAI_API_TYPE"] = "azure"
warnings.filterwarnings('ignore')

def extract_date_data_mapping(df):
    # 데이터프레임의 인덱스를 'original_index'로 저장
    df['original_index'] = df.index

    # 'Record_datetime' 칼럼을 날짜 변수로 취급
    record_dates_df = df[['original_index', 'Record_datetime']].dropna()
    record_dates_df['Variable_name'] = 'Record_datetime'
    record_dates_df.rename(columns={'Record_datetime': 'Date_value'}, inplace=True)
    
    # 'Mapping_info1'이 'date'인 데이터 추출
    date_mapping_df = df[df['Mapping_info_1'].str.contains('date', case=False, na=False)]
    date_mapping_df = date_mapping_df[['original_index', 'Variable_name', 'Value']]
    date_mapping_df.rename(columns={'Value': 'Date_value'}, inplace=True)

    # 데이터 병합
    combined_date_df = pd.concat([record_dates_df, date_mapping_df])

    # 필요한 칼럼만 선택
    return combined_date_df[['original_index', 'Variable_name', 'Date_value']]

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

def valid_date_custom(date_string):
    formats = ["%Y년 %m %d일", "%Y년 %m월 %d일", "%Y년 %m월 %d", "%Y %m월 %d", "%Y %m %일", "%Y년 %B %d"]
    for date_format in formats:
        try:
            datetime.datetime.strptime(date_string, date_format)
            return True
        except ValueError:
            pass
    return False

def validate_date_entry(row):
    date_string = row['Date_value']
    #유효한 날짜 필터링
    if pd.isna(date_string):
        return None  # NA Value은 무시
    if is_valid_date(date_string) or valid_date_custom(date_string):
        return None  # 유효한 날짜는 skip
    #유효하지 않은 날짜
    else:
        user_content = f"{date_string}"
        try:
            result = gpt_chat(client, system_content, user_content)
            if "no" in result:
                return True
            else:
                return None
        except Exception as e:
            print("Failed to call GPT service:", e)
            return True

def save_data(df, variable_name, save_path, total_dates, results):
    sanitized_variable_name = variable_name.replace('/', '_')
    
    output_folder = os.path.join(save_path, 'Date_validation')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path_csv = os.path.join(output_folder, f'{sanitized_variable_name}_invalid_dates.csv')
    if df is not None and not df.empty:
        df.to_csv(output_path_csv, index=False)
    

    invalid_dates = len(df)
    date_validation_percentage = 100-((invalid_dates / total_dates) * 100) if total_dates > 0 else 0
    print(f'{variable_name} Date validation: {date_validation_percentage:.2f}%\n')
    results.append({
                'Variable_name': variable_name,
            'Total number of dates': total_dates,
            'Number of invalid dates': invalid_dates,
            'Date_validity': f'{date_validation_percentage:.2f}%'
        })
    """
    output_path_txt = os.path.join(output_folder, f'{sanitized_variable_name}_text.txt')
    with open(output_path_txt, 'w') as file:
        file.write(f'Date_validation: {date_validation_percentage:.2f}%\n')
        file.write(f'Total number of dates: {total_dates}\n')
        file.write(f'Number of invalid dates: {invalid_dates}\n')
    """

def validate_dates(csv_file_name, save_path):
    df = pd.read_csv(csv_file_name)
    final_date_df = extract_date_data_mapping(df)
    
    if final_date_df.empty:
        print("No date data available.")
        return
    
    final_date_df['Invalid'] = final_date_df.apply(lambda row: validate_date_entry(row), axis=1).notna()
    
    results = []
    
    invalid_indexes = final_date_df[final_date_df['Invalid'] == True].index
    invalid_date_df = df.loc[invalid_indexes]
    
    # Save variable-specific data
    for variable_name in final_date_df['Variable_name'].unique():
        variable_specific_data = invalid_date_df[invalid_date_df['Variable_name'] == variable_name]
        save_data(variable_specific_data, variable_name, save_path, len(final_date_df[final_date_df['Variable_name'] == variable_name]),results)
    # Save overall invalid data
    #invalid_data = final_date_df[final_date_df['Invalid']]
    save_data(invalid_date_df, 'Total', save_path, len(final_date_df),results)
    
    
    result_df = pd.DataFrame(results)
    result_csv_path = os.path.join(save_path, 'Date_validation_total_result_csv.csv')
    result_df.to_csv(result_csv_path, index=False)


    #0723
    # 파이 차트 그리기 위한 준비
    total_dates_num = len(final_date_df)
    invalid_dates_num = len(invalid_indexes)
    valid_dates_num = total_dates_num - invalid_dates_num
    #print(total_dates_num, invalid_dates_num, valid_dates_num )
    
    # 파이 차트 데이터 준비
    labels = ['Valid Dates', 'Invalid Dates']
    sizes = [valid_dates_num, invlaid_dates_num]
    colors = ['green', 'red']
    explode = (0.1, 0)  # 유효한 날짜 부분을 강조

    # 파이 차트 그리기
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # 차트 타이틀 설정
    plt.title('Percentage of Valid and Invalid Dates in Dataset')

    # 차트 보여주기
    plt.show()

    
    
        
validate_dates(csv_path, save_path)

