#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import warnings
import ast
import openai
import langchain
from langchain.llms import OpenAI
import numpy as np
import yaml
import sys

warnings.filterwarnings(action='ignore')


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

# Read new configuration options
save_output = config_data.get('save_output', False)  # Defaults to False if not specified
show_details = config_data.get('show_details', True)  # Defaults to True if not specified

OPENAI_API_KEY = openai_api_key
csv_path = config_data.get('csv_path')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_TYPE"] = "azure"
client = openai.Client(api_key=OPENAI_API_KEY)



def filter_dataframe(df):
    # 조건 1: 값의 unique 개수가 2~50인 변수명 필터링
    df = df.dropna(subset=['Value'])
    grouped = df.groupby('Variable_name')['Value'].nunique().reset_index()
    filtered_variables = grouped[(grouped['Value'] > 1) & (grouped['Value'] <= 50)]['Variable_name'].tolist()
    df_filtered_by_value_count = df[df['Variable_name'].isin(filtered_variables)]
    
    # 조건 2: 변수 타입이 string인 행만 필터링
    df_filtered_by_type = df_filtered_by_value_count[df_filtered_by_value_count['Variable_type'].str.contains('string|str', case=False, na=False)]
    
    # 조건 3: 값이 특수기호나 숫자로만 이루어진 행 제외
    df_filtered_by_content = df_filtered_by_type[~df_filtered_by_type['Value'].astype(str).str.match('^[0-9\W]*$')]
    
    # 조건 4: 값의 길이가 50자 이하인 행만 필터링
    df_filtered_by_length = df_filtered_by_content[df_filtered_by_content['Value'].astype(str).str.len() < 50]
    
    # 최종 필터링: 다시 한번 값의 unique 개수가 2~50인 변수명으로 필터링
    grouped_final = df_filtered_by_length.groupby('Variable_name')['Value'].nunique().reset_index()
    final_filtered_variables = grouped_final[(grouped_final['Value'] > 1) & (grouped_final['Value'] <= 50)]['Variable_name'].tolist()
    final_df = df_filtered_by_length[df_filtered_by_length['Variable_name'].isin(final_filtered_variables)]
    
    return final_df


system_content = f"""We will evaluate the quality of the medical data. I will give you list of items. 
    I want to classify several items according to their meaning like below examples. 
    Give me full item name, Don't change the name of the item I gave you
    
    Example 1
    Input: ['F', 'M']
    Output:
    category1: 'F'
    category2: 'M'

    Example2 
    Intput: ['F', 'M', '여', 'Female', '남자', 'Male', '여자', '남']
    Output:
    category1: 'F', 'Female', '여', '여자'
    category2: '남', '남자', 'M', 'Male'
    
    Example 3
    Input: ['ENG','english','KOR','korean', 'SPN']
    Output:
    category1: 'ENG','english'
    category2: 'KOR','korean'
    category3: 'SPN'
    

    Give me only a result without any comment. No other comment excetp lists.
    Output format should be a format like below. 
    <Output Format>
    category1: 'A1','A2'
    category2: 'B1','B2'
    category3: 'C1'
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

def parse_gpt_response_by_line(response, unique_array):
    # GPT 응답을 줄별로 파싱하여 2D 배열 생성
    lines = response.strip().split('\n')
    array_2d = []
    for line in lines:
        # 각 줄에서 ':' 문자가 있는지 확인
        if ':' in line:
            # ':'로 구분된 두 번째 요소(항목 리스트)를 추출하고 쉼표로 분리
            items = line.split(':')[1].strip().split(',')
            # 주어진 unique_array에 있는 단어만을 포함하는 항목만 추출
            filtered_items = [item.strip().strip("'") for item in items if item.strip().strip("'") in unique_array]
            if filtered_items:  # 필터링된 항목이 비어있지 않은 경우에만 추가
                array_2d.append(filtered_items)
    return array_2d


def get_category_counts(unique_array):
    
    user_content = ', '.join(unique_array.tolist())
    result = gpt_chat(client, system_content, user_content)
    response_text = result[0] if result else ""
    
    #print(response_text)
    array_2d = parse_gpt_response_by_line(response_text, unique_array)
    
    category_counts = {}
    for category in array_2d:
        category_counts[category[0]] = len(category)

    return array_2d, category_counts



def calculate_inner_consistency(array_2d, df):
    total_weighted_percentage = 0
    total_weight = 0
    
    for line in array_2d:  # lines 대신 array_2d 사용
        count_by_col = df['Value'].apply(lambda x: x if x in line else None).value_counts().dropna()
        
        for value, count in count_by_col.items():
            max_percentage = count / count_by_col.sum()
            weighted_percentage = max_percentage * count
            total_weighted_percentage += weighted_percentage
        
        total_weight += count_by_col.sum()
    if total_weight == 0:
        return 0  #오류 처리
    else:
        return total_weighted_percentage / total_weight    
    


def show_detail(variable_name, consistency_value, array_2d, category_counts, save_output=False, show_details=True, file=None):
    lines = []  # 출력할 내용을 저장할 리스트
    detail_line = f"<<<{variable_name}의 일관성 detail>>>\n변수명 \"{variable_name}\"의 일관성: {consistency_value}\n"
    lines.append(detail_line)

    for category in array_2d:
        category_name = category[0]
        count = category_counts[category_name]
        items = ', '.join(category)
        line = f"{category_name}: {count}개 - 항목: {items}\n"
        lines.append(line)
    
    if save_output and file is not None:
        for line in lines:
            file.write(line)
        file.write("\n")  # 세부 사항 출력 후 빈 줄 추가
    
    if show_details:
        for line in lines:
            print(line, end='')
        print("\n")  # 세부 사항 출력 후 빈 줄 추가


def calculate_consistency(csv_file_name, save_output=False, show_details=True):
    df = pd.read_csv(csv_file_name)
    target_df = filter_dataframe(df)
    target_vars = target_df['Variable_name'].unique()
    # 예로 5개만 선택
    target_vars = np.random.choice(target_vars, size=5, replace=False)
    
    total_consistency = 0
    count_vars = 0

    output_file = None
    if save_output:
        output_file = open('Voca_consistency_details.txt', 'w', encoding='utf-8')
    
    for TARGET_VARIABLE in target_vars:
        unique_array = target_df[target_df['Variable_name'] == TARGET_VARIABLE]['Value'].dropna().unique()
        array_2d, category_counts = get_category_counts(unique_array)
        consistency_value = calculate_inner_consistency(array_2d, target_df[target_df['Variable_name'] == TARGET_VARIABLE])

        show_detail(TARGET_VARIABLE, consistency_value, array_2d, category_counts, save_output, show_details, output_file)

        total_consistency += consistency_value
        count_vars += 1

    if count_vars > 0:
        average_consistency = total_consistency / count_vars
        summary = f'전체 평균 일관성: {average_consistency:.4f}\n'
    else:
        summary = '계산할 변수명이 없습니다.\n'

    print(summary)  # 프롬프트에 항상 출력
    if save_output and output_file is not None:
        output_file.write(summary)  # 파일에 저장
        output_file.close()  # 파일 작업 종료


            
calculate_consistency(csv_path, save_output=save_output, show_details=show_details)



