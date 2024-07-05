#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import os

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

csv_path = config_data.get('csv_path')
if not csv_path:
    print("CSV path not found in the configuration file.")
    sys.exit(1)    

save_path = config_data.get('save_path')
if not save_path:
    print("Save path not found in the configuration file.")
    sys.exit(1)

# Create the folder path for storing results
results_folder = os.path.join(save_path, 'Cross-sectional_Consistency')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
        
    
    
# Read new configuration options
#save_output = config_data.get('save_output', True)  # Defaults to True if not specified
save_output = True

show_details = config_data.get('Cross-sectional_consistency_show_details', True)  # Defaults to True if not specified
#show_details = True 

OPENAI_API_KEY = openai_api_key


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



def llm_chat(client, system_content, user_content, temperature=0, max_tokens=1000, n=1):
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

def parse_llm_response_by_line(response, unique_array):
    # LLM 응답을 줄별로 파싱하여 2D 배열 생성
    lines = response.strip().split('\n')
    array_2d = []
    # unique_array에서 공백을 제거
    unique_array = [item.strip() for item in unique_array]
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


def get_category_counts(df, unique_array, attempt=1):
    user_content = ', '.join(unique_array.tolist())
    result = llm_chat(client, system_content, user_content)
    response_text = result[0] if result else ""
    
    array_2d = parse_llm_response_by_line(response_text, unique_array)
    
    category_counts = {}
    for category in array_2d:
        category_counts[category[0]] = len(category)

    # 유효성 검사: 반환된 분류 내용이 데이터에 실제로 존재하는지 확인
    valid = any(df['Value'].isin([item for sublist in array_2d for item in sublist]))
    if not valid and attempt < 5:  # 재시도 로직
        #print("No valid data matched with LLM categories, retrying classification...")
        return get_category_counts(df, unique_array, attempt + 1)
    
    return array_2d, category_counts, valid



def calculate_inner_consistency(array_2d, df):
    total_weighted_percentage = 0
    total_weight = 0
    
    for line in array_2d:  
        count_by_col = df['Value'].apply(lambda x: x if x in line else None).value_counts().dropna()
        
        total_counts = count_by_col.sum()
        if total_counts > 0:
            for value, count in count_by_col.items():
                max_percentage = count / total_counts
                weighted_percentage = max_percentage * count
                total_weighted_percentage += weighted_percentage
            
            total_weight += total_counts
        else:
            print(f"Warning: No data matched for line items {line} in df['Value'].")
            # 여기서 다시 LLM 호출이 필요할 경우 로직 추가
            return None  # 혹은 적절한 오류 처리

    if total_weight == 0:
        print("Total weight is zero, which indicates no valid data was processed for any categories.")
        return None  # 분모가 0인 경우 None 반환하거나 적절한 값 반환

    return (total_weighted_percentage / total_weight)*100
        
        


def show_detail(array_2d, category_counts, output_file=None):
    for category in array_2d:
        category_name = category[0]
        count = category_counts[category_name]
        items = ', '.join(category)
        line = f"{category_name}: {count} items - Includes: {items}\n"
        
        if output_file:
            output_file.write(line)
        print(line, end='')

def calculate_consistency(csv_file_name, save_output, show_details):
    df = pd.read_csv(csv_file_name)
    
    target_df = filter_dataframe(df)
    target_vars = target_df['Variable_name'].unique()
    #target_vars = np.random.choice(target_vars, size=100, replace=False)
    
    total_consistency = 0
    count_vars = 0
    consistency_results = []

    output_file_path = 'Cross_sectional_consistency_details.txt' if save_output else None
    output_file = None
    
    if output_file_path:
        output_file = open(os.path.join(results_folder, output_file_path), 'w', encoding='utf-8')

    
    try:
        for TARGET_VARIABLE in target_vars:
            unique_array = target_df[target_df['Variable_name'] == TARGET_VARIABLE]['Value'].dropna().unique()
            array_2d, category_counts, valid = get_category_counts(target_df, unique_array)
            if not valid:  # 유효하지 않은 경우 다시 시도
                continue

            consistency_value = calculate_inner_consistency(array_2d, target_df[target_df['Variable_name'] == TARGET_VARIABLE])
            if consistency_value is None:  # 계산 실패 처리
                continue
            
            consistency_results.append({
                'Variable_name': TARGET_VARIABLE,
                'Cross-sectional consistency': f'{consistency_value:.2f}%'
            })
            
            if show_details:
                detail = f'<<<Consistency Details for {TARGET_VARIABLE}>>>\n'
                if output_file:
                    output_file.write(detail)
                print(detail, end='')

                consistency_detail = f'Consistency of "{TARGET_VARIABLE}": {consistency_value:.2f}%\n'
                if output_file:
                    output_file.write(consistency_detail)
                print(consistency_detail, end='')

                # Call show_detail with output_file if needed
                show_detail(array_2d, category_counts, output_file)
                if output_file:
                    output_file.write("\n")
                print("\n", end='')

            total_consistency += consistency_value
            count_vars += 1

        if count_vars > 0:
            average_consistency = (total_consistency / count_vars)
            summary = f'Total average consistency: {average_consistency:.2f}%\n'
            if output_file:
                output_file.write(summary)
            print(summary)
            
            consistency_results.append({
                'Variable_name': 'Total',
                'Cross-sectional consistency': f'{average_consistency:.2f}%'
            })
            
        else:
            no_vars_msg = 'No variables to calculate.\n'
            if output_file:
                output_file.write(no_vars_msg)
            print(no_vars_msg)
        
        if save_output:
            consistency_results_df = pd.DataFrame(consistency_results)
            consistency_results_df.to_csv(os.path.join(results_folder, 'Cross_sectional_consistency_total_result.csv'), index=False)

    finally:
        if output_file:
            output_file.close()
            
calculate_consistency(csv_path, save_output, show_details)          

