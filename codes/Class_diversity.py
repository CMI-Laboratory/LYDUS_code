#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from collections import Counter
from math import log
import matplotlib.pyplot as plt
import warnings
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
csv_path = config_data.get('csv_path')



def filter_categorical(df):
    #범주형 데이터만 선택
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


def filter_data(df):
    # 클래스 매핑 조건 정의
    class_mapping = ['ICD9_Dx', 'ICD9_Px', 'ICD10_Dx', 'ICD10_Px']
    
    # 클래스 매핑에 따른 데이터 필터링
    class_mapping_df = df[df['Mapping_info_1'].isin(class_mapping) | df['Mapping_info_2'].str.contains('Drug', case=False, na=False)]
    
    # 환자 데이터 구분
    patient_info = df[df['Mapping_info_1']=='Patient_info']
    patient_info_categorical = filter_categorical(patient_info)  # 범주형 데이터 필터링 함수는 정의되어 있어야 함
    patient_info_non_categorical = patient_info[~patient_info['Variable_name'].isin(patient_info_categorical['Variable_name'])]
    
    # class_mapping_df 중 'Mapping_info_1' 별로 'Variable_name'이 하나인 경우 처리
    single_variable_names_in_class_mapping = class_mapping_df.groupby('Mapping_info_1')['Variable_name'].nunique()
    single_variable_names_in_class_mapping = single_variable_names_in_class_mapping[single_variable_names_in_class_mapping == 1].index.tolist()
    
    # 해당하는 모든 행을 class_mapping_df에서 제외
    single_variable_df = class_mapping_df[class_mapping_df['Mapping_info_1'].isin(single_variable_names_in_class_mapping)]
    class_mapping_df = class_mapping_df[~class_mapping_df['Mapping_info_1'].isin(single_variable_names_in_class_mapping)]
    
    # final_df_categorical에 추가
    final_df_categorical = pd.concat([patient_info_categorical, single_variable_df], axis=0, ignore_index=True)
    
    # 최종 데이터프레임 결합
    final_df_variable_name = pd.concat([class_mapping_df, patient_info_non_categorical], axis=0, ignore_index=True)
    
    return final_df_variable_name, final_df_categorical


def calculate_diversity(class_values):
    filtered_values = class_values.dropna()
    class_counter = Counter(filtered_values).most_common()
    class_num = len(class_counter)
    total_num = len(filtered_values)

    probabilities = [count / total_num for _, count in class_counter]
    shannon_diversity = -sum(prob * log(prob) for prob in probabilities)
    simpson_diversity = sum(prob**2 for prob in probabilities)
    class_diversity = (class_num / total_num)
    
    simpson_diversity_score = 1 - simpson_diversity
    
    return class_diversity, shannon_diversity,simpson_diversity_score, class_counter, class_num, total_num

def plot_class_counts(class_counts, title):
    labels, counts = zip(*class_counts)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()  # 그래프 레이아웃 조정
    
    # 파일 이름을 안전하게 만들기 위해 공백과 특수문자를 '_'로 대체
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    filename = f"{safe_title}_class_diversity.png"
    
    #plt.savefig(filename) 
    #plt.show()

def calculate_and_plot_diversity(df, column_name):
    diversity_results = []

    def show_detail(unique_value, simpson_diversity_score, class_counts, class_num, total_num):  
        print(f"{unique_value}:")
        print(f"  Class Simpson Diversity: {simpson_diversity_score}")
        print(f"  Number of Classes / Total Number of Data: {class_num} / {total_num}")
        plot_class_counts(class_counts, f"{unique_value}")
    
    for unique_value in df[column_name].unique():
        filtered_df = df[df[column_name] == unique_value]
        class_values = filtered_df['Value'] if column_name == 'Variable_name' else filtered_df['Variable_name']
        class_diversity, shannon_diversity, simpson_diversity_score, class_counts, class_num, total_num = calculate_diversity(class_values)
        
        diversity_results.append({
            'Variable_name': unique_value,
            'Class_Simpson_diversity': simpson_diversity_score,
            "Number of Classes": class_num,
            "Total Number of Data": total_num,
        })
        
        show_detail(unique_value, simpson_diversity_score, class_counts, class_num, total_num)

    return diversity_results

def calculate_weighted_average_simpson_from_list(diversity_results):
    total_weighted_simpson = 0
    total_count = 0
    
    for result in diversity_results:
        simpson_score = result['Class_Simpson_diversity']
        count = result["Total Number of Data"]
        
        total_weighted_simpson += simpson_score * count
        total_count += count
    
    if total_count > 0:
        weighted_average_simpson = total_weighted_simpson / total_count
    else:
        weighted_average_simpson = 0

    return weighted_average_simpson

def save_diversity_results_to_csv(diversity_results, csv_file_path):
    results_df = pd.DataFrame(diversity_results)
    results_df.to_csv(csv_file_path, index=False)
    print(f"Diversity results saved to {csv_file_path}")


def calculate_class_diversity_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    final_df_variable_name, final_df_categorical = filter_data(df)
    
    # 'Mapping_info_1'과 'Variable_name' 기준으로 다양성 계산 및 시각화
    diversity_by_mapping_info = calculate_and_plot_diversity(final_df_variable_name, 'Mapping_info_1')
    diversity_by_variable_name = calculate_and_plot_diversity(final_df_categorical, 'Variable_name')

    #가중 평균 다양성
    combined_diversity_results = diversity_by_mapping_info + diversity_by_variable_name
    save_diversity_results_to_csv(combined_diversity_results, 'Class_diversity.csv')

    
    weighted_average_simpson = calculate_weighted_average_simpson_from_list(combined_diversity_results)

    print(f"전체 데이터셋에 대한 가중평균 심슨 다양성: {weighted_average_simpson}")


calculate_class_diversity_from_csv(csv_path)


# In[ ]:




