#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from collections import Counter
from math import log
import matplotlib.pyplot as plt
import os
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
if not csv_path:
    print("CSV path not found in the configuration file.")
    sys.exit(1)
    
save_path = config_data.get('save_path')
if not save_path:
    print("Save path not found in the configuration file.")
    sys.exit(1)

# Create the folder path for storing results
results_folder = os.path.join(save_path, 'Class_diversity')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
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

def extract_class_data_using_mapping(df):
    
    # Mapping_info_1 기준으로 필터링
    class_diversity_mapping = ['event', 'diagnosis', 'prescription', 'procedure']
    pattern = '|'.join(class_diversity_mapping)
    df_mapping = df[df['Mapping_info_1'].str.contains(pattern, case=False, na=False)]

    #환자 정보 중 범주형 선택
    patient_info = df[df['Mapping_info_1'].str.contains('Patient_info', case = False, na = False)]
    patient_info_categorical = filter_categorical(patient_info)
    
    # 'Value'의 다양성 고려해야 하는 것: 환자 정보 중 범주형 및 medical code
    medical_code = df[df['Mapping_info_1'].str.contains('code', case=False, na=False)]
    categorical_df = pd.concat([medical_code, patient_info_categorical], axis=0, ignore_index=True)

    # Prescription인 경우에 Mapping_info_2가 Drug이어야 함
    condition_drug = (df_mapping['Mapping_info_1'].str.contains('prescription', case=False, na=False) &
                      df_mapping['Mapping_info_2'].str.contains('drug', case=False, na=False))
    condition_others = ~df_mapping['Mapping_info_1'].str.contains('prescription', case=False, na=False)
    
    # 조건을 만족하는 데이터만 필터링
    df_mapping = df_mapping[condition_drug | condition_others]
    
    

    return df_mapping, categorical_df


def calculate_diversity(class_values):
    filtered_values = class_values.dropna()
    class_counter = Counter(filtered_values).most_common()
    class_num = len(class_counter)
    total_num = len(filtered_values)

    probabilities = [count / total_num for _, count in class_counter]
    shannon_diversity = -sum(prob * log(prob) for prob in probabilities)
    simpson_diversity = sum(prob**2 for prob in probabilities)

    
    #0723 업데이트 (return도 수정하였으니 확인 부탁드립니다)
    class_diversity = (class_num / total_num) *100
    simpson_diversity_score = 100 if class_diversity == 100 else (1 - simpson_diversity) * 100
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
    
    plt.savefig(filename) 
    plt.show()

def calculate_and_plot_diversity(df, column_name):
    diversity_results = []

    def show_detail(unique_value, simpson_diversity_score, class_counts, class_num, total_num):  
        print(f"{unique_value}:")
        print(f"  Class Simpson Diversity: {simpson_diversity_score:.2f}%")
        print(f"  Number of Classes / Total Number of Data: {class_num} / {total_num}")
        #plot_class_counts(class_counts, f"{unique_value}")
    
    for unique_value in df[column_name].unique():
        filtered_df = df[df[column_name] == unique_value]
        class_values = filtered_df['Value'] if column_name == 'Variable_name' else filtered_df['Variable_name']
        class_diversity, shannon_diversity, simpson_diversity_score, class_counts, class_num, total_num = calculate_diversity(class_values)
        
        diversity_results.append({
            'Variable_name': unique_value,
            'Class_Simpson_diversity': f"{simpson_diversity_score:.2f}",
            "Number of Classes": class_num,
            "Total Number of Data": total_num,
        })
        
        show_detail(unique_value, simpson_diversity_score, class_counts, class_num, total_num)

    return diversity_results

    
#0723
def plot_diversity_boxplot(diversity_scores, title, results_folder):
    # 박스플롯 생성
    plt.figure(figsize=(8, 6))
    plt.boxplot(diversity_scores, vert=False)
    plt.title(title)
    plt.xlabel('Simpson Diversity Score')
    plt.yticks([])  # Y축 레이블 제거

    # 박스플롯 저장
    file_path = os.path.join(results_folder, 'simpson_class_diversity_boxplot.png')
    plt.savefig(file_path, format='png', dpi=300)
    plt.show()
    plt.close()  # 생성된 그림을 닫음
    print(f"Boxplot saved successfully at {file_path}")    


def calculate_weighted_average_simpson_from_list(diversity_results):
    total_weighted_simpson = 0
    total_count = 0
    
    for result in diversity_results:
        simpson_score = float(result['Class_Simpson_diversity'])
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
    
    weighted_average_simpson = calculate_weighted_average_simpson_from_list(diversity_results)
    results_df.loc[len(results_df)] = ['Total_Weighted_Average', f"{weighted_average_simpson:.2f}", None,None ] 
    
    results_df.to_csv(os.path.join(results_folder, csv_file_path), index=False)
    print(f"Diversity results saved to {csv_file_path}")

#0723
def collect_simpson_diversity_scores(diversity_results):
    # Simpson Diversity 값 수집
    simpson_scores = [float(result['Class_Simpson_diversity']) for result in diversity_results]
    return simpson_scores
    
def calculate_class_diversity_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    target_column_df, final_df_categorical = extract_class_data_using_mapping(df)
    
    # 'Mapping_info_1'과 'Variable_name' 기준으로 다양성 계산 및 시각화
    diversity_by_mapping_info = calculate_and_plot_diversity(target_column_df, 'Mapping_info_1')
    diversity_by_variable_name = calculate_and_plot_diversity(final_df_categorical, 'Variable_name')

    #가중 평균 다양성
    combined_diversity_results = diversity_by_mapping_info + diversity_by_variable_name
    save_diversity_results_to_csv(combined_diversity_results, 'Class_diversity_results.csv')
    
    # 0723
    simpson_scores = collect_simpson_diversity_scores(combined_diversity_results)
    plot_diversity_boxplot(simpson_scores, "Simpson Diversity Boxplot", results_folder)
    
    
    #가중 평균 simpson 다양성 출력
    weighted_average_simpson = calculate_weighted_average_simpson_from_list(combined_diversity_results)
    #0723
    print(f"Weighted average Simpson Class diversity: {weighted_average_simpson:.2f}%")


calculate_class_diversity_from_csv(csv_path)

