#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openai
import os
import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse

import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

min_percentile = config_data.get('Time_consistency_min_percentile', 10)
mul = config_data.get('Time_consistency_mul', 2)
min_patient_count = config_data.get('Time_consistency_min_patient_count', 100)

save_path = config_data.get('save_path')
if not save_path:
    print("Save path not found in the configuration file.")
    sys.exit(1)

# Create the folder path for storing results
results_folder = os.path.join(save_path, 'Time_series_consistency')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


# 매핑데이터 이용
def extract_data_using_mapping(df, min_patient_count=100):
    
    # Mapping_info_1 기준으로 필터링
    time_series_mapping = ['event', 'diagnosis', 'medical_code', 'prescription', 'procedure']
    pattern = '|'.join(time_series_mapping)
    df_mapping = df[df['Mapping_info_1'].str.contains(pattern, case=False, na=False)]

    # Prescription인 경우에 Mapping_info_2가 Drug이어야 함
    condition_drug = (df_mapping['Mapping_info_1'].str.contains('prescription', case=False, na=False) &
                      df_mapping['Mapping_info_2'].str.contains('drug', case=False, na=False))
    condition_others = ~df_mapping['Mapping_info_1'].str.contains('prescription', case=False, na=False)
    
    # 조건을 만족하는 데이터만 필터링
    df_mapping = df_mapping[condition_drug | condition_others]
    
    # medical_code의 특수성 고려하여 새로운 구조의 df 생성
    new_structure = df_mapping.copy()
    new_structure['Target_column'] = np.where(
        new_structure['Mapping_info_1'].str.contains('code', case=False, na=False),
        new_structure['Value'],
        new_structure['Variable_name']
    )
    
    # 필요한 칼럼만 선택
    final_columns = [
        'Primary_key', 'Variable_ID', 'Original_table_name', 'Target_column',
        'Record_datetime', 'Variable_type', 'Patient_number', 'Admission_number', 
        'Mapping_info_1', 'Mapping_info_2'
    ]
    new_structure = new_structure[final_columns]

    # 'Record_datetime' 열을 datetime 객체로 변환
    new_structure['Record_datetime'] = pd.to_datetime(new_structure['Record_datetime'], errors='coerce')

    # 날짜로 변환에 실패한 데이터 (NaT 포함된 행) 제거
    new_structure = new_structure[new_structure['Record_datetime'].notna()]
    
    # 유니크한 Patient_number의 수를 기준으로 필터링
    df_filtered = new_structure[new_structure['Patient_number'].notnull()]
    variable_counts = df_filtered.groupby('Target_column')['Patient_number'].nunique().sort_values(ascending=False)
    df_valid_counts = df_filtered[df_filtered['Target_column'].isin(variable_counts[variable_counts >= min_patient_count].index)]

    return df_valid_counts


def time_series(df, min_patient_count):
    col_name = 'Target_column'
    date_col_name = 'Record_datetime'

    #target_df = extract_target_data(df)
    target_df = extract_data_using_mapping(df, min_patient_count)
    
    yearly_counts = target_df.groupby([col_name, target_df[date_col_name].dt.year])['Patient_number'].nunique().unstack()
    
    # 3개 년도 이상 있는 변수명만 선택
    valid_columns = yearly_counts.dropna(thresh=3, axis=0).index
    yearly_counts = yearly_counts.loc[valid_columns]
 
    return yearly_counts


def detect_change_points_v3(data, variable_name, years, p=1, d=1, q=0, mul=2, min_percentile=10):
    
    # 예측값을 저장할 리스트
    forecast_values = []
    change_points = []

    # 데이터를 하나씩 추가하고 예측하며 변화점을 찾는 과정 반복
    up_data = np.array([])

    for i in range(len(data)):
        if len(up_data) >=3:  # 충분한 데이터가 모이면
            model = sm.tsa.ARIMA(up_data, order=(p, d, q))
            results = model.fit()
            try:
                forecast = results.forecast(steps=1)[0]
            except IndexError:  # 예외 처리
                forecast = up_data[-1]
        else:
            forecast = data[i] if i < len(data) else up_data[-1]  # 충분한 데이터가 없으면 마지막 값을 사용

        # 예측값을 리스트에 추가
        forecast_values.append(forecast)

        # up_data에 데이터 추가
        up_data = np.append(up_data, data[i])

        # 예측과 실제 값의 차이 계산
        error = abs(data[i] - forecast)
        change_points.append(error)

    # 변화점 탐지 함수 (최소 5% percentile 보다는 커야하고, 양옆의 평균의 2배 이상)
    def is_change_point(i, change_points, data, mul, min_percentile):
        if (change_points[i] < np.percentile(data, min_percentile) or change_points[i] <5 ):  # 추가된 조건
            return False
        if i == 0 or i == len(change_points) - 1:  # 첫 번째 또는 마지막 데이터 포인트인 경우
            if i == len(change_points) - 1:  # 마지막 포인트
                return change_points[i] > 2 * change_points[i-1]
            return False
        avg = (change_points[i-1] + change_points[i+1]) / 2
        return change_points[i] > mul * avg
    
    
    high_value_indices = [i for i in range(len(change_points)) if is_change_point(i, change_points, data, mul, min_percentile)]


    def draw_and_save_graphs():
        #plt.figure(figsize=(30, 10))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10), sharey=True)
        # 원본 데이터 시계열 그래프
        ax1 = plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치
        ax1.set_ylim(0, max(data) + max(data) * 0.1)
        ax1.plot(years, data, label='real data')
        ax1.plot(years, forecast_values, label='predicted data', linestyle='dashed')
        ax1.scatter([years[i] for i in high_value_indices], [data[i] for i in high_value_indices], color='red', label='ChangePoint')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        ax1.set_title('ARIMA')
        ax1.legend()

        # 원본과 예측의 차이 그래프 
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)  # y축 공유
        ax2.plot(years,change_points, label='Change Points')
        ax2.scatter([years[i] for i in high_value_indices], [change_points[i] for i in high_value_indices], color='red', label='ChangePoint')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Diff between prediction')
        ax2.set_title('Frequency diff between prediction')
        ax2.legend()
        ax2.set_xticks(years)  #년도를 x축 눈금으로 설정
        ax2.set_xticklabels(years, rotation=45)

        # 그래프 표시 및 파일로 저장
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"{variable_name}_time_consistency.png"))
        #plt.show()
    

    
    def show_detail():
        print(f"\n\nTarget Variable: {variable_name}\n")
        draw_and_save_graphs()
        print(f"Number of change points for <{variable_name}>: {len(high_value_indices)}\n")

    show_detail()
    return high_value_indices

def save_change_points_to_csv(results, csv_file_path):
    results_df = pd.DataFrame(results)
    results_df.columns = ['Variable_name', 'Change_points_count']
    results_df.to_csv(os.path.join(results_folder, csv_file_path), index=False)
    print(f"Change points saved to {csv_file_path}")

def save_summary_to_text(content, text_file_path):
    with open(os.path.join(results_folder, text_file_path), 'w') as file:
        file.write(content)


def calculate_time_consistency(csv, p=1, d=1, q=0, mul=2, min_percentile=10, min_patient = 100):
    df=pd.read_csv(csv)
    df_result = time_series(df, min_patient)
    df_result.fillna(0, inplace=True)
    years = df_result.columns
    
    if df_result.empty:
        print("No variables with sufficient data points (3 or more years) for analysis.")
        return
    
    change_point_results = []
    count_with_change_points = 0

    for idx in df_result.index:
        data = df_result.loc[idx].values
        change_points_indices = detect_change_points_v3(data, idx, years, p=1, d=1, q=0, mul=2, min_percentile=10)
        change_point_results.append([idx, len(change_points_indices)])
        
        if change_points_indices:  # changepoints가 발생한 경우
            count_with_change_points += 1

    save_change_points_to_csv(change_point_results, 'Variable_change_points.csv')
    
    percentage_with_change_points = (count_with_change_points / len(df_result.index)) * 100
    time_series_consistency = 100 - percentage_with_change_points
    summary = (
        f"Number of variables with change points / Total number of variables: {count_with_change_points} / {len(df_result.index)}\n"
        f"Time series consistency: {time_series_consistency:.2f}%"
    )
    print(summary)
    save_summary_to_text(summary, 'Time_consistency_total_results.txt')
    
    
calculate_time_consistency(csv_path, mul = mul, min_percentile = min_percentile, min_patient= min_patient_count)

