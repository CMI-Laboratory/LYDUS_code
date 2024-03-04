#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



# 매핑데이터 이용 변수 수출
def extract_data_using_mapping(df, min_patient_count = 100):
    df_filtered = df[df['Patient_number'].notnull() & df['Record_datetime'].notnull()]
    variable_counts = df_filtered.groupby('Variable_name')['Patient_number'].count().sort_values(ascending=False)
    df_valid_counts = df_filtered[df_filtered['Variable_name'].isin(variable_counts[variable_counts >= min_patient_count].index)]
    time_series_mapping = ['ICD9_Dx', 'ICD9_Px', 'ICD10_Dx', 'ICD10_Px', 'Events']
    df_mapping = df_valid_counts[df_valid_counts['Mapping_info_1'].isin(time_series_mapping) | df_valid_counts['Mapping_info_2'].str.contains('Drug', case=False, na=False)] 
    return df_mapping



#지표 계산
def time_series(csv):
    col_name = 'Variable_name'
    date_col_name = 'Record_datetime'
    df=pd.read_csv(csv)

    #GPT 이용한 변수 추출
    #target_df = extract_target_data(df)
    #매핑 이용한 변수 수출
    target_df = extract_data_using_mapping(df)
    
    target_df[date_col_name] = pd.to_datetime(target_df[date_col_name], errors='coerce')

    unique_values = target_df[col_name].dropna().unique()
 
    yearly_counts = target_df.groupby([col_name, target_df[date_col_name].dt.year]).size().unstack()
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
        if i == 0 or i == len(change_points) - 1:  # 첫 번째 또는 마지막 데이터 포인트인 경우
            if i == len(change_points) - 1:  # 마지막 포인트
                return change_points[i] > 2 * change_points[i-1]
            return False
        if (change_points[i] < np.mean(data)*0.2) or(change_points[i] < np.percentile(data, min_percentile) or change_points[i] <5 ):  # 추가된 조건
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
        ax2.set_title('Change Points with threshold')
        ax2.legend()

        ax2.set_xticks(years)  # 년도를 x축 눈금으로 설정
        ax2.set_xticklabels(years, rotation=45)

        # 그래프 표시 및 파일로 저장
        plt.tight_layout()
        #plt.savefig(f"{variable_name}_time_consistency.png")
        plt.show()
    

    
    def show_detail():
        print(f"\n\n분석 대상 변수명: {variable_name}\n")
        draw_and_save_graphs()
        print(f"{variable_name}에 대한 change point의 개수: {len(high_value_indices)}\n")
    #그래프 출력
    show_detail()
    return high_value_indices

def calculate_time_consistency(csv, p=1, d=1, q=0, mul=2, min_percentile=10):
    df_result = time_series(csv)
    df_result.fillna(0, inplace=True)
    years = df_result.columns
    

    count_with_change_points = 0

    for idx in df_result.index:
        data = df_result.loc[idx].values
        change_points_indices = detect_change_points_v3(data, idx, years, p, d, q, mul, min_percentile)

        if change_points_indices:  # changepoints가 발생한 경우
            count_with_change_points += 1

    percentage_with_change_points = (count_with_change_points / len(df_result.index))
    time_series_consistency = 1 - percentage_with_change_points
    print(f"변화점 있는 변수 개수 / 전체 변수 개수: {count_with_change_points} / {len(df_result.index)} ")
    print(f"시계열적 일관성: {time_series_consistency:.2f}")

calculate_time_consistency(csv_path)






