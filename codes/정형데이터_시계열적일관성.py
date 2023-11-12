#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

def time_series(csv):
    col_name = '변수명'
    date_col_name = '기록날짜'
    df=pd.read_csv(csv)
    #변수 선택
    #target_df = df[df['변수category'] == 'Laboratory']
    target_df = df
    
    target_df[date_col_name] = pd.to_datetime(target_df[date_col_name], format='mixed', errors='coerce')


    unique_values = target_df[col_name].dropna().unique()
 
    yearly_counts = target_df.groupby([col_name, target_df[date_col_name].dt.year]).size().unstack()
    return yearly_counts

def detect_change_points_v3(data, variable_name, p=1, d=1, q=0, mul=2, min_percentile=10):
    
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
        if (change_points[i] < np.mean(data)*0.2) or(change_points[i] < np.percentile(data, min_percentile) ):  # 추가된 조건
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
        # 첫 번째 그래프
        ax1 = plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 위치
        ax1.set_ylim(0, max(data) + max(data) * 0.1)
        ax1.plot(data, label='real data')
        ax1.plot(forecast_values, label='predicted data', linestyle='dashed')
        ax1.scatter(high_value_indices, [data[i] for i in high_value_indices], color='red', label='ChangePoint')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        ax1.set_title('ARIMA')
        ax1.legend()

        # 두 번째 그래프
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)  # y축 공유
        ax2.plot(change_points, label='Change Points')
        ax2.scatter(high_value_indices, [change_points[i] for i in high_value_indices], color='red', label='diff_peak')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Diff between prediction')
        ax2.set_title('Change Points with threshold')
        ax2.legend()

        # 그래프 표시 및 파일로 저장
        plt.tight_layout()
        plt.savefig(f"{variable_name}_time_consistency.png")
        plt.show()
    

    
    def show_detail():
        print(f"\n\n분석 대상 변수명: {variable_name}\n")
        draw_and_save_graphs()
        print(f"{variable_name}에 대한 change point의 개수: {len(high_value_indices)}\n")

    show_detail()
    return high_value_indices

def calculate_time_consistency(csv, p=1, d=1, q=0, mul=2, min_percentile=10):
    df_result = time_series(csv)
    df_result.fillna(0, inplace=True)
    index_list = df_result.index

    count_with_change_points = 0

    for idx in index_list:
        data = pd.Series(df_result.loc[idx])
        change_points_indices = detect_change_points_v3(data.values, idx, p, d, q, mul, min_percentile)

        if change_points_indices:  # changepoints가 발생한 경우
            count_with_change_points += 1

    percentage_with_change_points = (count_with_change_points / len(index_list))
    time_series_consistency = 1 - percentage_with_change_points

    print(f"시계열적 일관성: {time_series_consistency:.2f}")

    

