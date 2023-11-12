#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
from collections import Counter
from math import log
import matplotlib.pyplot as plt

def calculation(instance_values):
    filtered_values = [value for value in instance_values if pd.notnull(value)]
    instance_num = len(filtered_values)
    total_num = sum(filtered_values)
    instance_counter = Counter(filtered_values)  # 수정된 부분

    probabilities = [count / total_num for _, count in instance_counter.items()]  # 수정된 부분
    shannon_diversity = -sum(prob * log(prob) for prob in probabilities)
    simpson_diversity = sum(prob**2 for prob in probabilities)
    instance_diversity = (instance_num / total_num) 
    simpson_diversity_score = 1 - simpson_diversity
    return instance_diversity, shannon_diversity, simpson_diversity_score, instance_counter

def calculate_class_instance_counts(class_values, instance_values):
    df = pd.DataFrame({"Class": class_values, "Instance": instance_values})
    class_instance_counts = df.groupby(["Class", "Instance"]).size().reset_index(name="Instance_Count")
    class_counts = df.groupby("Class").size().reset_index(name="Total_Count")
    
    weighted_instance_diversities = []
    weighted_simpson_diversities = []
    total_items = len(class_values)

    def print_detail(class_value,instance_diversity,simpson_diversity_score, shannon_diversity):
            print(f"클래스: {class_value}")
            print(f"인스턴스 다양성: {instance_diversity}")       
            print(f"인스턴스 심슨지수: {simpson_diversity_score}")
            #print(f"인스턴스 샤논지수: {shannon_diversity}")
            print("\n")

    results_data = []
    
    for _, row in class_counts.iterrows():
        class_value = row["Class"]
        total_count = row["Total_Count"]
        instance_diversity, shannon_diversity, simpson_diversity_score, instance_counter = calculation(
            class_instance_counts[class_instance_counts["Class"] == class_value]["Instance_Count"]
        )
        
        results_data.append({
            'Class': class_value,
            'Instance_Diversity': instance_diversity,
            'Simpson_Diversity_Score': simpson_diversity_score
        })
        
        
        weighted_instance_diversities.append(instance_diversity * total_count)
        weighted_simpson_diversities.append(simpson_diversity_score * total_count)
        
        # 클래스별 세부 지표      
        print_detail(class_value,instance_diversity,simpson_diversity_score, shannon_diversity)
        
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('class_instance_diversity_results.csv', index=False)
    
    weighted_avg_instance_diversity = sum(weighted_instance_diversities) / total_items
    weighted_avg_simpson_diversity = sum(weighted_simpson_diversities) / total_items
    
    print(f"가중 평균 인스턴스 다양성: {weighted_avg_instance_diversity}")
    print(f"가중 평균 인스턴스 심슨 다양성: {weighted_avg_simpson_diversity}\n")
    
  
        
        
def calculate_instance_diversity(csv):
    df = pd.read_csv(csv)
    df_target = df[df['변수명']=='처방코드']
    class_values = df_target['값']
    instance_values = df_target['환자ID']
    calculate_class_instance_counts(class_values, instance_values)
    
