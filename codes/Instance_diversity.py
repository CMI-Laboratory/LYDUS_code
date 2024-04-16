#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
if not csv_path:
    print("CSV path not found in the configuration file.")
    sys.exit(1)
    
    
def calculation(instance_values):
    filtered_values = [value for value in instance_values if pd.notnull(value) and value != 0]
    if not filtered_values:
        return None, None, None, Counter()  # Avoid division by zero
    
    total_num = sum(filtered_values)
    if total_num == 0:
        return 0, 0, 0, Counter()  # Handling divide by zero
    
    instance_counter = Counter(filtered_values)
    
    probabilities = [count / total_num for _, count in instance_counter.items()]
    shannon_diversity = -sum(prob * log(prob) for prob in probabilities if prob > 0)
    simpson_diversity = sum(prob**2 for prob in probabilities)
    instance_diversity = (len(filtered_values) / total_num) * 100
    simpson_diversity_score = (1 - simpson_diversity) * 100
    return instance_diversity, shannon_diversity, simpson_diversity_score, instance_counter
    

def calculate_class_instance_counts(class_values, instance_values):
    df = pd.DataFrame({"Class": class_values, "Instance": instance_values})
    class_instance_counts = df.groupby(["Class", "Instance"]).size().reset_index(name="Instance_Count")
    class_counts = df.groupby("Class").size().reset_index(name="Total_Count")
    
    weighted_instance_diversities = []
    weighted_simpson_diversities = []
    total_items = len(class_values)

    def print_detail(class_value,instance_diversity,simpson_diversity_score, shannon_diversity):
            print(f"Class name: {class_value}")
            print(f"Instance Diversity: {instance_diversity:.2f}%")       
            print(f"Simpson Diversity: {simpson_diversity_score:.2f}%")
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
    
    print(f"Weighted Average Instance Diversity: {weighted_avg_instance_diversity:.2f}%")
    print(f"Weighted Average Simpson Diversity: {weighted_avg_simpson_diversity:.2f}%\n")
    
def calculate_instance_diversity(csv):
    df = pd.read_csv(csv)
    df_target = df[df['Variable_name'].notna() & df['Patient_number'].notna()]
    if df_target.empty:
        print("Filtered data frame is empty.")
        return
    
    class_values = df_target['Variable_name']
    instance_values = df_target['Patient_number']
    calculate_class_instance_counts(class_values, instance_values)
    
calculate_instance_diversity(csv_path)

