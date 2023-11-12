#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from collections import Counter
from math import log
import matplotlib.pyplot as plt



def class_extract(csv_file_name):
    df = pd.read_csv(csv_file_name)
    target_df =df[df['변수명']=='처방코드']
    column_name = "값"
    try:
        class_list = target_df[column_name]
        return class_list
    except KeyError:
        print("Invalid column name. Please try again.")
        return None


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


def plot_class_counts(class_counts):
    labels, counts = zip(*class_counts)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Counts")
    plt.xticks(rotation=45)
    plt.savefig('class_diversity.png')
    plt.show()
    



def calculate_class_diversity_from_csv(csv_file_name):
    class_list = class_extract(csv)
    
    class_diversity, shannon_diversity, simpson_diversity_score, class_counts, class_num, total_num= calculate_diversity(class_list)
    def print_detail():
        print("클래스 개수:", class_num) 
        print("총 데이터의 수:", total_num)
        #print("클래스의 다양성:", class_diversity)
    #print("class의 샤논지수:", shannon_diversity)
    print("class의 simpson 다양성 지수:", simpson_diversity_score)
    
    plot_class_counts(class_counts)

