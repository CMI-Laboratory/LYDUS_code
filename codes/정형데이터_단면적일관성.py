#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import warnings
import openai as OpenAI
import ast

import langchain
from langchain.llms import OpenAI
warnings.filterwarnings(action='ignore')

OPENAI_API_KEY = "" #@param {type:"string"}
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = OpenAI(model_name='text-davinci-003', temperature=0.9)


def get_category_counts(unique_array):
 
    prompt = f"""I want to classify several items according to their meaning and then create a 2D array without any comment. 
    For example, if the input is ['F', 'M'], then the output should be [['F'], ['M']], and if the input is ['F', 'M', '여', 'Female', '남자', 'Male', '여자', '남'], the output should be [['F', 'Female', '여', '여자'], ['남', '남자', 'M', 'Male']]. 
    The string I want to classify is {unique_array}. Give me only a 2D array without any comment."""
    #response = llm(prompt).strip()
    response = "[['F','여', '여성'], ['M','Male', '남', 'Man']]"
    #print('response', response)
    
    array_2d = ast.literal_eval(response)
    #print('array_2d:', array_2d)
    
    category_counts = {}
    for category in array_2d:
        category_counts[category[0]] = len(category)

    return array_2d, category_counts



def calculate_inner_consistency(array_2d, df, TARGET_VARIABLE):
    total_weighted_percentage = 0
    total_weight = 0
    
    for line in array_2d:  # lines 대신 array_2d 사용
        count_by_col = df[TARGET_VARIABLE].apply(lambda x: x if x in line else None).value_counts().dropna()
        
        for value, count in count_by_col.items():
            max_percentage = count / count_by_col.sum()
            weighted_percentage = max_percentage * count
            total_weighted_percentage += weighted_percentage
        
        total_weight += count_by_col.sum()
        
    return total_weighted_percentage / total_weight

def calculate_consistency(csv_file_name):
    TARGET_VARIABLE = '성별'
    df = pd.read_csv(csv_file_name)
    unique_array = df[TARGET_VARIABLE].dropna().unique()
    
    array_2d, category_counts = get_category_counts(unique_array)
    
    def show_detail(array_2d, category_counts):
        with open('text_consistency.txt', 'w', encoding='utf-8') as file:
            for category in array_2d:
                category_name = category[0]
                count = category_counts[category_name]
                items = ', '.join(category)
                line = f"{category_name}: {count}개 - 항목: {items}\n"
                # 파일에 쓰기
                file.write(line)
                # 프롬프트에 출력
                print(line, end='')
    
    consistency_value = calculate_inner_consistency(array_2d, df, TARGET_VARIABLE) 
    print('일관성:', consistency_value )
    show_detail(array_2d, category_counts)

