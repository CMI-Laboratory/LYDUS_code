#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import string


def noun_word(text):
    noun_pattern = re.compile(r'\bNN')
    noun_words=[]
    if not isinstance(text, str):
        return []
    words = word_tokenize(text)
    pos_tags = pos_tag(words)  # 전체 단어에 대한 품사 태깅
    for word, tag in pos_tags:
        has_noun = bool(noun_pattern.match(tag))
        if has_noun:
            word = word.strip()
            noun_words.append(word)
    # 문장 기호 제거
    noun_words = [word for word in noun_words if word not in string.punctuation]
    return noun_words

#명사 뿐 아니라 단어 모두 추출
def extract_words(text):
    if not isinstance(text, str):
        return []
    words = word_tokenize(text)
    # 문장 기호 제거
    words = [word for word in words if word not in string.punctuation]
    return words



def word_extract(csv_file):
    df_all = pd.read_csv(csv_file)
    df = df_all[df_all['변수명'] == '판독결과'].copy()
    
    col_name = '값' #text를 가지는 칼럼 이름
    #명사
    #df['TEXT_word'] = df[col_name].apply(lambda x: noun_word(x))
    
    #모든 품사
    df['TEXT_word'] = df[col_name].apply(lambda x: extract_words(x))
    flattened_list = [item for sublist in df['TEXT_word'] for item in sublist]
    return flattened_list


    
def count_unique_word(word_list):
    unique_word = set(word_list)
    return len(unique_word)

def percentage_top_items(data, percentage):
    total_count = len(data)
    sorted_items = Counter(data).most_common()

    top_items_count = int(total_count * percentage / 100)
    top_items_sum = sum(count for _, count in sorted_items[:top_items_count])
    percentage_top_items = (top_items_sum / total_count) * 100
    return percentage_top_items

def calculate_word_diversity(word_lists):
    total_elements = len(word_lists)
    unique_word_count = count_unique_word(word_lists)
    word_diversity = (unique_word_count / total_elements)
    
    return word_diversity

def calculate_word_diversity_from_csv(csv_file_name):
    word_list = word_extract(csv_file_name)
    
    top_percentages = [5, 10, 20]
    coverage_scores = {}
    for percentage in top_percentages:
        coverage_scores[percentage] = percentage_top_items(word_list, percentage)

    diversity_score = calculate_word_diversity(word_list)
    
    print("단어의 다양성:", diversity_score)
    for percentage, score in coverage_scores.items():
        print(f"상위 {percentage}% 항목이 전체의 {score:.2f}%를 차지합니다.")

