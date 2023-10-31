#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter

def custom_sent_tokenize(text):
    # 입력값이 문자열이 아니면 빈 문자열을 반환
    if not isinstance(text, str):
        return []

    sentences = re.split(r'(?<!\n)\n{2,}|[^\w\s\n,]+', text)
    return sentences

def verb_sentence(text):
    verb_sentences = []
    sentences = custom_sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        has_verb = any(re.match(r'\bVB', tag) for _, tag in pos_tags)
        if has_verb:
            sentence = sentence.strip()
            verb_sentences.append(sentence)
    return verb_sentences

def count_total_elements(lists):
    return sum(len(lst) for lst in lists)

def percentage_top_items(data, percentage):
    total_count = len(data)
    sorted_items = Counter(data).most_common()

    top_items_count = int(total_count * percentage / 100)
    top_items_sum = sum(count for _, count in sorted_items[:top_items_count])
    percentage_top_items = (top_items_sum / total_count) * 100

    return percentage_top_items

def calculate_sentence_diversity(csv_file):
    df_all = pd.read_csv(csv_file)
    df = df_all[df_all['변수명'] == '판독결과'].copy()
    
    col_name = '값' #text를 가지는 칼럼 이름
    df['TEXT_Verb'] = df[col_name].apply(verb_sentence)
    total_sentences = [sentence for sublist in df['TEXT_Verb'].tolist() for sentence in sublist]
    unique_sentence_count = len(set(total_sentences))
   
    total_elements = count_total_elements(df['TEXT_Verb'])
    sen_diversity = unique_sentence_count / total_elements
    
    top_percentages = [5, 10, 20]
    coverage_scores = {}
    for percentage in top_percentages:
        coverage_scores[percentage] = percentage_top_items(total_sentences, percentage)

    print("문장의 다양성:",sen_diversity)
    for percentage, score in coverage_scores.items():
        print(f"상위 {percentage}% 항목이 전체의 {score:.2f}%를 차지합니다.")

