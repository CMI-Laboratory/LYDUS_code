#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import matplotlib.pyplot as plt
import string
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


#명사만 추출
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



def counter_topn_items(total_count, total_counter, top_n=10):
    top_n_items = total_counter.most_common(top_n)
    top_items = []
    top_n_percentages = []
    for item, count in top_n_items:
        percentage = (count / total_count) * 100
        top_items.append(item)
        top_n_percentages.append(percentage)
    return top_items, top_n_percentages

def plot_top_n_items(top_items, percentages):
    plt.bar(top_items, percentages)
    plt.xlabel('Items')
    plt.ylabel('Percentage')
    plt.title('Top 10 Items by Percentage')
    plt.xticks(rotation=45)
    plt.savefig('word_diversity.png')
    plt.show()

def percentage_top_items(total_count, total_counter, percentage):
    top_items_count = int(total_count * percentage / 100)
    sorted_items = total_counter.most_common()
    top_items_sum = sum(count for _, count in sorted_items[:top_items_count])
    percentage_top_items = (top_items_sum / total_count) * 100
    return percentage_top_items

def show_detail(total_sentences, total_counter, total_count, top_n = 10):
    top_n_items, top_n_percentages = counter_topn_items(total_count, total_counter, top_n)
    plot_top_n_items(top_n_items, top_n_percentages)
    
    top_percentages = [5, 10, 20]
    coverage_scores = {}
    for percentage in top_percentages:
        coverage_scores[percentage] = percentage_top_items(total_count, total_counter, percentage)

    for percentage, score in coverage_scores.items():
        print(f"상위 {percentage}% 항목이 전체의 {score:.2f}%를 차지합니다.")

# 주 함수
def calculate_word_diversity(csv_file):
    df = pd.read_csv(csv_file)
    df_note = df[df['Mapping_info_1'].str.contains('note', na=False, case=False)]
    
    col_name = 'Value'
    df_note['TEXT_words'] = df_note[col_name].apply(noun_word) #모든 단어 시 extract_words 사용
    total_words = [word for sublist in df_note['TEXT_words'].tolist() for word in sublist]
    
    unique_word_count = len(set(total_words))
    total_count = len(total_words)
    total_counter = Counter(total_words)

    word_diversity = unique_word_count / total_count
    print("단어의 다양성:", word_diversity)

    # 필요한 경우 아래 라인의 주석을 해제하여 세부 정보를 출력
    top_n = 10
    show_detail(total_words, total_counter, total_count, top_n)

    
    
calculate_word_diversity(csv_path)

