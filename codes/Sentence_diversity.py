#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
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


def custom_sent_tokenize(text):
    if not isinstance(text, str):
        return []
    return re.split(r'(?<!\n)\n{2,}|[^\w\s\n,]+', text)

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
    plt.savefig('sentence_diversity.png')
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
def calculate_sentence_diversity(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    df_note = df[df['Mapping_info_1'].str.contains('note', na=False, case=False)]
    
    col_name = 'Value'
    df_note['TEXT_Verb'] = df_note[col_name].apply(verb_sentence)
    total_sentences = [sentence for sublist in df_note['TEXT_Verb'].tolist() for sentence in sublist]
    
    unique_sentence_count = len(set(total_sentences))
    total_count = len(total_sentences)
    total_counter = Counter(total_sentences)

    sen_diversity = unique_sentence_count / total_count
    print("문장의 다양성:", sen_diversity)

    # 필요한 경우 아래 라인의 주석을 해제하여 세부 정보를 출력
    top_n = 10
    show_detail(total_sentences, total_counter, total_count, top_n)


calculate_sentence_diversity(csv_path)

