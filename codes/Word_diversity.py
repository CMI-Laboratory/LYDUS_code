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
import os

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
    
save_path = config_data.get('save_path')
if not save_path:
    print("Save path not found in the configuration file.")
    sys.exit(1)

# Create the folder path for storing results
results_folder = os.path.join(save_path, 'Word_diversity')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
top_n = config_data.get('Text_diversity_top_n', 10)
target_word = config_data.get('target_word', None)

#명사만 추출
def noun_word(text):
    noun_pattern = re.compile(r'\bNN')
    if not isinstance(text, str):
        return []
    noun_words=[]
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

def counter_topn_items(total_count, total_counter, top_n):
    if total_count == 0:  # Prevent division by zero
        return [], []
    top_n_items = total_counter.most_common(top_n)
    top_items = []
    top_n_percentages = []
    for item, count in top_n_items:
        percentage = (count / total_count) * 100
        top_items.append(item)
        top_n_percentages.append(percentage)
    return top_items, top_n_percentages

def plot_top_n_items(top_items, percentages, folder, top_n):
    if not top_items:
        print("No data to plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.bar(top_items, percentages)  
    plt.xlabel('Items')
    plt.ylabel('Percentage (%)')  # Specify that the y-axis is in percentage
    plt.title(f'Top {top_n} Words by Percentage')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(results_folder, f'top_{top_n}_words.png'))
    #plt.show()
    plt.close()


    
def percentage_top_items(total_count, total_counter, percentage):
    if total_count == 0:
        return 0
    top_items_count = int(total_count * (percentage / 100))
    sorted_items = total_counter.most_common()
    top_items_sum = sum(count for _, count in sorted_items[:top_items_count])
    return (top_items_sum / total_count) * 100

def show_detail(total_sentences, total_counter, total_count, top_n, output_file):
    top_n_items, top_n_percentages = counter_topn_items(total_count, total_counter, top_n)
    plot_top_n_items(top_n_items, top_n_percentages,results_folder, top_n)
    
    top_percentages = [5, 10, 20]
    coverage_scores = {}
    for percentage in top_percentages:
        coverage_scores[percentage] = percentage_top_items(total_count, total_counter, percentage)

    for percentage, score in coverage_scores.items():
        output_file.write(f"Top {percentage}% of items account for {score:.2f}% of the total.")
        print(f"Top {percentage}% of items account for {score:.2f}% of the total.")

# 특정 단어의 비율 계산
def calculate_word_percentage(total_counter, total_count, word):
    if word not in total_counter:
        print(f"The word '{target_word}' is not present in the text.")
        return None
    word_count = total_counter[word]
    word_percentage = (word_count / total_count) * 100 if total_count > 0 else 0
    return word_percentage
        
# 주 함수
def calculate_word_diversity(csv_file, top_n, target_word=None):
    df = pd.read_csv(csv_file, low_memory=False)
    if df.empty:
        print("The dataframe is empty.")
        return

    df_note = df[df['Mapping_info_1'].str.contains('note', na=False, case=False)]
    if df_note.empty:
        print("No notes available for analysis.")
        return
    
    
    df_note['TEXT_words'] = df_note['Value'].apply(noun_word) #모든 단어 시 extract_words 사용
    total_words = [word for sublist in df_note['TEXT_words'].tolist() for word in sublist]
    
    with open(os.path.join(results_folder, 'word_diversity_details.txt'), 'w', encoding='utf-8') as output_file:
        if not total_words:
            output_file.write("No words extracted from notes.\n")
            return
        
        total_counter = Counter(total_words)
        total_count = len(total_words)
        unique_word_count = len(set(total_words))
        word_diversity = (unique_word_count / total_count) * 100
        output_file.write(f"Word Diversity: {word_diversity:.2f}%\n")
        print(f"Word Diversity: {word_diversity:.2f}%\n")
        
        if target_word:
            target_word_percentage = calculate_word_percentage(total_counter, total_count, target_word)
            if target_word_percentage is not None:
                output_file.write(f"The word '{target_word}' accounts for {target_word_percentage:.2f}% of the total words.\n")
                print(f"The word '{target_word}' accounts for {target_word_percentage:.2f}% of the total words.\n")
            else:
                output_file.write(f"The word '{target_word}' is not present in the text.\n")
                print(f"The word '{target_word}' is not present in the text.\n")
        
        show_detail(total_words, total_counter, total_count, top_n, output_file)
  
    
 
        
calculate_word_diversity(csv_path, top_n, target_word=None)

