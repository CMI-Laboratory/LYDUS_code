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
if not csv_path:
    print("CSV path not found in the configuration file.")
    sys.exit(1)
    
top_n = config_data.get('top_n', 10)


def custom_sent_tokenize(text):
    if not isinstance(text, str) or not text.strip():
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
    if total_count == 0:  # To avoid division by zero
        return [], []
    top_n_items = total_counter.most_common(top_n)
    top_items = []
    top_n_percentages = []
    for item, count in top_n_items:
        percentage = (count / total_count) * 100
        top_items.append(item)
        top_n_percentages.append(percentage)
    return top_items, top_n_percentages


def plot_top_n_items(top_items, percentages, top_n=10):
    if not top_items:
        print("No data to plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.bar(top_items, percentages)  # Ensure 'percentages' are scaled from 0 to 100 if not already
    plt.xlabel('Items')
    plt.ylabel('Percentage (%)')  # Specify that the y-axis is in percentage
    plt.title(f'Top {top_n} Sentences by Percentage')
    plt.xticks(rotation=45)
    #plt.tight_layout()  # Adjust layout to make room for tick labels
    plt.savefig(f'top_{top_n}_sentences.png')
    plt.show()

def percentage_top_items(total_count, total_counter, percentage):
    if total_count == 0:
        return 0
    top_items_count = int(total_count * (percentage / 100))
    sorted_items = total_counter.most_common()
    top_items_sum = sum(count for _, count in sorted_items[:top_items_count])
    return (top_items_sum / total_count) * 100

def show_detail(total_sentences, total_counter, total_count, top_n=10):
    top_n_items, top_n_percentages = counter_topn_items(total_count, total_counter, top_n)
    plot_top_n_items(top_n_items, top_n_percentages, top_n)
    
    top_percentages = [5, 10, 20]
    for percentage in top_percentages:
        score = percentage_top_items(total_count, total_counter, percentage)
        print(f"Top {percentage}% of items account for {score:.2f}% of the total.")

# 주 함수
def calculate_sentence_diversity(csv_file, top_n=10):
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        if 'Mapping_info_1' not in df or 'Value' not in df:
            print("Required columns are missing in the dataset.")
            return
    except FileNotFoundError:
        print("The CSV file was not found.")
        return

    df_note = df[df['Mapping_info_1'].str.contains('note', na=False, case=False)]
    if df_note.empty:
        print("No 'note' related records found.")
        return
    
    df_note['TEXT_Verb'] = df_note['Value'].apply(verb_sentence)
    total_sentences = [sentence for sublist in df_note['TEXT_Verb'].tolist() for sentence in sublist]

    if not total_sentences:
        print("No sentences with verbs were found.")
        return
    
    unique_sentence_count = len(set(total_sentences))
    total_count = len(total_sentences)
    sen_diversity = (unique_sentence_count / total_count) * 100
    print(f"Sentence Diversity: {sen_diversity:.2f}%")

    show_detail(total_sentences, Counter(total_sentences), total_count, top_n)

calculate_sentence_diversity(csv_path, top_n)

