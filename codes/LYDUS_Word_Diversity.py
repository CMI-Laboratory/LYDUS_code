import yaml
import argparse
import re
import string
import pandas as pd
from typing import List, Tuple, Dict
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def draw_word_diversity_histogram(ax: Axes, item_vs_percentage: dict, n: int):
    ax.clear()
    top_n_items = []
    top_n_percentages = []
    for i, item in enumerate(item_vs_percentage):
        if i == n:
            break
        top_n_items.append(item)
        top_n_percentages.append(item_vs_percentage[item])
    ax.bar(top_n_items, top_n_percentages)
    ax.set_xlabel('Items')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

def _noun_word(text):
    noun_pattern = re.compile(r'\bNN')
    if not isinstance(text, str):
        return []
    noun_words = []
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    for word, tag in pos_tags:
        has_noun = bool(noun_pattern.match(tag))
        if has_noun:
            word = word.strip()
            noun_words.append(word)
    noun_words = [word for word in noun_words if word not in string.punctuation]
    return noun_words

def _counter_topn_items(total_count, total_counter, top_n=10):
    if total_count == 0:
        return [], []
    top_n_items = total_counter.most_common(top_n)
    top_items = []
    top_n_percentages = []
    for item, count in top_n_items:
        percentage = round(count / total_count*100, 2)
        top_items.append(item)
        top_n_percentages.append(percentage)
    return top_items, top_n_percentages

def _percentage_top_items(total_count, total_counter, percentage):
    if total_count == 0:
        return 0
    top_items_count = int(total_count * percentage / 100)
    sorted_items = total_counter.most_common()
    top_items_sum = sum(count for _, count in sorted_items[:top_items_count])
    return round(top_items_sum / total_count*100, 2)

def get_vocabulary_diversity(quiq: pd.DataFrame, top_n: int) -> Tuple[float, Dict[str, float], Dict[float, float], pd.DataFrame]:
    
    df = quiq.copy()
    df['Mapping_info_1'] = df['Mapping_info_1'].astype(str)
    
    df_note = df[df['Mapping_info_1'].str.contains('note', na=False, case=False)]
    assert len(df_note) > 0, 'FAIL : No note values.'

    col_name = 'Value'
    df_note['TEXT_words'] = df_note[col_name].apply(_noun_word)
    total_words = [word for sublist in df_note['TEXT_words'].tolist() for word in sublist]

    unique_word_count = len(set(total_words))
    total_count = len(total_words)
    total_counter = Counter(total_words)
    total_words = sum(total_counter.values())

    freq_df = pd.DataFrame(total_counter.items(), columns=['Word', 'Count'])
    freq_df['Percentage'] = round(freq_df['Count'] / total_words * 100, 2)
    word_diversity = round(unique_word_count / total_count*100, 2) if total_count > 0 else 0

    items, percentages = _counter_topn_items(total_count, total_counter, total_count)
    item_vs_percentage = {label: percentages[i] for i, label in enumerate(items)}

    top_percentages = [5, 10, 20]
    coverage_scores = {percentage: _percentage_top_items(total_count, total_counter, percentage) for percentage in top_percentages}

    return word_diversity, item_vs_percentage, coverage_scores, freq_df

if __name__ == '__main__':
    print('<LYDUS - Vocabulary Diversity>\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    quiq_path = config.get('quiq_path')
    save_path = config.get('save_path')
    top_n = config.get('top_n', 10)

    quiq = pd.read_csv(quiq_path)

    vocabulary_diversity, item_vs_percentage, coverage_scores, freq_df = get_vocabulary_diversity(
        quiq=quiq,
        top_n=top_n
    )

    freq_df = freq_df.sort_values(by = 'Percentage', ascending = False)
    freq_df.to_csv(f"{save_path}/vocabulary_diversity_frequency.csv", index=False)

    with open(f"{save_path}/vocabulary_diversity_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Vocabulary Diversity: {vocabulary_diversity}\n")
        f.write("Coverage Scores (Top %):\n")
        for k, v in coverage_scores.items():
            f.write(f"  Top {k}%: {v}\n")
            
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_word_diversity_histogram(ax, item_vs_percentage, n=top_n)
    fig.tight_layout()
    fig.savefig(f"{save_path}/vocabulary_diversity_plot.png")

    print('<SUCCESS>')
