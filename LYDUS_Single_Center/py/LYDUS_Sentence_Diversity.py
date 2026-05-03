import yaml
import argparse
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def draw_sentence_diversity_histogram(ax:Axes, item_vs_percentage:dict, n:int):
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

def _custom_sent_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return re.split(r'(?<!\n)\n{2,}|[^\w\s\n,]+', text)

def _verb_sentence(text):
    verb_sentences = []
    sentences = _custom_sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        has_verb = any(re.match(r'\bVB', tag) for _, tag in pos_tags)
        if has_verb:
            sentence = sentence.strip()
            verb_sentences.append(sentence)
    return verb_sentences

def _counter_topn_items(total_count, total_counter, top_n=10):
    if total_count == 0:  # To avoid division by zero
        return [], []
    top_n_items = total_counter.most_common(top_n)
    top_items = []
    top_n_percentages = []
    for item, count in top_n_items:
        percentage = round(count / total_count * 100, 2)
        top_items.append(item)
        top_n_percentages.append(percentage)
    return top_items, top_n_percentages

def _percentage_top_items(total_count, total_counter, percentage):
    if total_count == 0:
        return 0
    top_items_count = int(len(total_counter) * percentage / 100)
    sorted_items = total_counter.most_common()
    top_items_sum = sum(count for _, count in sorted_items[:top_items_count])
    return round(top_items_sum/total_count * 100, 2)

def get_sentence_diversity(quiq:pd.DataFrame, top_n:int) :
    df = quiq.copy()
    df['Mapping_info_1'] = df['Mapping_info_1'].astype(str)

    df_note = df[df['Mapping_info_1'].str.contains('note', na=False, case=False)]
    assert len(df_note) > 0, 'FAIL : No note values.'
    
    df_note['TEXT_Verb'] = df_note['Value'].apply(_verb_sentence)
    total_sentences = [sentence for sublist in df_note['TEXT_Verb'].tolist() for sentence in sublist]
    
    unique_sentence_count = len(set(total_sentences))
    total_count = len(total_sentences)
    sen_diversity = round(unique_sentence_count/total_count *100 , 2) if total_count>0 else 0
    total_counter = Counter(total_sentences)

    total_sentences = sum(total_counter.values())
    freq_df = pd.DataFrame(total_counter.items(), columns=['Sentence', 'Count'])
    freq_df['Percentage'] = round(freq_df['Count'] / total_sentences * 100, 2)

    items, percentages = _counter_topn_items(total_count, total_counter, total_count)
    item_vs_percentage = {label: percentages[i] for i, label in enumerate(items)}
    
    top_percentages = [5, 10, 20]
    coverage_scores = {percentage: _percentage_top_items(total_count, total_counter, percentage) for percentage in top_percentages}

    return sen_diversity, item_vs_percentage, coverage_scores, freq_df


if __name__ == '__main__':
    print('<LYDUS - Sentence Diversity>\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    quiq_path = config.get('quiq_path')
    save_path = config.get('save_path')
    top_n = config.get('top_n', 10)

    quiq = pd.read_csv(quiq_path)

    sen_diversity, item_vs_percentage, coverage_scores, freq_df = get_sentence_diversity(
        quiq=quiq,
        top_n=top_n
    )

    freq_df = freq_df.sort_values(by = 'Percentage', ascending = False)
    freq_df.to_csv(f"{save_path}/sentence_diversity_frequency.csv", index=False)

    with open(f"{save_path}/sentence_diversity_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Sentence Diversity: {sen_diversity}\n")
        f.write("Coverage Scores (Top %):\n")
        for k, v in coverage_scores.items():
            f.write(f"  Top {k}%: {v}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_sentence_diversity_histogram(ax, item_vs_percentage, n=top_n)
    fig.tight_layout()
    fig.savefig(f"{save_path}/sentence_diversity_plot.png")

    print('<SUCCESS>')
