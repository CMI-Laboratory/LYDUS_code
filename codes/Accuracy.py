#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelBinarizer
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

maximum_class_n = config_data.get('maximum_class_n', 10) 

def calculation(y_true, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')
    
    lb = LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)
    y_pred_binary = lb.transform(y_pred)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Binary classification case
    if len(lb.classes_) == 2:
        fpr[0], tpr[0], _ = roc_curve(y_true_binary[:, 0], y_pred_binary[:, 0])
        roc_auc[0] = auc(fpr[0], tpr[0])
    # Multiclass case
    else:
        for i, class_label in enumerate(lb.classes_):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
    mean_auc = np.mean(list(roc_auc.values()))
    
    return accuracy, precision, recall, f1score, mean_auc

def save_graphs(y_true, y_pred, n):
    unique_labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    
    if len(unique_labels) > n:
        print(f"Graphs for the confusion matrix and AUROC are not displayed because the number of classes exceeds {n}.")
        return
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Confusion Matrix
    unique_labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')

    # ROC Curve
    lb = LabelBinarizer()
    lb.fit(unique_labels)
    y_true_binary = lb.transform(y_true)
    y_pred_binary = lb.transform(y_pred)

    fpr, tpr, roc_auc = {}, {}, {}
    if len(lb.classes_) == 2:
        fpr[0], tpr[0], _ = roc_curve(y_true_binary[:, 0], y_pred_binary[:, 0])
        roc_auc[0] = auc(fpr[0], tpr[0])
        axes[1].plot(fpr[0], tpr[0], label='ROC curve (AUC = %0.2f)' % roc_auc[0])
    else:
        for i, class_label in enumerate(lb.classes_):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            axes[1].plot(fpr[i], tpr[i], label='ROC curve for class %s (AUC = %0.2f)' % (class_label, roc_auc[i]))

    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (FPR)')
    axes[1].set_ylabel('True Positive Rate (TPR)')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('accuracy.png')
    plt.show()
    plt.close()

def calculate_accuracy(csv, n):
    df = pd.read_csv(csv)
    
    # Filter rows where 'Ground_truth' is not null
    df = df[df['Value'].notnull() & df['Ground_truth'].notnull()]
    
    # Group by 'Variable_name' and iterate through each group
    grouped = df.groupby('Variable_name')
    
    results = []

    total_samples = 0
    weighted_sum_accuracy = 0
    weighted_sum_precision = 0
    weighted_sum_recall = 0
    weighted_sum_f1score = 0
    weighted_sum_auc = 0

    for variable_name, target_df in grouped:
        if target_df.empty:
            print(f'No data to calculate Accuracy, Precision, Recall, F1 Score, AUROC for {variable_name}.')
            continue
        
        target_df['Value'] = target_df['Value'].astype(str)
        target_df['Ground_truth'] = target_df['Ground_truth'].astype(str)
        y_true = target_df['Ground_truth']
        y_pred = target_df['Value']
        
        print(f'\nMetrics for {variable_name}:')
        accuracy, precision, recall, f1score, mean_auc = calculation(y_true, y_pred)
        print(f'Accuracy: {accuracy*100:.2f}%')
        print(f'Precision: {precision*100:.2f}%')
        print(f'Recall: {recall*100:.2f}%')
        print(f'F1 Score: {f1score*100:.2f}%')
        print(f"AUROC: {mean_auc*100:.2f}%")
        save_graphs(y_true, y_pred, n)
        
        results.append([variable_name, accuracy, precision, recall, f1score, mean_auc])
        
        num_samples = len(y_true)
        total_samples += num_samples
        weighted_sum_accuracy += accuracy * num_samples
        weighted_sum_precision += precision * num_samples
        weighted_sum_recall += recall * num_samples
        weighted_sum_f1score += f1score * num_samples
        weighted_sum_auc += mean_auc * num_samples
    
    if total_samples > 0:
        weighted_accuracy = weighted_sum_accuracy / total_samples
        weighted_precision = weighted_sum_precision / total_samples
        weighted_recall = weighted_sum_recall / total_samples
        weighted_f1score = weighted_sum_f1score / total_samples
        weighted_auc = weighted_sum_auc / total_samples
        
        print('\nWeighted Averages:')
        print(f'Accuracy: {weighted_accuracy*100:.2f}%')
        print(f'Precision: {weighted_precision*100:.2f}%')
        print(f'Recall: {weighted_recall*100:.2f}%')
        print(f'F1 Score: {weighted_f1score*100:.2f}%')
        print(f"AUROC: {weighted_auc*100:.2f}%")
    else:
        print('No data available to calculate weighted averages.')
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['Variable_name', 'Accuracy', 'Precision', 'Recall', 'F1Score', 'AUROC'])
    results_df.to_csv('accuracy_and_metrics.csv', index=False)


calculate_accuracy(csv_path, maximum_class_n)

