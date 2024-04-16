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
    try:
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')
    except ValueError as e:
        print("Error in metric calculation:", e)
        return
    
    
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

    
    def show_detail():
        print(f'Accuracy: {accuracy*100:.2f}%')
        print(f'Precision: {precision*100:.2f}%')
        print(f'Recall: {recall*100:.2f}%')
        print(f'F1 Score: {f1score*100:.2f}%')
        print(f"AUROC: {mean_auc*100:.2f}%")
        
    show_detail()

def save_graphs(y_true, y_pred, n):
    unique_labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    
    if len(unique_labels) > n:
        print(f"Graphs for the confusion matrix and AUROC are not displayed because the number of classes exceeds {n}.")
        return
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Confusion Matrix
    unique_labels = np.union1d(np.unique(y_true), np.unique(y_pred))  # 실제와 예측 레이블의 고유한 값 합집합
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)  # labels 매개변수를 사용하여 혼동 행렬 계산
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)  # display_labels에 고유한 레이블 제공
    disp.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')

    # ROC Curve
    lb = LabelBinarizer()
    lb.fit(unique_labels)  # unique_labels를 사용하여 LabelBinarizer 학습
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
    
    # Ground_truth 값이 null 이 아닌 row만 선택
    target_df = df[df['Ground_truth'].notnull() & df['Value'].notnull()]
    
    if target_df.empty:
        print('No data to calculate Accuracy, Precision, Recall, F1 Score, AUROC.')
        return
    
    target_df['Value'] = target_df['Value'].astype(str)
    target_df['Ground_truth'] = target_df['Ground_truth'].astype(str)
    y_true  = target_df['Ground_truth']
    y_pred = target_df['Value']
    
    calculation(y_true, y_pred)
    save_graphs(y_true, y_pred, n)



calculate_accuracy(csv_path, maximum_class_n)

