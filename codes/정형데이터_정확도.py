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

warnings.filterwarnings(action='ignore') 


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

    
    def show_detail():
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('f1 score:', f1score)
        print("AUROC: %f" % mean_auc)
        
    print('Accuracy:', accuracy)
    #show_detail()

def save_graphs(y_true, y_pred):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')

    # ROC Curve
    lb = LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)
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

def calculate_accuracy(csv):
    df = pd.read_csv(csv)
    target_df = df[df['변수명'] == '골절유무']
    target_df['true'] = target_df['true'].astype(int).astype(str)
    target_df['값'] = target_df['값'].astype(int).astype(str)
    y_pred  = target_df['값']
    y_true = target_df['true']
    
    calculation(y_true, y_pred)
    save_graphs(y_true, y_pred)

