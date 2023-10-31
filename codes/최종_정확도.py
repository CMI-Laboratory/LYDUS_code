#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelBinarizer

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
    
    print("AUROC: %f" % mean_auc)
    
    def show_detail():
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('f1 score:', f1score)
        print("AUROC: %f" % mean_auc)
    

        
def draw_ROC(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)
    y_pred_binary = lb.transform(y_pred)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure()
    
    # Binary classification case
    if len(lb.classes_) == 2:
        fpr[0], tpr[0], _ = roc_curve(y_true_binary[:, 0], y_pred_binary[:, 0])
        roc_auc[0] = auc(fpr[0], tpr[0])
        plt.plot(fpr[0], tpr[0], label='ROC curve (AUC = %0.2f)' % roc_auc[0])
    # Multiclass case
    else:
        for i, class_label in enumerate(lb.classes_):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label='ROC curve for class %s (AUC = %0.2f)' % (class_label, roc_auc[i]))
        
        # Macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(lb.classes_))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(lb.classes_)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(lb.classes_)
        
        plt.plot(all_fpr, mean_tpr, label='Macro-average ROC curve (AUC = %0.2f)' % np.mean(list(roc_auc.values())), linestyle=":")

    plt.plot([0, 1], [0, 1], 'k--')  # Reference line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def draw_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    
def calculate_accuracy(csv):
    df = pd.read_csv(csv)
    target_df = df[df['변수명'] == '골절유무']
    target_df['true'] = target_df['true'].astype(int).astype(str)
    target_df['값'] = target_df['값'].astype(int).astype(str)
    y_pred  = target_df['값']
    y_true = target_df['true']
    
    calculation(y_true, y_pred)
    #draw_confusion_matrix(y_true, y_pred)
    #draw_ROC(y_true, y_pred)
    

