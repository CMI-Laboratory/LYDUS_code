import os
import re
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelBinarizer

def draw_auroc_variable_plot(save_path, n, table_name, variable_name, y_true, y_pred, unique_labels):
    fig = Figure(figsize = (15, 5))

    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 1.8])
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title(f'{variable_name} - Confusion Matrix')

    lb = LabelBinarizer()
    lb.fit(unique_labels)
    y_true_binary = lb.transform(y_true)
    y_pred_binary = lb.transform(y_pred)

    fpr, tpr, roc_auc = {}, {}, {}
    if len(lb.classes_) == 2: 
        fpr[0], tpr[0], _ = roc_curve(y_true_binary[:, 0], y_pred_binary[:, 0])
        roc_auc[0] = auc(fpr[0], tpr[0])
        axes[1].plot(fpr[0], tpr[0], label=f'{variable_name} - ROC curve (AUC = %0.2f)' % roc_auc[0])
    else: 
        for i, class_label in enumerate(lb.classes_):
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            axes[1].plot(fpr[i], tpr[i], label=f'{variable_name} - ROC curve for class {class_label} (AUC = %0.2f)' % roc_auc[i])

    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{variable_name} - ROC Curve')
    axes[1].legend(loc='lower right')

    table_name = re.sub(r'[\\/:*?"<>|]', ' ', table_name)
    variable_name = re.sub(r'[\\/:*?"<>|]', ' ', variable_name)
    fig.savefig(save_path + f'/classification_plots/{n}_{table_name}_{variable_name}.png')

def get_classification(quiq:pd.DataFrame) :

    assert quiq['Ground_truth'].notnull().sum() > 0, 'FAIL : No ground truth values'

    cond = quiq['Value'].notnull() & quiq['Ground_truth'].notnull()
    df_filtered = quiq[cond]
    
    assert len(df_filtered) > 0, 'FAIL : No available data.'
    
    df_grouped = df_filtered.groupby(['Original_table_name', 'Variable_name'])

    results = []
    total_samples = 0
    weighted_sum_accuracy = 0
    weighted_sum_precision = 0
    weighted_sum_recall = 0
    weighted_sum_f1score = 0
    weighted_sum_auc = 0
    
    for idx, target_df in df_grouped:
        table_name = idx[0]
        variable_name = idx[1]
        
        y_true = target_df['Ground_truth'].astype(str) 
        y_pred = target_df['Value'].astype(str) 

        accuracy = round(accuracy_score(y_true, y_pred), 3) # accuracy
        precision = round(precision_score(y_true, y_pred, average='macro'), 3) # precision
        recall = round(recall_score(y_true, y_pred, average='macro'), 3) # recall
        f1score = round(f1_score(y_true, y_pred, average='macro'), 3) # f1 score

        lb = LabelBinarizer() # One Hot Encoding
        y_true_binary = lb.fit_transform(y_true)
        y_pred_binary = lb.transform(y_pred)

        fpr = dict() 
        tpr = dict()
        roc_auc = dict()

        if len(lb.classes_) == 2: 
            fpr[0], tpr[0], _ = roc_curve(y_true_binary[:, 0], y_pred_binary[:, 0])
            roc_auc[0] = auc(fpr[0], tpr[0])
        else:
            for i, class_label in enumerate(lb.classes_):
                fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_binary[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        mean_auc = round(np.mean(list(roc_auc.values())), 3)

        results.append([table_name, variable_name, accuracy, precision, recall, f1score, mean_auc])
        num_samples = len(y_true)
        total_samples += num_samples
        weighted_sum_accuracy += accuracy * num_samples
        weighted_sum_precision += precision * num_samples
        weighted_sum_recall += recall * num_samples
        weighted_sum_f1score += f1score * num_samples
        weighted_sum_auc += mean_auc * num_samples

    weighted_accuracy = round(weighted_sum_accuracy / total_samples, 3)
    weighted_precision = round(weighted_sum_precision / total_samples, 3)
    weighted_recall = round(weighted_sum_recall / total_samples, 3)
    weighted_f1score = round(weighted_sum_f1score / total_samples, 3)
    weighted_auc = round(weighted_sum_auc / total_samples, 3)

    result_df = pd.DataFrame(results, columns=['Original_table_name', 'Variable_name', 'Accuracy', 'Precision', 'Recall', 'F1Score', 'AUROC'])
    
    return result_df, df_grouped, (weighted_accuracy, weighted_precision, weighted_recall, weighted_f1score, weighted_auc)

if __name__ == '__main__' :
    print('<LYDUS - Classification Metrics>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    
    result_df, df_grouped, (weighted_accuracy, weighted_precision, weighted_recall, weighted_f1score, weighted_auc) = get_classification(quiq)
    
    with open(save_path + '/classification_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Weighted Accuracy = {weighted_accuracy}\n')
        file.write(f'Weighted Precision = {weighted_precision}\n')
        file.write(f'Weighted Recall = {weighted_recall}\n')
        file.write(f'Weighted F1score = {weighted_f1score}\n')
        file.write(f'Weighted AUROC = {weighted_auc}\n')
    
    result_df.to_csv(save_path + '/classification_summary.csv', index = False)
    
    os.makedirs(save_path + '/classification_plots', exist_ok = True)
    
    for n, (idx, target_df) in enumerate(df_grouped):
        table_name = idx[0]
        variable_name = idx[1]

        y_true = target_df['Ground_truth'].astype(str) 
        y_pred = target_df['Value'].astype(str)

        unique_labels = y_true.unique()

        draw_auroc_variable_plot(save_path, n, table_name, variable_name, y_true, y_pred, unique_labels)
    
    print('<SUCCESS>')
