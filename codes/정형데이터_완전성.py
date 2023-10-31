# -*- coding: utf-8 -*-
"""정형데이터_완전성1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p4GYrnzGiNdGLXj38xhIOPic4nK_t02E
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, na_values=["null", "?", "UNKNOWN", "unknown", "NOT SPECIFIED", "not specified", ""])

def calculate_completeness_for_variable(column, total_rows):
    """Calculate completeness for a single variable."""
    variable_name = column.name
    n_rows = len(column)
    n_rows_checked = total_rows - column.isnull().sum()
    fraction_empty = column.isnull().sum() / total_rows * 100
    completeness_ratio = (n_rows_checked / n_rows) * 100
    return {
        'Field': variable_name,
        'N rows': n_rows,
        'N rows checked': n_rows_checked,
        'Fraction Empty': f"{fraction_empty:.2f}%",
        'Completeness ratio for variable': f"{completeness_ratio:.2f}%"
    }

def calculate_overall_completeness(df):
    """Calculate overall completeness for the dataset."""
    total_rows = len(df)
    overall_completeness = sum(df.count()) / (total_rows * len(df.columns)) * 100
    return overall_completeness

def plot_completeness(completeness_ratios):
    """Plot the completeness ratios."""
    variable_names = list(completeness_ratios.keys())
    completeness_values = list(completeness_ratios.values())
    plt.bar(variable_names, completeness_values)
    plt.xlabel('Variable')
    plt.ylabel('Completeness Ratio (%)')
    plt.title('Completeness Ratio for Each Variable')
    plt.xticks(rotation=90)
    plt.show()

def calculate_completeness(file_path, display_chart=False):
    df = load_data(file_path)
    scan_report = []
    total_rows = len(df)
    completeness_ratios = {}

    for column in df.columns:
        report = calculate_completeness_for_variable(df[column], total_rows)
        scan_report.append(report)
        completeness_ratios[report['Field']] = float(report['Completeness ratio for variable'].replace("%", ""))

    overall_completeness = calculate_overall_completeness(df)

    if display_chart:
        plot_completeness(completeness_ratios)

    return overall_completeness, scan_report, completeness_ratios

if __name__ == "__main__":
    file_path = '/content/20231030_mimic_lab_sample.csv'
    overall_completeness, scan_report, completeness_ratios = calculate_completeness(file_path, display_chart=False)
    print(f"완전성: {overall_completeness:.2f}%")