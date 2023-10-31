# -*- coding: utf-8 -*-
"""코드유효성.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/162n1_M7VBryFjZ5i3jswpdz-_ppPF3JC
"""

import re
import pandas as pd

ICD9_REGEX = "|".join([
    r"\d{3}\.?\d{0,2}",
    r"E\d{3}\.?\d?",
    r"V\d{2}\.?\d{0,2}"
])

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    return df

def is_valid_icd9(code):
    """Check if a given code is a valid ICD-9 code."""
    return re.match(ICD9_REGEX, str(code))

def validate_icd9_codes(df):
    """Validate ICD-9 codes in the dataframe."""
    icd9_rows = df['변수명'] == 'icd9_code'
    df.loc[icd9_rows, 'Validation'] = df.loc[icd9_rows, '변수 ID'].apply(
        lambda code: "valid" if is_valid_icd9(code) else "invalid"
    )
    return df

def calculate_validity_percentage(df):
    """Calculate the validity percentage of ICD-9 codes in the dataframe."""
    icd9_rows = df['변수명'] == 'icd9_code'
    valid_count = df.loc[icd9_rows & (df['Validation'] == "valid")].shape[0]
    total_count = df[icd9_rows].shape[0]
    return (valid_count / total_count) * 100

def validate_icd_from_csv(file_path):
    df = load_data(file_path)
    df = validate_icd9_codes(df)
    valid_percentage = calculate_validity_percentage(df)
    return valid_percentage

if __name__ == "__main__":
    csv_path = "/content/icd_9_labmix2.csv"
    valid_percentage = validate_icd_from_csv(csv_path)
    print(f"코드 유효성: {valid_percentage:.2f}%")