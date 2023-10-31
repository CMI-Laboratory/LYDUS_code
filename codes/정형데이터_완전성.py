#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 정형데이터_완전성.py
# in terminal run python 정형데이터_완전성.py /path/to/your/csvfile.csv

import pandas as pd
import sys

def calculate_completeness(dataframe, required_columns):
    """
    Calculate the overall completeness percentage of a dataset based on the specified required columns.
    
    Parameters:
    - dataframe: The dataset in the form of a pandas DataFrame.
    - required_columns: List of columns that are required to be filled out.
    
    Returns:
    - Overall completeness percentage based on the required columns.
    """
    # Check for missing values in the required columns
    missing_values = dataframe[required_columns].isnull().sum()
    
    # Calculate the completeness percentage for each required column
    completeness_percentage = 100 - (missing_values / len(dataframe) * 100)
    
    # Calculate the overall completeness of the dataset based on the required columns
    overall_completeness = completeness_percentage.mean()
    
    return overall_completeness

def evaluate_dataset_completeness(csv_path):
    """
    Load a given CSV file, examine its structure, identify the truly required columns, 
    and evaluate the overall completeness using the calculate_completeness function.
    
    Parameters:
    - csv_path: Path to the CSV file.
    
    Returns:
    - Overall completeness percentage.
    """
    # Load the data from the provided CSV path
    dataframe = pd.read_csv(csv_path)
    
    # Identify the truly required columns based on prior understanding
    truly_required_columns = ['변수 category', '변수명', '기록 날짜', '값', '변수 타입', '환자 번호']
    
    # Calculate the overall completeness using the function
    completeness = calculate_completeness(dataframe, truly_required_columns)
    
    return completeness
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the CSV file as an argument.")
        sys.exit(1)

    # Retrieve the CSV path from command-line arguments
    csv_path = sys.argv[1]

    # Use the function to get the overall completeness and round it
    completeness_percentage = round(evaluate_dataset_completeness(csv_path), 2)

    print(f"Overall completeness: {completeness_percentage}%")

