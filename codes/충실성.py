import pandas as pd
import numpy as np
import os
import sys
import yaml

def create_folders_and_files(base_path):
    os.makedirs(base_path, exist_ok=True)

def save_overall_fidelity(base_path, overall_fidelity):
    with open(os.path.join(base_path, 'total_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Overall Fidelity Score: {overall_fidelity}\n")

def save_detailed_fidelity(base_path, grouped_stats):
    output_path = os.path.join(base_path, 'total_table.csv')
    grouped_stats.to_csv(output_path, index=False)

def save_category_fidelity(base_path, grouped_stats):
    for (category, subcategory), group in grouped_stats.groupby(['Mapping_info_1', 'Mapping_info_2']):
        # Handle subcategory properly
        if pd.isna(subcategory) or subcategory == '':
            category_folder = os.path.join(base_path, category)
        else:
            category_folder = os.path.join(base_path, f"{category}_{subcategory}")
        os.makedirs(category_folder, exist_ok=True)

        avg_fidelity = np.average(group['Mean'], weights=group['Patient_number'])
        avg_fidelity_rounded = round(avg_fidelity, 2)
        with open(os.path.join(category_folder, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Average Fidelity Score: {avg_fidelity_rounded}\n")

        group_rounded = group.round({'Mean': 2, 'Std': 2, 'Patient_number': 0})
        group_rounded.to_csv(os.path.join(category_folder, 'table.csv'), index=False, columns=['Variable_name', 'Mean', 'Std', 'Patient_number'])

def process_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    pd.set_option('display.max_columns', None)
    df['Mapping_info_2'] = df['Mapping_info_2'].fillna('')
    return df

def calculate_statistics(df):
    # Calculate occurrences for non-medical codes
    non_medical_code_df = df[df['Mapping_info_1'] != 'medical_code']
    counts = non_medical_code_df.groupby(['Mapping_info_1', 'Mapping_info_2', 'Variable_name', 'Patient_number']).size().reset_index(name='occurrences')
    grouped_stats = counts.groupby(['Mapping_info_1', 'Mapping_info_2', 'Variable_name']).agg(
        Mean=('occurrences', 'mean'),
        Std=('occurrences', 'std'),
        Patient_number=('Patient_number', 'nunique')
    ).reset_index()

    # Round the mean and std to 2 decimal places
    grouped_stats['Mean'] = grouped_stats['Mean'].round(2)
    grouped_stats['Std'] = grouped_stats['Std'].round(2)
    grouped_stats['Patient_number'] = grouped_stats['Patient_number'].round(0)

    # Calculate occurrences for medical codes
    medical_code_df = df[df['Mapping_info_1'] == 'medical_code']
    if not medical_code_df.empty:
        medical_counts = medical_code_df.groupby(['Patient_number', 'Variable_name', 'Value']).size().reset_index(name='Occurrences')
        medical_counts['Variable_name'] = medical_counts['Variable_name'] + " - " + medical_counts['Value'].astype(str)
        medical_code_stats = medical_counts.groupby('Variable_name').agg(
            Mean=('Occurrences', 'mean'),
            Std=('Occurrences', 'std'),
            Patient_number=('Patient_number', 'nunique')
        ).reset_index()

        # Round the mean and std to 2 decimal places
        medical_code_stats['Mean'] = medical_code_stats['Mean'].round(2)
        medical_code_stats['Std'] = medical_code_stats['Std'].round(2)
        medical_code_stats['Patient_number'] = medical_code_stats['Patient_number'].round(0)

        medical_code_stats['Mapping_info_1'] = 'medical_code'
        medical_code_stats['Mapping_info_2'] = ''
        grouped_stats = pd.concat([grouped_stats, medical_code_stats], ignore_index=True)

    overall_fidelity = np.average(grouped_stats['Mean'], weights=grouped_stats['Patient_number'])
    rounded_overall_fidelity = round(overall_fidelity, 2)
    return rounded_overall_fidelity, grouped_stats

def run(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    csv_path = config['csv_path']
    save_path = config['save_path']

    base_path = os.path.join(save_path, 'Fidelity(Structured)')
    create_folders_and_files(base_path)

    df = process_data(csv_path)
    rounded_overall_fidelity, grouped_stats = calculate_statistics(df)

    save_overall_fidelity(base_path, rounded_overall_fidelity)
    save_detailed_fidelity(base_path, grouped_stats)
    save_category_fidelity(base_path, grouped_stats)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <yaml_config_path>")
    else:
        config_path = sys.argv[1]
        run(config_path)
