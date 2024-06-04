import pandas as pd
import numpy as np
import sys


def process_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    pd.set_option('display.max_columns', None)
    df['Mapping_info_2'] = df['Mapping_info_2'].fillna('Not Provided')

    non_medical_code_df = df[df['Mapping_info_1'] != 'medical_code']
    medical_code_df = df[df['Mapping_info_1'] == 'medical_code']

    return df, non_medical_code_df, medical_code_df


def calculate_statistics(df, non_medical_code_df, medical_code_df):
    counts = non_medical_code_df.groupby(
        ['Mapping_info_1', 'Mapping_info_2', 'Variable_name', 'Patient_number']).size().reset_index(name='occurrences')
    grouped_stats = counts.groupby(['Mapping_info_1', 'Mapping_info_2', 'Variable_name']).agg(
        {'occurrences': ['mean', 'std']}).reset_index()
    grouped_stats.columns = ['Mapping_info_1', 'Mapping_info_2', 'Variable_name', 'Mean', 'Std']

    patient_level_counts = medical_code_df.groupby(['Patient_number', 'Variable_name', 'Value']).size().reset_index(
        name='Occurrences')
    patient_level_counts['Variable_name'] = patient_level_counts['Variable_name'] + " - " + patient_level_counts[
        'Value'].astype(str)
    medical_code_stats = patient_level_counts.groupby('Variable_name').agg(Mean=('Occurrences', 'mean'),
                                                                           Std=('Occurrences', 'std')).reset_index()

    combined_stats = pd.concat([grouped_stats.drop(columns=['Mapping_info_1', 'Mapping_info_2']), medical_code_stats])

    unique_counts = df.groupby('Variable_name')['Patient_number'].nunique().reset_index(name='Unique_Patients')
    grouped_stats_with_counts = pd.merge(combined_stats, unique_counts, on='Variable_name')

    total_weighted_average = np.average(grouped_stats_with_counts['Mean'],
                                        weights=grouped_stats_with_counts['Unique_Patients'])
    rounded_total_weighted_average = round(total_weighted_average, 2)

    return rounded_total_weighted_average, grouped_stats, medical_code_stats


def print_and_save(output, file):
    print(output)
    file.write(output + '\n')


def run(csv_path):
    df, non_medical_code_df, medical_code_df = process_data(csv_path)
    rounded_total_weighted_average, grouped_stats, medical_code_stats = calculate_statistics(df, non_medical_code_df,
                                                                                             medical_code_df)

    with open('결과_충실성.txt', 'w', encoding='utf-8') as f:
        print_and_save(f"충실성: {rounded_total_weighted_average}", f)

        for (mapping_info_1, mapping_info_2), group in grouped_stats.groupby(['Mapping_info_1', 'Mapping_info_2']):
            if mapping_info_2 != 'Not Provided':
                output = f"Category: {mapping_info_1}, Subcategory: {mapping_info_2}\n{group[['Variable_name', 'Mean', 'Std']].to_string(index=False)}\n"
                print_and_save(output, f)

        for mapping_info_1, group in grouped_stats[grouped_stats['Mapping_info_2'] == 'Not Provided'].groupby(
                'Mapping_info_1'):
            output = f"Category: {mapping_info_1}\n{group[['Variable_name', 'Mean', 'Std']].to_string(index=False)}\n"
            print_and_save(output, f)

        output = "Category: medical_code\n" + medical_code_stats.to_string(index=False) + '\n'
        print_and_save(output, f)

    print("Detailed results are saved as '결과_충실성.txt'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <csv_file_path>")
    else:
        config_path = (sys.argv[1])
        config_data = read_yaml(config_path)
        csv_path = config_data.get('csv_path')
        run(csv_path)
