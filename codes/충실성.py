import pandas as pd
import sys


def process_data(csv_path):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_path, low_memory=False)
    # Apply filters
    df_filtered = df.dropna(subset=['Patient_number']).dropna(subset=['Value'])
    return df_filtered


def calculate_statistics(df):
    # Main and detailed indicator calculations
    main_indicator = df.groupby(['Mapping_info_1', 'Patient_number']).size().reset_index(name='counts')
    detailed_indicator = df.groupby(['Mapping_info_1', 'Variable_name', 'Patient_number']).size().reset_index(
        name='counts')

    # Statistics for main indicator
    main_stats = main_indicator.groupby('Mapping_info_1')['counts'].agg(['mean', 'std']).reset_index()
    main_stats.columns = ['Mapping_info_1', '평균', '표준편차']
    total_counts = main_indicator.groupby('Mapping_info_1')['counts'].sum().reset_index(name='total_counts')
    weighted_mean = (main_stats['평균'] * total_counts['total_counts']).sum() / total_counts['total_counts'].sum()

    # Statistics for detailed indicator
    detailed_stats = detailed_indicator.groupby(['Mapping_info_1', 'Variable_name'])['counts'].agg(
        ['mean', 'std']).reset_index()
    detailed_stats.columns = ['Mapping_info_1', '변수명', '평균', '표준편차']

    # Rounding
    main_stats['평균'] = main_stats['평균'].round(3)
    main_stats['표준편차'] = main_stats['표준편차'].round(3)
    detailed_stats['평균'] = detailed_stats['평균'].round(3)
    detailed_stats['표준편차'] = detailed_stats['표준편차'].round(3)

    return weighted_mean, main_stats, detailed_stats


def run(csv_path):
    df_valid_values = process_data(csv_path)
    weighted_mean, main_stats, detailed_stats = calculate_statistics(df_valid_values)

    # Output
    print("\n충실성:", round(weighted_mean, 3))
    print("\n카테고리 별:")
    print(main_stats.to_string(index=False))
    print("\n세부 내용:")
    print(detailed_stats.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <csv_file_path>")
    else:
        run(sys.argv[1])
