import pandas as pd
import sys
import os
import yaml

def main(config_path):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    csv_path = config['csv_path']
    save_path = config['save_path']

    # Create a folder called "Completeness" in the save path
    completeness_folder = os.path.join(save_path, 'Completeness')
    os.makedirs(completeness_folder, exist_ok=True)

    # Read CSV into DataFrame
    df = pd.read_csv(csv_path, low_memory=False)

    # Calculate the overall completeness ratio
    total_non_missing_values = df['Value'].count()
    total_observations = len(df)
    overall_completeness_ratio = (total_non_missing_values / total_observations) * 100

    total_results_text = f"Completeness: {overall_completeness_ratio:.2f}%"
    print(total_results_text)

    # Save the overall completeness ratio to a text file
    total_results_path = os.path.join(completeness_folder, 'total_results.txt')
    with open(total_results_path, 'w') as file:
        file.write(total_results_text)

    # Calculate completeness ratio per variable
    non_null_counts_per_variable = df.groupby('Variable_name')['Value'].count()
    total_counts_per_variable = df.groupby('Variable_name').size()
    completeness_ratio_per_variable = (non_null_counts_per_variable / total_counts_per_variable) * 100

    completeness_df = pd.DataFrame({
        'Variable': completeness_ratio_per_variable.index,
        'Completeness (%)': completeness_ratio_per_variable.values.round(2)  # Round to 2 decimal places
    })

    print(completeness_df)

    # Save the completeness details to a CSV file
    completeness_csv_path = os.path.join(completeness_folder, 'total_results.csv')
    completeness_df.to_csv(completeness_csv_path, index=False)
    print("Detailed results are saved as 'total_results.csv'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the config YAML file path as a command line argument.")
        sys.exit(1)  # Exit the script with an error code

    config_path = sys.argv[1]  # Get the config file path provided as a command line argument
    main(config_path)
