import pandas as pd
import yaml
import openai
import re
import ast
import sys
import os

# Function to read data from a YAML file
def read_yaml(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except UnicodeEncodeError as e:
        print(f"Unicode encoding error while reading YAML file at {path}: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

# Function to read variables and their descriptions from the specified CSV file
def read_variables_csv(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    # Select only the 'Variable_name' and 'Description' columns for the dictionary
    return pd.Series(data['Description'].values, index=data['Variable_name']).to_dict()

# Function to check if a string is a date
def is_date_string(s):
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
    ]
    return any(re.match(pattern, s) for pattern in date_patterns)

# Process CSV file and extract unique time-related variables
def get_combined_time_df_with_unique_variable_names(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, "Failed to load data"

    # Apply filters to identify relevant rows
    timestamp_types = ['pandas._libs.tslibs.timestamps.Timestamp', 'datetime64[ns]', 'datetime64[ns, UTC]']
    filter_one_df = data[data['Variable_type'].isin(timestamp_types)]
    remaining_df = data[~data.index.isin(filter_one_df.index)]
    filter_two_df = remaining_df[remaining_df['Value'].apply(lambda x: isinstance(x, str) and is_date_string(x))]

    combined_df = pd.concat([filter_one_df, filter_two_df])
    unique_names = combined_df['Variable_name'].unique().tolist()

    return combined_df, ', '.join(unique_names)

# Function to save results and details to files
def save_results_and_details(save_path, validity_percentage, total_checked_records, valid_records_count, errors_count, errors_details):
    # Create Sequence Validity folder
    sequence_validity_folder = os.path.join(save_path, "Sequence Validity")
    os.makedirs(sequence_validity_folder, exist_ok=True)

    # Save overall results to total_results.txt
    with open(os.path.join(sequence_validity_folder, "total_results.txt"), 'w') as f:
        f.write(f"Sequential validity: {validity_percentage}%\n")
        f.write(f"Total date/time records: {total_checked_records}\n")
        f.write(f"Valid records: {valid_records_count}\n")
        f.write(f"Records with sequential errors: {errors_count}\n")

    # Save errors details to total_table.csv
    errors_df = pd.DataFrame(errors_details, columns=["Admission_Number", "Start_Variable", "Value", "End_Variable", "Value"])
    errors_df.to_csv(os.path.join(sequence_validity_folder, "total_table.csv"), index=False)

    # Process each unique Start Variable
    unique_start_variables = errors_df["Start_Variable"].unique()
    for start_var in unique_start_variables:
        start_var_folder = os.path.join(sequence_validity_folder, start_var)
        os.makedirs(start_var_folder, exist_ok=True)

        # Filter errors details for the current Start Variable
        filtered_errors_df = errors_df[errors_df["Start_Variable"] == start_var]

        # Calculate specific statistics for the current Start Variable
        start_var_total_records = dataset[dataset['Variable_name'] == start_var].shape[0]
        start_var_errors_count = len(filtered_errors_df)
        start_var_valid_records_count = start_var_total_records - start_var_errors_count
        start_var_validity_percentage = round((start_var_valid_records_count / start_var_total_records) * 100, 2) if start_var_total_records > 0 else 0

        # Save current results to [Start_Variable]_results.txt
        with open(os.path.join(start_var_folder, f"{start_var}_results.txt"), 'w') as f:
            f.write(f"Sequential validity: {start_var_validity_percentage}%\n")
            f.write(f"Total date/time records: {start_var_total_records}\n")
            f.write(f"Valid records: {start_var_valid_records_count}\n")
            f.write(f"Records with sequential errors: {start_var_errors_count}\n")

        # Save filtered errors details to [Start_Variable]_table.csv
        filtered_errors_df.to_csv(os.path.join(start_var_folder, f"{start_var}_table.csv"), index=False)

# Function to validate records in long format
def validate_records_long_format(dataset, timepoint_pairs):
    total_records = len(dataset)
    errors_count = 0
    errors_details = []

    for index, row in dataset.iterrows():
        for start_col, end_col in timepoint_pairs:
            if row['Variable_name'] == start_col:
                end_row = dataset[(dataset['Admission_number'] == row['Admission_number']) & (dataset['Variable_name'] == end_col)]
                if not end_row.empty:
                    start_time = row['Value']
                    end_time = end_row.iloc[0]['Value']
                    if pd.to_datetime(start_time) >= pd.to_datetime(end_time):
                        errors_count += 2
                        errors_details.append((row['Admission_number'], start_col, start_time, end_col, end_time))

    valid_records_count = total_records - errors_count
    validity_percentage = round((valid_records_count / total_records) * 100, 2) if total_records > 0 else 0

    return validity_percentage, errors_count, valid_records_count, errors_details, total_records

# Check if the script is provided with the correct number of command-line arguments
if len(sys.argv) < 2:
    print("Usage: python script_name.py <config_path>")
    sys.exit(1)

# Get the config file path from the command-line arguments
config_path = sys.argv[1]

# Read configuration from config.yml
config_data = read_yaml(config_path)

openai_api_key = config_data.get('open_api_key')
if not openai_api_key:
    print("OpenAI API key not found in the configuration file.")
    sys.exit(1)
openai.api_key = openai_api_key  # Set the OpenAI API key

csv_path = config_data.get('csv_path')

# Read variables and their descriptions from the CSV file specified by VIA_path
via_path = config_data.get('VIA_path')
if not via_path:
    print("VIA_path not found in the configuration file.")
    sys.exit(1)

yaml_variable_dictionary = read_variables_csv(via_path)

# Get combined DataFrame and unique variables string
combined_time_df, unique_variables_string = get_combined_time_df_with_unique_variable_names(csv_path)

combined_info = "YAML Variables and Descriptions:\n"
combined_info += "\n".join([f"{k}: {v}" for k, v in yaml_variable_dictionary.items()]) if yaml_variable_dictionary else "No variables defined in YAML configuration."
combined_info += f"\n\nUnique Time-related Variables from CSV:\n{unique_variables_string}"

# System prompt for OpenAI API
system_prompt = """
As a data analyst in a healthcare setting, your task is to validate the sequential ordering of time-related fields within medical datasets. The aim is to uphold the data quality by ensuring the chronological order of medical events as recorded across various hospitals.

Procedure:
- CSV File Analysis: Review CSV files from medical data, noting that variable names may vary in terms of explicitness and abbreviation.
- Chronological Understanding: Establish the chronological order of events based on the time-related variables within the medical data.
- Exclude Non-Time Variables: Any variable that does not represent a time point should be excluded (e.g., location, status codes, categorical data).
- Exclude Unpaired Time Variables: If a variable representing a start time does not have a corresponding end time (or vice versa), it should be excluded from the pairing process.
- Exclude Sensitive or Complex Time Variables: Variables that are sensitive in nature (like 'death_time' or 'year_of_birth') or that have complex relationships (like 'diagnosis_date' preceding 'admission_date') that require additional context should be excluded from direct pairing.
- Exclude additional context variables such as ('insertion date' preceding 'tubing change') that require additional context.
- Create Time Variable Pairs: Only after exclusions have been identified, formulate pairs of time-related variables that logically represent the beginning and end of an event or process.
- Validate and Finalize Sequence: Confirm that the pairs are in a universally applicable sequence, according to the dataset's context.
- Preserve Original Formatting: Maintain the exact case and naming of the variable names from the dataset.
- Universal Sequences: Focus on sequences with broad applicability, like medication start and end dates.
- Avoid Assumptions: Steer clear of inferring fixed orders where they may not universally apply.
- Circumvent Presumptive Sequences: Refrain from deducing sequences where the order is not consistently established.
- Case Sensitivity: Retain the exact case (uppercase or lowercase) of the original column names in the output.
Once exclusions are identified, proceed to create the output without including any of the excluded variables.

Example 1:
Input Format: admission_time, discharge_time, death_time, admission_location, ventilatorstart_time, emergencyreg_time, emergencyout_time, dateofbirth
Output Format: timepoint_pairs = [
    ('admission_time', 'discharge_time'), 
    ('emergencyreg_time', 'emergencyout_time'),  
]

Example 2:
Input Format: centralLinePlacementTime, centralLineRemovalTime, antibioticAdminTime, sedationStartTime, sedationEndTime, labOrderTime, labResultTime, admissionTime, dischargeTime
Output Format: timepoint_pairs = [
    ('intime', 'outime'), 
    ('sedationStartTime', 'sedationEndTime'),  
]
"""

# Initialize timepoint pairs
timepoint_pairs = []

# Save the results and details
save_path = config_data.get('save_path', '.')

try:
    # OpenAI GPT-4 API call logic
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": unique_variables_string}
        ],
        temperature=0,
        max_tokens=1200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print("API call successful.")
    # Extract timepoint_pairs string from API response and parse
    gpt_output = response.choices[0].message.content
    start_of_list = gpt_output.find('[')
    timepoint_pairs_str = gpt_output[start_of_list:]
    timepoint_pairs = ast.literal_eval(timepoint_pairs_str)

    # Save GPT output to gpt_output.txt in the "Sequence Validity" folder
    sequence_validity_folder = os.path.join(save_path, "Sequence Validity")
    os.makedirs(sequence_validity_folder, exist_ok=True)
    with open(os.path.join(sequence_validity_folder, "gpt_output.txt"), 'w') as f:
        f.write(gpt_output)

except SyntaxError as e:
    print(f"Syntax error while parsing timepoint pairs: {e}")
    print("Failed to parse string:")
    print(timepoint_pairs_str)
except Exception as e:
    print(f"Unexpected error: {e}")

# Call the function with the dataset and timepoint_pairs
dataset = combined_time_df
results = validate_records_long_format(dataset, timepoint_pairs)

# Unpack the results
validity_percentage, errors_count, valid_records_count, errors_details, total_checked_records = results
rounded_validity_percentage = round(validity_percentage, 2)

# Save the results and details
save_results_and_details(save_path, rounded_validity_percentage, total_checked_records, valid_records_count, errors_count, errors_details)
