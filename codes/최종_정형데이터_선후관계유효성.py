import sys
import pandas as pd
import openai
import ast
import yaml

# Function to read data from a YAML file
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

# Read configuration from config.yml
config_data = read_yaml('config.yml')
openai.api_key = config_data['open_api_key']

# Check if a command line argument (CSV file path) is provided
if len(sys.argv) > 1:
    file_path = sys.argv[1]  # Get the file path from command line argument
else:
    print("No CSV file path provided. Exiting.")
    sys.exit(1)
    
# Function to validate records based on timepoint pairs
def validate_records_long_format(dataset, timepoint_pairs):
    valid_records_count = 0
    total_checked_records = 0
    errors_count = 0
    errors_details = []

    # Group by 연구등록번호 to process each admission record
    grouped = dataset.groupby('연구등록번호')

    for 연구등록번호, group in grouped:
        is_record_valid = True
        record_has_data = False

        # Create a dictionary for easy timepoint lookup
        timepoints = {row['변수명']: row['기록날짜'] for _, row in group.iterrows()}

        for start_col, end_col in timepoint_pairs:
            start_time = timepoints.get(start_col)
            end_time = timepoints.get(end_col)

            if pd.notnull(start_time) and pd.notnull(end_time):
                record_has_data = True
                # Adjusted condition to allow same start and end dates
                if pd.to_datetime(start_time) > pd.to_datetime(end_time):
                    is_record_valid = False
                    errors_count += 1
                    errors_details.append((연구등록번호, start_col, start_time, end_col, end_time))
                    break
            elif pd.isnull(start_time) or pd.isnull(end_time):
                continue

        if record_has_data:
            total_checked_records += 1
            if is_record_valid:
                valid_records_count += 1

    validity_percentage = (valid_records_count / total_checked_records) * 100 if total_checked_records > 0 else 0
    return validity_percentage, errors_count, errors_details

# Read configuration from config.yml
config_data = read_yaml('config.yml')
openai.api_key = config_data['open_api_key']

file_path = config_data['csv_path']
variable_list = config_data['Variable_list']

# Convert the entire Variable_list to a string for GPT-4 input
variable_list_string = '\n'.join([f"{key}: {value}" for key, value in variable_list.items()])

# Define the system prompt for GPT-4
system_prompt = """
As a data analyst in a healthcare setting, your task is to validate the sequential ordering of time-related fields within medical datasets. The aim is to uphold the data quality by ensuring the chronological order of medical events as recorded across various hospitals.

Procedure:
- Read variable names: Review variable names and their labels from medical data, noting that variable names may vary in terms of explicitness and abbreviation.
- Chronological Understanding: Establish the chronological order of events based on the time-related variables within the medical data.
- Exclude Non-Time Variables: Any variable that does not represent a time point should be excluded (e.g., location, status codes, categorical data).
- Exclude Unpaired Time Variables: If a variable representing a start time does not have a corresponding end time (or vice versa), it should be excluded from the pairing process.
- Exclude Sensitive or Complex Time Variables: Variables that are sensitive in nature (like 'death_time' or 'year_of_birth') or that have complex relationships (like 'diagnosis_date' preceding 'admission_date') that require additional context should be excluded from direct pairing.
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
    ('admissionTime', 'dischargeTime'),
    ('centralLinePlacementTime', 'centralLineRemovalTime'),
    ('sedationStartTime', 'sedationEndTime'),
]
"""

# GPT-4 call using the updated OpenAI API
try:
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": variable_list_string}
        ],
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    gpt_response = completion.choices[0].message.content
except Exception as e:
    print(f"An error occurred: {e}")

# Parsing GPT-4 response and loading dataset
timepoint_pairs_str = gpt_response.split("=", 1)[-1].strip()
timepoint_pairs_str = timepoint_pairs_str.strip("Output Format: ").rstrip("., ")
timepoint_pairs = ast.literal_eval(timepoint_pairs_str)

# Print the timepoint pairs
print("Generated Timepoint Pairs:", timepoint_pairs)

dataset = pd.read_csv(file_path)

# Validate records and output results
validity_percentage, errors_count, errors_details = validate_records_long_format(dataset, timepoint_pairs)
print(f"Percentage of valid records: {validity_percentage}%")
include_more_info = True

if include_more_info:
    more_info = f"Number of errors: {errors_count}\n"
    more_info += "Error details (입원 ID, start column, start time, end column, end time):\n"
    for detail in errors_details:
        more_info += f"{detail}\n"
    print(more_info)


config_data = read_yaml('config.yml')
openai.api_key = config_data['open_api_key']

