#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
import os
import openai
import ast

def get_unique_variable_names_with_record_date(file_path):
    data = pd.read_csv(file_path)
    filtered_data = data[data['기록 날짜'].notnull()]
    unique_variable_names = filtered_data['변수명'].unique().tolist()
    return ', '.join(unique_variable_names)

def validate_records_long_format(dataset, timepoint_pairs):
    valid_records_count = 0
    total_checked_records = 0
    errors_count = 0
    null_excluded_records = 0
    errors_details = []
    grouped = dataset.groupby('입원 ID')
    for 입원_id, group in grouped:
        is_record_valid = True
        record_has_data = False
        timepoints = {row['변수명']: row['기록 날짜'] for _, row in group.iterrows()}
        for start_col, end_col in timepoint_pairs:
            start_time = timepoints.get(start_col)
            end_time = timepoints.get(end_col)
            if pd.notnull(start_time) and pd.notnull(end_time):
                record_has_data = True
                if pd.to_datetime(start_time) >= pd.to_datetime(end_time):
                    is_record_valid = False
                    errors_count += 1
                    errors_details.append((입원_id, start_col, start_time, end_col, end_time))
                    break
            elif pd.isnull(start_time) or pd.isnull(end_time):
                continue
        if record_has_data:
            total_checked_records += 1
            if is_record_valid:
                valid_records_count += 1
    validity_percentage = (valid_records_count / total_checked_records) * 100 if total_checked_records > 0 else 0
    return validity_percentage, errors_count, null_excluded_records, errors_details

# This part will run when the script is invoked from the command line
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 정형데이터_선후관계유효성.py <path_to_csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset = pd.read_csv(file_path)
    unique_variables_string = get_unique_variable_names_with_record_date(file_path)

    # Set your API key from an environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")

# System prompt for GPT-4 (provided earlier in your example)
system_prompt = """
As a data analyst in a healthcare setting, your task is to validate the sequential ordering of time-related fields within medical datasets. The aim is to uphold the data quality by ensuring the chronological order of medical events as recorded across various hospitals.

Procedure:
- CSV File Analysis: Review CSV files from medical data, noting that variable names may vary in terms of explicitness and abbreviation.
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

# Call to GPT-4 using OpenAI's Python library
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": unique_variables_string}
    ],
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Extracting timepoint pairs from the response
gpt_response = response.choices[0].message.content
timepoint_pairs_str = gpt_response.split("=", 1)[-1].strip()
timepoint_pairs_str = timepoint_pairs_str.strip("Output Format: ").rstrip("., ")
timepoint_pairs = ast.literal_eval(timepoint_pairs_str)

# Call the validation function with your dataset and the timepoint pairs
validity_percentage, errors_count, null_excluded_records, errors_details = validate_records_long_format(dataset, timepoint_pairs)

# Output the results
print(f"Percentage of valid records: {validity_percentage}%")

# Optional: If you want to print error details, set this to True
include_more_info = True  # Set this to True if you want to include more information
if include_more_info:
    print(f"Number of errors: {errors_count}")
    print("Error details (입원 ID, start column, start time, end column, end time):")
    for detail in errors_details:
        print(detail)

