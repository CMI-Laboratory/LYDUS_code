import pandas as pd
import yaml
import sys
import openai
import re
import ast

# Function to read data from a YAML file
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to read variables and their descriptions from the specified CSV file
def read_variables_csv(file_path):
    data = pd.read_csv(file_path)
    # Select only the 'Variable_name' and 'Description' columns for the dictionary
    return pd.Series(data['Description'].values, index=data['Variable_name']).to_dict()

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

# Continue with the rest of the original script for data processing...
# Define the function to check if a string is a date
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
        data = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, "Failed to load data"

    # Apply filters to identify relevant rows
    filter_one_df = data[data['Variable_type'] == 'pandas._libs.tslibs.timestamps.Timestamp']
    remaining_df = data[~data.index.isin(filter_one_df.index)]
    filter_two_df = remaining_df[remaining_df['Value'].apply(lambda x: isinstance(x, str) and is_date_string(x))]

    combined_df = pd.concat([filter_one_df, filter_two_df])
    unique_names = combined_df['Variable_name'].unique().tolist()

    return combined_df, ', '.join(unique_names)

# Get combined DataFrame and unique variables string
combined_time_df, unique_variables_string = get_combined_time_df_with_unique_variable_names(csv_path)

combined_info = "YAML Variables and Descriptions:\n"
combined_info += "\n".join([f"{k}: {v}" for k, v in
                            yaml_variable_dictionary.items()]) if yaml_variable_dictionary else "No variables defined in YAML configuration."
combined_info += f"\n\nUnique Time-related Variables from CSV:\n{unique_variables_string}"

#print(combined_info)

# Count the total number of records in the combined DataFrame
total_records = len(combined_time_df)
#print(f"Total records in the combined DataFrame: {total_records}")

# 2. OpenAI GPT API 호출 및 응답 처리
# 설명: GPT-4를 호출하여 데이터셋 검증에 필요한 timepoint_pairs를 얻는 섹션. GPT 시스템 프롬트를 정의 하고, 응답으로부터 timepoint_pairs 문자열을 추출하고,
# 파싱하여 python 객체로 변환합니다.

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

# timepoint pairs 초기화
timepoint_pairs = []

try:
    # OpenAI GPT-4 API 호출 로직
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
    # print("API call successful.")
    # API 응답에서 timepoint_pairs 문자열 추출 및 파싱
    start_of_list = response.choices[0].message.content.find('[')
    timepoint_pairs_str = response.choices[0].message.content[start_of_list:]
    # print("Extracted timepoint pairs string:")
    # print(timepoint_pairs_str)

    # 문자열을 리스트로 파싱
    timepoint_pairs = ast.literal_eval(timepoint_pairs_str)
    # print("Successfully parsed timepoint pairs:")
    #print(timepoint_pairs)

except SyntaxError as e:
    print(f"Syntax error while parsing timepoint pairs: {e}")
    print("Failed to parse string:")
    print(timepoint_pairs_str)
except Exception as e:
    print(f"Unexpected error: {e}")


# 3  데이터 검증 함수
# 설명: LLM에서 추출된 timepoint_pairs 를 사용해서 데이터셋 내 레코드를 검증하는 함수입니다.
# 각 입원 번호 (ID)별로 데이터셋을 순회하면서, 시작 시간과 종료 시간의 순서가 올바른지 검증합니다.
# 또한 검증 결과, 유효한 레코드 비율과 오류 상세 정보를 반환 합니다.

def validate_records_long_format(dataset, timepoint_pairs):
    total_records = len(dataset)  # Total number of records in combined_time_df
    errors_count = 0  # Initialize the count of errors (rows involved)
    errors_details = []  # To store details of the errors

    # Iterate through the dataset to check each pair for errors
    for index, row in dataset.iterrows():
        for start_col, end_col in timepoint_pairs:
            # Check if current row variable name matches the start_col and has a corresponding end time in the dataset
            if row['Variable_name'] == start_col:
                end_row = dataset[
                    (dataset['Admission_number'] == row['Admission_number']) & (dataset['Variable_name'] == end_col)]
                if not end_row.empty:
                    start_time = row['Value']
                    end_time = end_row.iloc[0]['Value']
                    # If start time is not before end time, indicating an error, affects 2 rows
                    if pd.to_datetime(start_time) >= pd.to_datetime(end_time):
                        errors_count += 2  # Increment by 2 for each error as it involves two rows
                        errors_details.append((row['Admission_number'], start_col, start_time, end_col, end_time))

    valid_records_count = total_records - errors_count  # Calculate valid records count
    validity_percentage = (valid_records_count / total_records) * 100 if total_records > 0 else 0

    return validity_percentage, errors_count, valid_records_count, errors_details, total_records


# Call the function with the dataset and timepoint_pairs
dataset = combined_time_df  # Ensure this is your DataFrame
results = validate_records_long_format(dataset, timepoint_pairs)

# After calling the function, unpack the results
validity_percentage, errors_count, valid_records_count, errors_details, total_checked_records = results

# Round the validity_percentage to 3 decimal places before printing
rounded_validity_percentage = round(validity_percentage, 3)

# Print the results with the rounded validity percentage
print(f"선후관계 유효성: {rounded_validity_percentage}%")
print(f"시간/날짜 데이터 기록 수: {total_checked_records}")
print(f"유효한 기록 수: {valid_records_count}")
print(f"선후 관계 오류가 있는 기록 수: {errors_count}")
print("선후 관계 오류 세부 내용: 입원 번호, '시작 변수명', '값', '종료 변수명' , '값'")
for detail in errors_details:
    print(detail)
