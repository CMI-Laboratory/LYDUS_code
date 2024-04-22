import sys
import yaml
import pandas as pd
import re
from collections import defaultdict, Counter
import openai

def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Check for the command-line argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <config_path>")
    sys.exit(1)

# Load configuration data from the YAML file
config_path = sys.argv[1]
config = read_yaml(config_path)

# Retrieve and set the OpenAI API key
openai_api_key = config.get('open_api_key')
if not openai_api_key:
    print("OpenAI API key not found in the configuration file.")
    sys.exit(1)
openai.api_key = openai_api_key

# Retrieve the path to the QUIQ CSV file
csv_path = config.get('csv_path')
if not csv_path:
    print("CSV path not found in the configuration file.")
    sys.exit(1)

# Retrieve the path to the VIA CSV file
VIA_path = config.get('VIA_path')
if not VIA_path:
    print("VIA path not found in the configuration file.")
    sys.exit(1)

#start of code 
quiq_df = pd.read_csv(csv_path, low_memory=False)
via_df = pd.read_csv(VIA_path, low_memory=False)


# Filter rows where "Mapping_info_1" contains "medical_code"
filtered_quiq_df = quiq_df[quiq_df['Mapping_info_1'].str.contains('medical_code', na=False)]

# Read VIA table
via_df = pd.read_csv(VIA_path, low_memory=False)

# Extract unique variable names from the filtered QUIQ table
unique_variable_names = filtered_quiq_df['Variable_name'].unique()

# Find matching descriptions in the VIA table for the unique variable names
matched_columns_df = via_df[via_df['Variable_name'].isin(unique_variable_names)][['Variable_name', 'Description']]

# Ensure each variable name is unique in the matched DataFrame
unique_matched_columns_df = matched_columns_df.drop_duplicates(subset=['Variable_name'])

# Initialize a dictionary for storing regex patterns and their frequencies per variable
variable_pattern_frequencies = defaultdict(lambda: Counter())

def generate_regex_gpt4(variable_name, description):
    system_prompt = f"""
    For each medical coding system described below, provide a regular expression pattern that accurately matches the most standard codes of that system based on the provided characteristics. The regular expression should not include Python function calls or variable assignments. Only provide the pattern itself, enclosed in quotes.
    Now, for the medical coding system named '{variable_name}' described as '{description}', provide the regular expression pattern.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please generate the regex pattern for the description provided."}
            ],
            temperature=0,
            max_tokens=100,
            top_p=1.0,
            n=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        text = response.choices[0]['message']['content'].strip()
        pattern_str = text.strip('"')
        return pattern_str
    except Exception as e:
        return None

# Perform regex pattern generation and frequency tracking
NUM_GENERATIONS = 10  # Number of iterations to generate patterns for each variable
for _ in range(NUM_GENERATIONS):
    for _, row in unique_matched_columns_df.iterrows():
        variable_name = row['Variable_name']
        description = row['Description']
        pattern_str = generate_regex_gpt4(variable_name, description)
        if pattern_str:
            variable_pattern_frequencies[variable_name][pattern_str] += 1

# Select the most frequent pattern for each variable and compile it for regex matching
regex_dict = {}
for variable_name, patterns in variable_pattern_frequencies.items():
    most_frequent_pattern, frequency = patterns.most_common(1)[0]
    regex_dict[variable_name] = re.compile(most_frequent_pattern)

# Function to validate codes against the regex patterns
def validate_code(variable_name, code):
    pattern = regex_dict.get(variable_name)
    if pattern:
        return bool(pattern.match(str(code)))
    else:
        return False

# Collecting detailed validation results
detailed_validation_results = []
for index, row in filtered_quiq_df.iterrows():
    valid = validate_code(row['Variable_name'], row['Value'])
    detailed_validation_results.append({
        "Variable": row['Variable_name'],
        "Code": row['Value'],
        "Valid": "True" if valid else "False"
    })

validation_df = pd.DataFrame(detailed_validation_results)
validation_df['Valid'] = validation_df['Valid'].replace({'True': True, 'False': False}).astype(bool)

# Calculate the total number of codes and valid codes
total_codes = len(validation_df)
valid_count = validation_df['Valid'].sum()

# Calculate validity percentage
validity_percentage = (valid_count / total_codes) * 100 if total_codes > 0 else 0

# The output
print(f"Code Validity: {validity_percentage:.2f}%")

# Prepare data for the text file
data_rows = [
    "<Code Validity Summary>",
    f"Code Validity: {validity_percentage:.2f}%",
    f"Valid Codes: {valid_count}",
    f"Total Codes: {total_codes}",
    "",
    "<Summary Regular Expression Suggestion>"
]

# Collecting regex pattern frequencies for each variable
for variable_name in variable_pattern_frequencies:
    for pattern, frequency in variable_pattern_frequencies[variable_name].items():
        data_row = f"Variable: {variable_name}\tPattern: {pattern}\tFrequency: {frequency}"
        data_rows.append(data_row)

# Adding the most frequent regex patterns selected
data_rows.append("")
data_rows.append("<Most Frequent Regex Patterns Selected>")
for variable_name, patterns in variable_pattern_frequencies.items():
    most_frequent_pattern, frequency = patterns.most_common(1)[0]
    selected_pattern = f"Variable: {variable_name}\tSelected Pattern: {most_frequent_pattern}\tFrequency: {frequency}"
    data_rows.append(selected_pattern)

# Prepare the GPT model validation results
data_rows.append("")
data_rows.append("<Final Validation Results>")
validation_df['Result'] = validation_df.apply(lambda x: f"{x['Variable']}\t{x['Code']}\t{x['Valid']}", axis=1)
data_rows.extend(validation_df['Result'].tolist())

# Save to text file
with open("결과_code_validity.txt", "w", encoding="utf-8") as text_file:
    for row in data_rows:
        text_file.write(row + "\n")

# The output
print("Detailed results are saved as '결과_code_validity.txt'")
