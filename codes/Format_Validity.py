import sys
import yaml
import pandas as pd
import re
from collections import defaultdict
import openai
import os
import matplotlib.pyplot as plt

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

# Set the OpenAI API key
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

# Retrieve the save path
save_path = config.get('save_path', '.')

# Create Format Validity folder
format_validity_folder = os.path.join(save_path, "Format Validity")
os.makedirs(format_validity_folder, exist_ok=True)

# Create a file to save GPT-4 output
gpt_output_path = os.path.join(format_validity_folder, "gpt_output.txt")
gpt_output_file = open(gpt_output_path, 'w')

# Start of code
quiq_df = pd.read_csv(csv_path, low_memory=False)
via_df = pd.read_csv(VIA_path, low_memory=False)

# Filter rows where "Mapping_info_1" contains "medical_code"
filtered_quiq_df = quiq_df[quiq_df['Mapping_info_1'].str.contains('medical_code', na=False)]

# Check if there are any rows with "medical_code"
if filtered_quiq_df.empty:
    print("No rows with 'medical_code' found. Exiting the script.")
    sys.exit(0)

# Extract unique variable names from the filtered QUIQ table
unique_variable_names = filtered_quiq_df['Variable_name'].unique()

# Find matching descriptions in the VIA table for the unique variable names
matched_columns_df = via_df[via_df['Variable_name'].isin(unique_variable_names)][['Variable_name', 'Description']]

# Ensure each variable name is unique in the matched DataFrame
unique_matched_columns_df = matched_columns_df.drop_duplicates(subset=['Variable_name'])

# Initialize a dictionary for storing validation results
validation_results = defaultdict(list)

# Cache for storing previously validated codes and formats
validated_cache = {}
format_cache = {}

# Function to identify the format pattern of a code
def get_code_format(code):
    return ''.join(['9' if c.isdigit() else 'A' if c.isalpha() else c for c in code])

# Function to ask GPT-4 for validation
def validate_code_with_gpt4(variable_name, description, code):
    system_prompt = f"""
    You are a medical code validation expert. Given a variable name, description, and a code, confirm if the code is valid based on the description.
    Variable name: {variable_name}
    Description: {description}
    Code: {code}
    Using your knowledge base, confirm if the code is likely valid or not. Answer "True" if it is correct, otherwise answer "False". Try to rely on its training data to make a more educated guess regarding the validity of the codes, even you cannot access real-time databases.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please validate the code based on the provided description using your knowledge base."}
            ]
        )
        text = response.choices[0]['message']['content'].strip()
        gpt_output_file.write(f"GPT-4 response for {variable_name} ({code}): {text}\n")
        return text == "True"
    except Exception as e:
        gpt_output_file.write(f"Error validating {variable_name} ({code}): {e}\n")
        return None

# Perform validations with format caching
for variable_name in unique_variable_names:
    # Filter rows specific to the current variable
    variable_df = filtered_quiq_df[filtered_quiq_df['Variable_name'] == variable_name]

    # Get description for the current variable
    description = unique_matched_columns_df.loc[unique_matched_columns_df['Variable_name'] == variable_name, 'Description'].values[0]

    for _, row in variable_df.iterrows():
        code = row['Value']
        code_format = get_code_format(code)

        # Check if the format has already been validated
        if code_format in format_cache:
            validation_results[variable_name].append({
                "Code": code,
                "Valid": format_cache[code_format]
            })
            validated_cache[code] = format_cache[code_format]
            continue

        # Check if the exact code has already been validated
        if code in validated_cache:
            validation_results[variable_name].append({
                "Code": code,
                "Valid": validated_cache[code]
            })
            continue

        # Validate using GPT-4 if not already validated
        is_valid = validate_code_with_gpt4(variable_name, description, code)
        validation_results[variable_name].append({
            "Code": code,
            "Valid": is_valid
        })
        validated_cache[code] = is_valid
        format_cache[code_format] = is_valid

# Close the GPT-4 output file
gpt_output_file.close()

# Convert validation results to DataFrame
validation_df = pd.DataFrame([
    {"Variable": variable, "Code": result["Code"], "Valid": result["Valid"]}
    for variable, results in validation_results.items()
    for result in results
])

# Apply a fallback mechanism for obviously incorrect GPT-4 validations and handle None
def fallback_validation(row):
    code = row['Code']
    if not re.match(r'^[A-Za-z0-9\.\-]*$', code):  # Basic validation for alphanumeric codes with optional periods and hyphens
        return False
    return row['Valid'] if row['Valid'] is not None else False

# Fill None with False explicitly before applying ~
validation_df['Valid'] = validation_df['Valid'].fillna(False)

# Save errors details to total_table.csv
error_details = validation_df[~validation_df['Valid']]

# Calculate the total number of codes and valid codes
total_codes = len(validation_df)
valid_count = validation_df['Valid'].sum()
invalid_count = total_codes - valid_count

# Calculate validity percentage
validity_percentage = (valid_count / total_codes) * 100 if total_codes > 0 else 0

# Save overall results to total_results.txt
with open(os.path.join(format_validity_folder, "total_results.txt"), 'w') as f:
    f.write(f"Format Validity (%): {validity_percentage:.2f}%\n")
    f.write(f"Total Code: {total_codes}\n")
    f.write(f"Valid Code: {valid_count}\n")
    f.write(f"Invalid Code: {invalid_count}\n")

# Save errors details to total_table.csv
error_summary = validation_df.groupby('Variable').agg(
    Total_Code=('Code', 'count'),
    Invalid_Code=('Valid', lambda x: (~x).sum())
).reset_index()
error_summary['Format Validity (%)'] = ((error_summary['Total_Code'] - error_summary['Invalid_Code']) / error_summary['Total_Code']) * 100
error_summary['Format Validity (%)'] = error_summary['Format Validity (%)'].round(2)  # Round to two decimal places

# Save the summary to total_table.csv
error_summary.to_csv(os.path.join(format_validity_folder, "total_table.csv"), index=False)

# Save pie chart
labels = 'Valid Codes', 'Invalid Codes'
sizes = [valid_count, invalid_count]
colors = ['#FFFFE0', '#F0F8FF']  # light yellow and light blue colors
explode = (0.1, 0)

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Format Validity')
plt.savefig(os.path.join(format_validity_folder, "total_piechart.jpg"))

# For every unique "Variable_name" in the error details
for variable_name in error_details['Variable'].unique():
    variable_folder = os.path.join(format_validity_folder, variable_name)
    os.makedirs(variable_folder, exist_ok=True)

    variable_df = validation_df[validation_df['Variable'] == variable_name]
    variable_total_codes = len(variable_df)
    variable_valid_count = variable_df['Valid'].sum()
    variable_invalid_count = variable_total_codes - variable_valid_count
    variable_validity_percentage = (
        variable_valid_count / variable_total_codes) * 100 if variable_total_codes > 0 else 0

    # Save relevant results to [Variable_name]_results.txt
    with open(os.path.join(variable_folder, f"{variable_name}_results.txt"), 'w') as f:
        f.write(f"Format Validity (%): {variable_validity_percentage:.2f}%\n")
        f.write(f"Total Code: {variable_total_codes}\n")
        f.write(f"Valid Code: {variable_valid_count}\n")
        f.write(f"Invalid Code: {variable_invalid_count}\n")

    # Save relevant error details to [Variable_name]_table.csv
    variable_df[['Variable', 'Code', 'Valid']].to_csv(os.path.join(variable_folder, f"{variable_name}_table.csv"),
                                                      index=False)

print("The output is saved in the 'Format Validity' folder.")
