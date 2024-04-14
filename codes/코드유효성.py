import sys
import yaml
import pandas as pd
import re
from collections import defaultdict, Counter
from io import StringIO
import openai

# Define all necessary functions first
def read_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def parse_gpt_response_to_df(response_text):
    response_text = response_text.strip()
    if not response_text.startswith("Variable\tCode\tValid"):
        response_text = "Variable\tCode\tValid\n" + response_text
    return pd.read_csv(StringIO(response_text), sep="\t")

def generate_regex_gpt4(variable_name, description):
    system_prompt = f"""
    Provide a regular expression pattern for '{variable_name}' described as '{description}'. The pattern should match the most standard codes of the system without being too strict or too loose, and enclosed in quotes.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please generate the regex pattern."}],
            temperature=0,
            max_tokens=100,
            top_p=1.0,
            n=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0]['message']['content'].strip().strip('"')
    except Exception as e:
        print(f"Error generating regex for {variable_name}: {e}")
        return None

def validate_medical_codes_gpt4(unique_matched_columns_df, validation_df):
    system_prompt_parts = [
        "You are a medical code expert. Determine if these codes are valid by rewriting True for correct code, and False for wrong code."
    ]
    for _, row in unique_matched_columns_df.iterrows():
        system_prompt_parts.append(f"{row['Variable_name']}\tContains {row['Description']}")

    system_prompt = "\n".join(system_prompt_parts)
    user_prompt_parts = ["Variable\tCode\tValid"]
    for index, row in validation_df.iterrows():
        user_prompt_parts.append(f"{index}\t{row['Variable']}\t{row['Code']}\t{row['Valid']}")

    user_prompt = "\n".join(user_prompt_parts)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0,
            max_tokens=533,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0]['message']['content'].strip()
    except Exception as e:
        print(f"Error in validating medical codes: {e}")
        return None

def calculate_validity_percentage(response_df):
    valid_count = response_df['Valid'].value_counts().get(True, 0) + response_df['Valid'].value_counts().get('True', 0)
    total_count = len(response_df)
    return (valid_count / total_count) * 100 if total_count > 0 else 0, valid_count, total_count

# Main script execution
if len(sys.argv) < 2:
    print("Usage: python script_name.py <config_path>")
    sys.exit(1)

config_path = sys.argv[1]
config_data = read_yaml(config_path)

openai_api_key = config_data.get('open_api_key')
if not openai_api_key:
    print("OpenAI API key not found in the configuration file.")
    sys.exit(1)
openai.api_key = openai_api_key

csv_path = config_data.get('csv_path')
if not csv_path:
    print("CSV path not found in the configuration file.")
    sys.exit(1)

VIA_path = config_data.get('VIA_path')
if not VIA_path:
    print("VIA path not found in the configuration file.")
    sys.exit(1)

quiq_df = pd.read_csv(csv_path, low_memory=False)
filtered_quiq_df = quiq_df[quiq_df['Mapping_info_1'].str.contains('medical_code', na=False)]
via_df = pd.read_csv(VIA_path, low_memory=False)
unique_variable_names = filtered_quiq_df['Variable_name'].unique()
matched_columns_df = via_df[via_df['Variable_name'].isin(unique_variable_names)][['Variable_name', 'Description']]
unique_matched_columns_df = matched_columns_df.drop_duplicates(subset=['Variable_name'])

variable_pattern_frequencies = defaultdict(Counter)
for _ in range(10):
    for _, row in unique_matched_columns_df.iterrows():
        variable_name = row['Variable_name']
        description = row['Description']
        pattern_str = generate_regex_gpt4(variable_name, description)
        if pattern_str:
            variable_pattern_frequencies[variable_name][pattern_str] += 1

regex_dict = {var: re.compile(patterns.most_common(1)[0][0]) for var, patterns in variable_pattern_frequencies.items() if patterns}
validation_df = pd.DataFrame([{
    "Variable": row['Variable_name'],
    "Code": row['Value'],
    "Valid": "True" if regex_dict.get(row['Variable_name'], re.compile('^$')).match(str(row['Value'])) else "False"
} for index, row in filtered_quiq_df.iterrows()])

response_text = validate_medical_codes_gpt4(unique_matched_columns_df, validation_df)
if response_text and response_text.strip():
    try:
        response_df = parse_gpt_response_to_df(response_text)
        if response_df.empty:
            print("No valid data was returned to create DataFrame.")
            sys.exit(1)
        response_df['Result'] = response_df.apply(lambda x: f"{x['Variable']}\t{x['Code']}\t{x['Valid']}", axis=1)

        # Calculate the validity percentage from the DataFrame
        validity_percentage, valid_count, total_count = calculate_validity_percentage(response_df)
        print(f"Code Validity: {validity_percentage:.2f}%")

        # Prepare data for the text file
        data_rows = [
            "<Code Validity Summary>",
            f"Code Validity Percentage: {validity_percentage:.2f}%",
            f"Valid Codes: {valid_count}",
            f"Total Codes: {total_count}",
            "",
            "<Summary Regular Expression Suggestion>"
        ]

        # Collecting regex suggestions
        for variable_name in variable_pattern_frequencies:
            for pattern, frequency in variable_pattern_frequencies[variable_name].items():
                data_rows.append(f"Variable: {variable_name}\tPattern: {pattern}\tFrequency: {frequency}")

        # Adding the most frequent regex patterns selected
        data_rows.append("")
        data_rows.append("<Most Frequent Regex Patterns Selected>")
        for variable_name, patterns in variable_pattern_frequencies.items():
            most_frequent_pattern, frequency = patterns.most_common(1)[0]
            data_rows.append(f"Variable: {variable_name}\tSelected Pattern: {most_frequent_pattern}\tFrequency: {frequency}")

        # Prepare the GPT model validation results
        data_rows.append("")
        data_rows.append("<Final Validation Results>")
        response_df['Result'] = response_df.apply(lambda x: f"{x['Variable']}\t{x['Code']}\t{x['Valid']}", axis=1)
        data_rows.extend(response_df['Result'].tolist())

        # Save to text file
        with open("결과_code_validity.txt", "w", encoding="utf-8") as text_file:
            for row in data_rows:
                text_file.write(row + "\n")

        print("Detailed results are saved as '결과_code_validity.txt'")
    except Exception as e:
        print(f"An error occurred while processing the DataFrame: {e}")
        sys.exit(1)
else:
    print("No response or invalid response text received from GPT-4.")
    sys.exit(1)
