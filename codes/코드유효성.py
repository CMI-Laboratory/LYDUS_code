import re
import pandas as pd
import sys

# Definitions of ICD regex patterns
ICD9_DIAGNOSIS_REGEX = re.compile(r"^(?:\d{3}(?:\.\d{1,2})?|E\d{3}(?:\.\d)?|V\d{2}(?:\.\d{1,2})?)$")
ICD10_DIAGNOSIS_REGEX = re.compile(
    r"^[A-Za-z]\d{2}$"
    r"|^[A-Za-z]\d{2}\.\d{0,3}$"
    r"|^[A-Za-z]\d{2}\.[1-9]\d{0,1}[xX]\d$"
    r"|^[A-Za-z]\d{2}[xX]\d$",
    re.IGNORECASE
)
ICD9_PROCEDURE_REGEX = re.compile(r"^\d{2,3}(\.\d{1,2})?$")
ICD10_PROCEDURE_REGEX = re.compile(r"^[0-9A-Z]{7}$", re.IGNORECASE)

# Function to validate ICD codes
def is_valid_icd_code(code, category):
    code = str(code).replace(' ', '')
    if 'ICD9_Dx' in category:
        return bool(ICD9_DIAGNOSIS_REGEX.match(code))
    elif 'ICD10_Dx' in category:
        return bool(ICD10_DIAGNOSIS_REGEX.match(code))
    elif 'ICD9_Px' in category:
        return bool(ICD9_PROCEDURE_REGEX.match(code))
    elif 'ICD10_Px' in category:
        return bool(ICD10_PROCEDURE_REGEX.match(code))
    return False

# Function to validate ICD codes from a CSV file and generate a summary
def validate_icd_from_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    df['Mapping_info_1'] = df['Mapping_info_1'].astype(str)
    df['Validation'] = df.apply(
        lambda row: 'valid' if is_valid_icd_code(row['Value'], row['Mapping_info_1']) else 'invalid', axis=1
    )

    summary = {
        'ICD-9 Diagnosis': {'Total': 0, 'Correct': 0, 'Error': 0, 'Details': []},
        'ICD-10 Diagnosis': {'Total': 0, 'Correct': 0, 'Error': 0, 'Details': []},
        'ICD-9 Procedure': {'Total': 0, 'Correct': 0, 'Error': 0, 'Details': []},
        'ICD-10 Procedure': {'Total': 0, 'Correct': 0, 'Error': 0, 'Details': []}
    }

    for _, row in df.iterrows():
        category = 'ICD-9 Diagnosis' if 'ICD9_Dx' in row['Mapping_info_1'] else \
                   'ICD-10 Diagnosis' if 'ICD10_Dx' in row['Mapping_info_1'] else \
                   'ICD-9 Procedure' if 'ICD9_Px' in row['Mapping_info_1'] else \
                   'ICD-10 Procedure' if 'ICD10_Px' in row['Mapping_info_1'] else None
        if category:
            summary[category]['Total'] += 1
            if row['Validation'] == 'valid':
                summary[category]['Correct'] += 1
            else:
                summary[category]['Error'] += 1
                summary[category]['Details'].append((row['Value'], row['Variable_name'], row.get('Patient_number', 'N/A')))

    total_valid = sum(cat['Correct'] for cat in summary.values())
    total_count = sum(cat['Total'] for cat in summary.values())
    valid_percentage = (total_valid / total_count) * 100 if total_count else 0

    return {**summary, "valid_percentage": valid_percentage}

# Function to format and print the validation summary
def format_error_details(results, include_details=True):
    error_messages = [f"ICD Code Validity: {results['valid_percentage']:.2f}%\n"]
    if include_details:
        for category, info in results.items():
            if category == 'valid_percentage':
                continue
            error_messages.append(f"{category} - Total: {info['Total']}, Correct: {info['Correct']}, Error: {info['Error']}")
            if info['Error'] > 0:
                error_messages.append(f"{category} Error Details:")
                for detail in info['Details']:
                    error_messages.append(f"- Code: {detail[0]}, Variable Name: {detail[1]}, Patient Number: {detail[2]}")
                error_messages.append("")
    return "\n".join(error_messages)

def run(file_path):
    validation_results = validate_icd_from_csv(file_path)
    error_details = format_error_details(validation_results, include_details=True)
    print(error_details)

# Entry point for script execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 코드유효성.py <path_to_csv_file>")
        sys.exit(1)
    else:
        csv_file_path = sys.argv[1]
        run(csv_file_path)
