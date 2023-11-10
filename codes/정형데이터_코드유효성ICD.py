import re
import pandas as pd

# Regular expressions for ICD-9 and ICD-10 codes
# Updated ICD-9 regex pattern
ICD9_REGEX = re.compile(r"^(?:\d{3}(?:\.\d{1,2})?|E\d{3}(?:\.\d)?|V\d{2}(?:\.\d{1,2})?)$")

# Updated ICD-10 regex pattern
ICD10_REGEX = re.compile(
    r"^[A-Za-z]\d{2}$"
    r"|^[A-Za-z]\d{2}\.\d{0,3}$"
    r"|^[A-Za-z]\d{2}\.[1-9]\d{0,1}[xX]\d$"
    r"|^[A-Za-z]\d{2}[xX]\d$",
    re.IGNORECASE
)


# Function to validate ICD codes
def is_valid_icd_code(code, icd_version):
    if icd_version == 'ICD-9':
        return bool(re.match(ICD9_REGEX, str(code)))
    elif icd_version == 'ICD-10':
        return bool(ICD10_REGEX.match(str(code)))
    return False

# Function to match ICD code variations
def match_icd_variations(value):
    if pd.isnull(value):
        return None
    value = value.lower()
    if 'icd9' in value or 'icd-9' in value:
        return 'ICD-9'
    elif 'icd10' in value or 'icd-10' in value:
        return 'ICD-10'
    return None

# Function to validate ICD codes from a CSV file
def validate_icd_from_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]

    df['ICD_Version'] = df['변수명'].apply(match_icd_variations)
    df['Validation'] = df.apply(
        lambda row: 'valid' if is_valid_icd_code(row['변수 ID'], row['ICD_Version']) else 'invalid', axis=1
    )
    df = df.dropna(subset=['ICD_Version'])  # Drop rows without ICD versions

    # Separate error details for ICD-9 and ICD-10
    icd9_error_details = df[(df['ICD_Version'] == 'ICD-9') & (df['Validation'] == 'invalid')]
    icd10_error_details = df[(df['ICD_Version'] == 'ICD-10') & (df['Validation'] == 'invalid')]

    return {
        "valid_percentage": (df[df['Validation'] == 'valid'].shape[0] / df.shape[0]) * 100,
        "total_errors": df[df['Validation'] == 'invalid'].shape[0],
        "icd9_errors": icd9_error_details.shape[0],
        "icd10_errors": icd10_error_details.shape[0],
        "icd9_error_details": icd9_error_details[['변수 ID', '변수명']].to_dict('records'),
        "icd10_error_details": icd10_error_details[['변수 ID', '변수명']].to_dict('records')
    }

def format_error_details(results, total_rows_checked, include_details=True):
    error_messages = [f"코드유효성: {results['valid_percentage']:.2f}%"]

    if include_details:
        error_messages.append("More details:")
        error_messages.append(f"Total Rows Checked: {total_rows_checked}")
        error_messages.append(f"Errors Detected: {results['total_errors']} (ICD-9: {results['icd9_errors']}, ICD-10: {results['icd10_errors']})\n")

        if results['icd9_errors']:
            error_messages.append("ICD-9 Error Breakdown:")
            for error in results['icd9_error_details']:
                error_messages.append(f"- Code: {error['변수 ID']}, 변수명: {error['변수명']}")
        else:
            error_messages.append("No ICD-9 code errors found.")

        if results['icd10_errors']:
            error_messages.append("\nICD-10 Error Breakdown:")
            for error in results['icd10_error_details']:
                error_messages.append(f"- Code: {error['변수 ID']}, 변수명: {error['변수명']}")
        else:
            error_messages.append("No ICD-10 code errors found.")

    return "\n".join(error_messages)

# The rest of your main function remains unchanged
if __name__ == "__main__":
    csv_path = "/content/icd9높은.csv"  # Update this path to your actual CSV file path
    results = validate_icd_from_csv(csv_path)
    total_rows_checked = len(pd.read_csv(csv_path))  # Get total number of rows from the CSV

    # Print Code Validity without details
    print(format_error_details(results, total_rows_checked, include_details=False))
