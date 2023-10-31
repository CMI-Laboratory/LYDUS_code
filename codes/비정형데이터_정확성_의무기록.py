import openai
import pandas as pd
import signal
import yaml


def run():
    import pandas as pd
    from itertools import islice
    import openai
    import time
    import threading
    from tqdm import tqdm
    import yaml

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    openai.api_key = cfg['openai_api_key']

    df = pd.read_csv(cfg['csv_path'])

    df = df[df['변수 category'] == 'medical record']

    df = df[df['변수명'] == cfg['variable_name']]
 

    system_content = """
    You are a medical expert detecting errors in medical record.
    Please review the medical record provided and identify errors in the following specific categories:

    1) Spelling or grammatical error
    2) Diagnostic Information Errors: 
    - Incorrect or missing disease diagnosis (eg. DM type 1 —> DM type 2), inaccuracies in the anatomic locations mentioned (eg. ascending colon —> rectum), discrepancies in locations (eg. right —> left).
    - For example, right/left side errors.
    3) Drug Information Errors
    - Incorrect or missing in prescribed drugs in report.
    4) Procedure Information Errors: 
    - Incorrect or missing procedure names, inaccuracies in the anatomic locations mentioned, discrepancies in locations.
    5)  Demographic Information Errors:
    - Incorrect or missing patient details such as name, age, or sex.
    6) Date Information Errors
    - Incorrect or missing dates, chronological errors.

    Format your response exactly as follows:
    - Spelling or Grammatical Errors: Yes/No (Brief Reason)
    - Diagnostic Information Error: Yes/No (Brief Reason)
    - Drug Information Error: Yes/No (Brief Reason)
    - Procedure Information Error: Yes/No (Brief Reason)
    - Demographic Information Error: Yes/No (Brief Reason)
    - Date Information Error: Yes/No (Brief Reason)

    Note: 
    -Limit your explanation for each error to fewer than 5 words. 
    -Only report errors that fall into these 6 specified categories. 
    -If multiple errors occur within a single category, number them.
    -Medication instruction or treatment plan or may change between admission and discharge, but the diagnosis and treatment names should remain the same.
    categories. If multiple errors occur within a single category, number them.
    """


    # Unified API calling function
    def api_call(row, discharge_col='discharge', results=[]):
        report = ""
        
        discharge_content = row[discharge_col]
        report += f"<discharge note> {discharge_content}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": report
                    }
                ],
                temperature=0.1
            )
            results[0] = response['choices'][0]['message']['content']
        except Exception as e:
            results[1] = e

    # Unified function to process rows
    def process_rows(discharge_col='discharge', result_col='result'):
        start_time = time.time()
        errors_to_retry = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing rows for {result_col}"):
            result = [None, None]

            thread = threading.Thread(target=api_call, args=(row, discharge_col, result))
            thread.start()
            thread.join(timeout=60)

            if thread.is_alive():
                print(f"Timeout at row {i}. Moving to the next row.")
                errors_to_retry.append(i)
                thread.join()
            elif result[1]:
                print(f"Error at row {i} with exception {str(result[1])}")
                errors_to_retry.append(i)
            else:
                df.loc[i, result_col] = result[0]

        # Retry for rows that had errors
        for i in errors_to_retry:
            row = df.iloc[i]
            result = [None, None]

            thread = threading.Thread(target=api_call, args=(row, discharge_col, result))
            thread.start()
            thread.join(timeout=60)
            
            if not thread.is_alive() and not result[1]:
                df.loc[i, result_col] = result[0]

        elapsed_time = time.time() - start_time
        print(f"Total elapsed time for {result_col}: {elapsed_time} seconds")

    # Call the function for different scenarios
    process_rows(discharge_col='값', result_col='result_TEXT')
    #process_rows(discharge_col='error_discharge', result_col='result_p2_error_only discharge')



    # 각 행에서 Diagnostic과 Procedure 정보만 추출하여 error가 no인 비율을 계산하는 함수
    def calculate_no_error_ratio(cell_content):
        lines = cell_content.split('\n')
        columns_to_check = ['Diagnostic Information Error', 'Procedure Information Error']
        total_count = 0
        no_error_count = 0
        
        for line in lines:
            for col in columns_to_check:
                if col in line:
                    total_count += 1
                    if 'No' in line:
                        no_error_count += 1
        
        if total_count == 0:
            return None  # 이 경우는 없어야 하지만, 혹시 모르니까 처리해 둡니다.
        
        return no_error_count / total_count

    # 새로운 열을 생성
    for column_name in ['result_TEXT']:
        new_column_name = "Diagnostic_Procedure_NoErrorRatio"
        df[new_column_name] = df[column_name].apply(calculate_no_error_ratio)

    # No error의 비율 계산
    total_rows = len(df)
    count_no_errors = (df['Diagnostic_Procedure_NoErrorRatio'] == 1).sum()
    ratio_of_no_errors = (count_no_errors / total_rows) * 100

    # DataFrame과 통계 출력
    print(f"No error의 비율은 {ratio_of_no_errors:.2f}%입니다 ({count_no_errors}/{total_rows})")


    # 결과를 새로운 CSV 파일에 저장
    print("Detailed results are saved as '결과_accuracy_medical.csv'")

    df.to_csv('결과_accuracy_medical.csv', index=False)

run()