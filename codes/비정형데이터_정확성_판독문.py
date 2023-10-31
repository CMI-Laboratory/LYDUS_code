import openai
import pandas as pd
import signal
import ast  # 4번 문제점 반영
from itertools import islice
import time
import threading
from tqdm import tqdm
import yaml



def run():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    openai.api_key = cfg['openai_api_key']
    df = pd.read_csv(cfg['csv_path'])


    df = df[df['변수 category'] == 'radiology report']
    df = df[df['변수명'] == cfg['variable_name']]

    system_content = """
    Task:
    To systematically assess the "Impression" section of a brain CT report for critical errors that may have significant clinical implications.
    
    Procedure: 
    - Begin by understanding the entire radiology report's context, purpose, and content. This will help you appreciate the clinical relevance of the findings and the intended message.
    - Compare the "Findings" and "Impression" sections closely. Specifically, identify:
    a. Omissions in the "Impression" section that concern critical, actionable abnormalities present in the "Findings".
    b. Statements in the "Impression" section that directly contradict or misrepresent the information in the "Findings".
    - Do not be overly critical about subtle differences in verbiage or style. The emphasis should be on identifying discrepancies that could potentially lead to significant clinical misunderstandings or incorrect management decisions.
    - By narrowing down the focus on actionable abnormalities and direct contradictions, we hope to reduce false positives and concentrate on the most critical aspects of the report.
    
    Output Format (JSON):
    {
        "error 1": "{Specify the identified error clearly or state 'no error'}",
    }     
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



    def check_error1(row):
        try:
            # 문자열을 딕셔너리로 변환
            row_dict = ast.literal_eval(row)  # ast 사용
            if 'error 1' in row_dict:
                return 0 if row_dict['error 1'].lower() != 'no error' else 1
            else:
                return None  # 'error 1' 키가 없는 경우
        except (ValueError, SyntaxError):
            return None  # 문자열을 딕셔너리로 변환할 수 없는 경우

    # 'Error_result' 대신 'result_TEXT'를 사용 (3번 문제점 반영)
    df['final_result'] = df['result_TEXT'].apply(check_error1)



    # 평균값 계산
    mean_value = df['final_result'].mean()

    # 1의 비율 계산
    total_rows = len(df)
    count_ones = df['final_result'].value_counts().get(1, 0)
    ratio_of_ones = (count_ones / total_rows) * 100

    # DataFrame과 통계 출력
    print(f"No error의 비율은 {ratio_of_ones:.2f}%입니다 ({count_ones}/{total_rows})")

    # 결과를 새로운 CSV 파일에 저장
    print("Detailed results are saved as '결과_accuracy_radiology.csv'")
    df.to_csv('결과_accuracy_radiology.csv', index=False)

run()