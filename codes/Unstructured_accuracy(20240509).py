
import pandas as pd
from openai import OpenAI
import ast  
import time
import threading
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns



def run_clinical():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    save_path= cfg['save_path']

    try:
        os.mkdir(save_path + 'accuracy(unstructured)')
    except:
        pass

    save_path_accuracy = save_path + 'accuracy(unstructured)/'


    openaiapi_key = cfg['openai_api_key']
    client = OpenAI(api_key=openaiapi_key) #나
    df = pd.read_csv(cfg['csv_path'])

    df = df[df['Mapping_info_1'] == 'note_clinical']

    system_content = """Please review the medical record provided and identify errors in the following specific categories:

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

    Format your response exactly as follows (JSON):
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
    def api_call(row, value_col='Value', results=[]):
        report = ""
        value_content = row[value_col]
        report += f"<value note> {value_content}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
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
                temperature=0
            )
            results[0] = response.choices[0].message.content
        except Exception as e:
            results[1] = e

    # Unified function to process rows
    def process_rows(value_col='Value', result_col='result'):
        start_time = time.time()
        errors_to_retry = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing rows for {result_col}"):
            result = [None, None]

            thread = threading.Thread(target=api_call, args=(row, value_col, result))
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

            thread = threading.Thread(target=api_call, args=(row, value_col, result))
            thread.start()
            thread.join(timeout=60)
            
            if not thread.is_alive() and not result[1]:
                df.loc[i, result_col] = result[0]

        elapsed_time = time.time() - start_time
        print(f"Total elapsed time for {result_col}: {elapsed_time} seconds")

    # Call the function for different scenarios
    process_rows(value_col='Value', result_col='accuracy_clinical')

    #Diagnostic과 procedure만 뽑음!!!!
    def check_error1(cell_content):
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

    # 'Error_result' 대신 'result_TEXT'를 사용 (3번 문제점 반영)
    df['accuracy_clinical_result'] = df['accuracy_clinical'].apply(check_error1)

    # 평균값 계산
    mean_value = df['accuracy_clinical_result'].mean()

    # 1의 비율 계산
    total_rows = len(df)
    count_ones = df['accuracy_clinical_result'].value_counts().get(1, 0)
    ratio_of_ones = (count_ones / total_rows) * 100

    # 평균 정확도 계산 (이미 계산된 mean_value 사용)
    mean_accuracy = mean_value  # 이미 비율 형태로 계산되었음

    # 표준 편차 계산
    std_accuracy = df['accuracy_clinical_result'].std()

    # 결과 출력
    print(f"Accuracy score_clinical (mean, %): {mean_accuracy * 100:.2f}%")
    print(f"Accuracy score_clinical (standard deviation): {std_accuracy:.2f}")

    # 결과를 새로운 CSV 파일에 저장
    df.to_csv(save_path_accuracy + 'accuracy(unstructured)_clinical.csv', index=False, columns=['Primary_key', 'Variable_name', 'Mapping_info_1', 'Value', 'accuracy_clinical', 'accuracy_clinical_result'])
    return df[['accuracy_clinical_result']]





def run_radiology():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    save_path= cfg['save_path']

    try:
        os.mkdir(save_path + 'accuracy(unstructured)')
    except:
        pass

    save_path_accuracy = save_path + 'accuracy(unstructured)/'
    

    openaiapi_key = cfg['openai_api_key']
    client = OpenAI(api_key=openaiapi_key) #나
    df = pd.read_csv(cfg['csv_path'])

    df = df[df['Mapping_info_1'] == 'note_rad']


    system_content = """
    Task:
    To systematically assess the "Impression" section of a CT report for critical errors that may have significant clinical implications.
    
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
        "error 1 reason": "{Specify the reason of error if applicable, or state 'N/A'}"
    }     
    """

    # Unified API calling function
    def api_call(row, value_col='Value', results=[]):
        report = ""
        value_content = row[value_col]
        report += f"<value note> {value_content}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
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
                temperature=0
            )
            results[0] = response.choices[0].message.content
        except Exception as e:
            results[1] = e

    # Unified function to process rows
    def process_rows(value_col='Value', result_col='result'):
        start_time = time.time()
        errors_to_retry = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing rows for {result_col}"):
            result = [None, None]

            thread = threading.Thread(target=api_call, args=(row, value_col, result))
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

            thread = threading.Thread(target=api_call, args=(row, value_col, result))
            thread.start()
            thread.join(timeout=60)
            
            if not thread.is_alive() and not result[1]:
                df.loc[i, result_col] = result[0]

        elapsed_time = time.time() - start_time
        print(f"Total elapsed time for {result_col}: {elapsed_time} seconds")

    # Call the function for different scenarios
    process_rows(value_col='Value', result_col='accuracy_radiology')


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
    df['accuracy_radiology_result'] = df['accuracy_radiology'].apply(check_error1)



    # 평균값 계산
    mean_value = df['accuracy_radiology_result'].mean()

    # 1의 비율 계산
    total_rows = len(df)
    count_ones = df['accuracy_radiology_result'].value_counts().get(1, 0)
    ratio_of_ones = (count_ones / total_rows) * 100

    # 평균 정확도 계산 (이미 계산된 mean_value 사용)
    mean_accuracy = mean_value  # 이미 비율 형태로 계산되었음

    # 표준 편차 계산
    std_accuracy = df['accuracy_radiology_result'].std()

    # 결과 출력
    print(f"Accuracy score_radiology (mean, %): {mean_accuracy * 100:.2f}%")
    print(f"Accuracy score_radiology (standard deviation): {std_accuracy:.2f}")

    # 결과를 새로운 CSV 파일에 저장
    df.to_csv(save_path_accuracy + 'accuracy(unstructured)_radiology.csv', index=False, columns=['Primary_key', 'Variable_name', 'Mapping_info_1', 'Value', 'accuracy_radiology', 'accuracy_radiology_result'])
    return df[['accuracy_radiology_result']]



def main():
    run_clinical()
    run_radiology()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    save_path= cfg['save_path']

    try:
        os.mkdir(save_path + 'accuracy(unstructured)')
    except:
        pass

    save_path_accuracy = save_path + 'accuracy(unstructured)/'

    #불러와서 합치기, 계산
    df1 = pd.read_csv(save_path_accuracy + 'accuracy(unstructured)_clinical.csv')
    df2 = pd.read_csv(save_path_accuracy + 'accuracy(unstructured)_radiology.csv')
    df1_renamed = df1.rename(columns={
        'accuracy_clinical': 'accuracy',
        'accuracy_clinical_result': 'accuracy_results'
    })
    df2_renamed = df2.rename(columns={
        'accuracy_radiology': 'accuracy',
        'accuracy_radiology_result': 'accuracy_results'
    })
    result_df = pd.concat([df1_renamed, df2_renamed], ignore_index=True)
    result_df = result_df.dropna(subset=['accuracy_results'])
    result_df.to_csv(save_path_accuracy + 'visualization.csv', index=False)

    #계산 txt
    mean_accuracy = result_df['accuracy_results'].mean().round(3)
    std_accuracy = result_df['accuracy_results'].std().round(3)
    with open(save_path_accuracy + 'accuracy(unstructured)_mean.txt', 'w') as file:
        file.write(str(mean_accuracy))
    with open(save_path_accuracy + 'accuracy(unstructured)_std.txt', 'w') as file:
        file.write(str(std_accuracy))

    #Box-plot 그리기
    df=pd.read_csv(save_path_accuracy + 'visualization.csv')
    valid_categories = ["note_rad", "note_clinical"]
    filtered_data = df[df['Mapping_info_1'].isin(valid_categories)]
    plt.figure(figsize=(6, 6))
    sns.barplot(x='Mapping_info_1', y='accuracy_results', data=filtered_data, errorbar=('ci', 95))
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Accuracy Results', fontsize=12)
    plt.savefig(save_path_accuracy + 'accuracy(unstructured)_box plot.png')


    #변수별 충실성 계산 dataframe 만들기
    df=pd.read_csv(save_path_accuracy + 'visualization.csv')
    summary_df = df.groupby('Mapping_info_1')['accuracy_results'].agg(
        Count='count',
        Accuracy_score_mean='mean',
        Accuracy_score_std='std'
    ).reset_index()
    summary_df['Accuracy_score_mean'] = (summary_df['Accuracy_score_mean'] * 100).round(3)
    summary_df['Accuracy_score_std'] = summary_df['Accuracy_score_std'].round(3)
    summary_df = summary_df.rename(columns={'Mapping_info_1': 'Category'})
    summary_df = summary_df.rename(columns={'Accuracy_score_mean': 'Accuracy_score_mean (%)'})
    summary_df.to_csv(save_path_accuracy + 'accuracy(unstructured)_summary dataframe.csv', index=False)

if __name__ == "__main__":
    main()