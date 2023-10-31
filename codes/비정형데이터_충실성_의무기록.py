import openai
import pandas as pd
import signal
import re
import yaml


def run():
    import openai
    import numpy as np
    import pandas as pd
    import signal
    import re
    import random
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    openai.api_key = cfg['openai_api_key']
    df = pd.read_csv(cfg['csv_path'])


    df = df[df['변수 category'] == 'medical record']
    df = df[df['변수명'] == cfg['variable_name']]

    report_templates = {
        "입원기록": """
        입원과:
        Chief complaint:
        Present ilness:
        Past medical history :
        Social & Familty history :
        Physical exammination :
        Review of systems :
        Diagnosis :
        Treatment plan :
        """,  
        "퇴원기록": """
        입원과:
        퇴원과:
        입원 사유:
        진단명:
        경과 요약:
        약 처방:
        수술 혹은 시술명:
        치료결과:
        퇴원 후 진료 계획:
        퇴원 형태:
        """,
        "수술기록": """
        수술 전 진단명:
        수술 후 진단명:
        수술 전 수술명:
        수술 후 수술명:
        마취 종류:
        수술 관찰 소견:
        수술 절차
        """,
        "응급진료기록": """
        내원정보:
        과거력:
        투약력:
        현병력:
        검사소견:
        추정 진단명:
        치료계획:
        """
    }

    selected_column = '값'  # 무조건 '값' 컬럼에서 진행

    print("사용 가능한 report 종류:")
    for report_type in report_templates.keys():
        print(f"- {report_type}")

    selected_type = input("원하는 report 종류를 입력하세요: ")


    prompt = """"Everything above is a report template, and the text that follows is the actual medical record.
    What I require from you is the following:
    Output according to the report template.
    - If an item from the template exists in the report, label it as 'mentioned'.
    - If it doesn't exist, label it as 'not mentioned'.
    - Never create anything outside of the items in the TEMPLATE."""


    # 시간제한을 초과하면 발생하는 예외
    class TimeoutException(Exception): pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    # 타임아웃 시간을 설정합니다. 
    signal.signal(signal.SIGALRM, timeout_handler)

    # 선택한 리포트 타입이 유효한 경우
    if selected_type in report_templates:
        template = report_templates[selected_type]

        new_column = [None] * len(df[selected_column])  # Initialize the new column with None
        failed_indices = []  # Keep track of failed indices

        # First pass: Process each row, skipping those that take too long
        for index, report in enumerate(df[selected_column], start=1):
            try:
                print(f"Processing row {index}...")  # Display progress
                query = template + prompt + report

                # Start the timer
                signal.alarm(50)
                messages = [{"role": "system", "content": "You are a physician calculating fidelity of the medical report."}, {"role": "user", "content": query}]
                response = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0)
                structured_report = response['choices'][0]['message']['content']
                new_column[index-1] = structured_report  # Save the structured report

                # Stop the timer
                signal.alarm(0)
            except TimeoutException:
                print(f"Processing row {index} took too long. Skipping...")
                failed_indices.append(index-1)  # Save the index of the failed row
            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                failed_indices.append(index-1)  # Save the index of the failed row

        # Second pass: Retry the failed indices
        for index in failed_indices:
            try:
                print(f"Retrying row {index+1}...")
                query = template + "Replace the report below with a structured report in the format above. Please send '(not mentioned)' for items with no content in the report below." + df[selected_column][index]

                # Start the timer
                signal.alarm(50)
                messages = [{"role": "system", "content": "You are a physician calculating fidelity of the medical report."}, {"role": "user", "content": query}]
                response = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0)
                structured_report = response['choices'][0]['message']['content']
                new_column[index] = structured_report  # Save the structured report

                # Stop the timer
                signal.alarm(0)
            except Exception as e:
                print(f"Error processing row {index+1} on retry: {str(e)}")

        # Add the structured reports to the DataFrame
        df['Structured_Report'] = new_column

        # Processing complete
        print("Processing complete!")
    else:
        print("선택한 report 종류가 유효하지 않습니다.")

        
    def report_fidelity_score(report_text):
        # If report_text is None, return a default value (for example, 0)
        if report_text is None:
            return 0

        # 정규 표현식을 사용하여 "항목: 내용" 형태의 텍스트를 분리합니다.
        findings_items = re.findall(r':\s*("[^"]+"|[^:\n]+)', report_text)

        # 전체 항목 수 계산
        total_items = len(findings_items)

        # 'not mentioned'라고 표시된 항목 수 계산 (대소문자 구분 없이)
        not_mentioned_items = sum('not mentioned' in item.lower() for item in findings_items)

        # 비율 계산 후 1에서 뺀 값 반환
        return 1 - (not_mentioned_items / total_items)

    df['fidelity_scores'] = df['Structured_Report'].apply(report_fidelity_score)

    # 평균 및 표준 편차 계산
    average_score = df['fidelity_scores'][df['fidelity_scores'] != 0].dropna().mean()
    std_deviation = df['fidelity_scores'][df['fidelity_scores'] != 0].dropna().std()

    print(f"Fidelity score (mean): {average_score}")
    print(f"Fidelity score (standard deviation): {std_deviation}")

    print("Detailed results are saved as '결과_fidelity_medical.csv'")
    df.to_csv("결과_fidelity_medical.csv", index=False)

run()