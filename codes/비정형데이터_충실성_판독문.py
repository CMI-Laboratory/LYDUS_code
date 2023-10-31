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
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    openai.api_key = cfg['openai_api_key']
    df = pd.read_csv(cfg['csv_path'])


    df = df[df['변수 category'] == 'radiology report']
    df = df[df['변수명'] == cfg['variable_name']]

    report_templates = {
        "CT abdomen": """
        Contrast media:
        Clinical information:
        Comparison:

        FINDINGS
        Liver :
        Gallbladder :
        Spleen :
        Pancreas :
        Adrenals :
        Kidneys :
        Bowel :
        Mesentery/Peritoneum :
        Nodes :
        Pelvis :
        Bone windows :
        Vasculature :
        Soft Tissues :

        IMPRESSION
        """,  
        "CT brain": """
        Contrast media:
        Clinical information:
        Comparison:

        FINDINGS
        Extra-axial spaces:
        Intracranial hemorrhage:
        Ventricular system:
        Basal cisterns:
        Cerebral parenchyma:
        Midline shift:
        Cerebellum:
        Brainstem:
        Calvarium:
        Vascular system:
        Paranasal sinuses and mastoid air cells:
        Visualized orbits:
        Visualized upper cervical spine:
        Sella:
        Skull base:

        IMPRESSION
        """,
        "CT chest": """
        Contrast media:
        Clinical information:
        Comparison:

        FINDINGS
        Pulmonary Parenchyma and Airways:
        Pleural Space:
        Heart and Pericardium:
        Mediastinum and Hila:
        Thoracic Vessels:
        Osseous Structures and Chest Wall: 
        Upper Abdomen:
        Additional findings: 
        
        IMPRESSION
        """,
        "CT spine": """
        Contrast media:
        Clinical information:
        Comparison:

        FINDINGS
        Alignment :
        Bones :
        Intervertebral Discs :
        Spinal canal :
        Paraspinal soft tissues :
        Others :

        IMPRESSION
        """,
        "X-ray chest": """
    Clinical information:
    Comparison:

    FINDINGS
    Lungs :
    Heart :
    Mediastinum :

    IMPRESSION
    """,
    "X-ray abdomen": """
    Clinical information:
    Comparison:

    FINDINGS
    Bowel gas pattern :
    Abnormal calcifications :
    Bones :
    Others :

    IMPRESSION
    """,
    "X-ray spine": """
    Clinical information:
    Comparison:

    FINDINGS
    Alignment :
    Vertebral bodies :
    Intervertebral spaces :
    Soft Tissues :
    Others:

    IMPRESSION
    """,
    "X-ray abdomen": """
    Clinical information:
    Comparison:

    FINDINGS
    Bowel gas pattern :
    Abnormal calcifications :
    Bones :
    Others :

    IMPRESSION
    """,
    "Echocardiography": """
    Clinical information:
    Comparison:

    FINDINGS
    Left ventricle :
    Left atrium :
    Right atrium :
    Right ventricle :
    Aortic valve :
    Mitral valve :
    Tricuspid valve :
    Pulmonic valve :
    Pericardium :
    Aorta :
    Pulmonary artery :
    Inferior vena cava and pulmonary veins :

    IMPRESSION
    """

    }

    selected_column = '값'

    print("사용 가능한 report 종류:")
    for report_type in report_templates.keys():
        print(f"- {report_type}")

    selected_type = input("원하는 report 종류를 입력하세요: ")

    prompt = """"Everything above is a report template, and the text that follows is the actual radiology report.
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
                signal.alarm(30)
                messages = [{"role": "system", "content": "You are a radiologist structuring radiology report."}, {"role": "user", "content": query}]
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
                query = template + prompt + df[selected_column][index]

                # Start the timer
                signal.alarm(30)
                messages = [{"role": "system", "content": "You are a radiologist structuring radiology report."}, {"role": "user", "content": query}]
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

        # Convert report_text to lowercase for case-insensitive comparison
        report_text = report_text.lower()

        # "FINDINGS" 부분과 "IMPRESSION" 부분을 분리
        findings_section, _ = report_text.split("impression", 1)

        # "FINDINGS" 부분을 항목으로 분리
        findings_items = re.findall(r':\s*("[^"]+"|[^:\n]+)', findings_section)

        # 전체 항목 수 계산
        total_items = len(findings_items)

        # 'not mentioned'라고 표시된 항목 수 계산
        not_mentioned_items = sum('not mentioned' in item for item in findings_items)

        # 비율 계산 후 1에서 뺀 값 반환
        return 1 - (not_mentioned_items / total_items)


    df['fidelity_scores'] = df['Structured_Report'].apply(report_fidelity_score)


    mean_fidelity = df['fidelity_scores'].mean()
    std_fidelity = df['fidelity_scores'].std()

    print(f"Fidelity score (mean): {mean_fidelity:.2f}")
    print(f"Fidelity score (standard deviation): {std_fidelity:.2f}")



    print("Detailed results are saved as '결과_fidelity_radiology.csv'")

    df.to_csv("결과_fidelity_radiology.csv", index=False)
        
run()