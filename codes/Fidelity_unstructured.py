from openai import OpenAI
import pandas as pd
import re
import yaml
import os
from tqdm import tqdm

from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("config_all.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    save_path= cfg['save_path']

    try:
        #os.mkdir(save_path + 'fidelity(unstructured)')
        os.mkdir(os.path.join(save_path, 'fidelity(unstructured)'))
    except:
        pass

    #save_path_fidelity = save_path + 'fidelity(unstructured)/'
    save_path_fidelity = os.path.join(save_path, 'fidelity(unstructured)/')

    openaiapi_key = cfg['open_api_key']
    client = OpenAI(api_key=openaiapi_key) #나
    df = pd.read_csv(cfg['csv_path'])

    df = df[df['Mapping_info_1'] == 'note_clinical']

    # Mapping_info_2에 따른 템플릿 매핑
    mapping = {
        "ADM": "입원기록",
        "DIS": "퇴원기록",
        "SUR": "수술기록",
        "EME": "응급진료기록"
    }

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
        Admission department:
        Discharge department:
        Discharge reason:
        Diagnosis:
        Summary of progression:
        Medical prescription:
        Surgery or procedure:
        Treatment result:
        Discharge plan:
        Discharge form:
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

    system_prompt = """"Input은 template과 해당 medical record야.
    Task: Report template의 항목을 보고, 해당 내용이 medical record에 있는지 판단해서 mentioned/not mentioned로 구분해줘.
    Output format
    - (template의 항목): mentioned/not mentioned"""


    def process_row(row):
        template = report_templates.get(mapping.get(row['Mapping_info_2'], ""), "")
        if not template:
            return "Invalid mapping or template missing"
        query = template + row['Value']
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
            return response.choices[0].message.content
        except Exception as e:
            return None  # 재시도를 위해 None 반환

    # 재시도 로직 추가
    def retry_process_row(row, attempts=3):
        for _ in range(attempts):
            result = process_row(row)
            if result is not None:
                return result
        return "Failed after retries"  # 최대 재시도 횟수 후 실패

    # 진행 상태 바 추가
    tqdm.pandas(desc="Processing rows")
    df['fidelity_clinical'] = df.progress_apply(retry_process_row, axis=1)


    def clinical_fidelity_score(report_text):
        # If report_text is None, return a default value (for example, 0)
        if report_text is None:
            return 0

        # Use regular expressions to split the text into "item: content" formats.
        findings_items = re.findall(r':\s*("[^"]+"|[^:\n]+)', report_text)

        # Calculate the total number of items
        total_items = len(findings_items)

        # If total_items is 0, return a default value to avoid division by zero
        if total_items == 0:
            return 0  # Or any other default value you deem appropriate

        # Calculate the number of items marked as 'not mentioned' (case insensitive)
        not_mentioned_items = sum('not mentioned' in item.lower() for item in findings_items)

        # Calculate the ratio and then subtract from 1 to return the value
        return 1 - (not_mentioned_items / total_items)


    # Filter out rows where 'fidelity_clinical' is 'Invalid mapping or template missing'
    df_filtered_clinical = df[df['fidelity_clinical'] != 'Invalid mapping or template missing']

    df_filtered_clinical['fidelity_clinical_results'] = df_filtered_clinical['fidelity_clinical'].apply(clinical_fidelity_score)


    mean_fidelity = df_filtered_clinical['fidelity_clinical_results'].mean()
    std_fidelity = df_filtered_clinical['fidelity_clinical_results'].std()


    #결과 출력
    print(f"Fidelity score_clinical (mean, %): {mean_fidelity * 100:.2f}%")
    print(f"Fidelity score_clinical (standard deviation): {std_fidelity:.2f}")
    df_filtered_clinical.to_csv(save_path_fidelity + "fidelity(unstructured)_clinical.csv", index=False, columns=['Primary_key', 'Variable_name', 'Mapping_info_2', 'Value', 'fidelity_clinical', 'fidelity_clinical_results'])
        


    

    with open("config_all.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    openaiapi_key = cfg['open_api_key']
    client = OpenAI(api_key=openaiapi_key) #나
    df = pd.read_csv(cfg['csv_path'])
    df = df[df['Mapping_info_1'] == 'note_rad']

    # Mapping_info_2에 따른 템플릿 매핑
    mapping = {
        "ACT": "CT abdomen",
        "BCT": "CT brain",
        "CCT": "CT chest",
        "SCT": "CT spine",
        "CXR": "X-ray chest",
        "AXR": "X-ray abdomen",
        "SXR": "X-ray spine",
        "ECH": "Echocardiography"
    }

    report_templates = {
        "CT abdomen": """
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
        """,  
        "CT brain": """
        Extra-axial spaces:
        Ventricular system:
        Basal cisterns:
        Cerebral parenchyma:
        Cerebellum:
        Brainstem:
        Vascular system:
        Paranasal sinuses and mastoid air cells:
        Visualized orbits:
        Bone:
        """,
        "CT chest": """
        Pulmonary Parenchyma and Airways:
        Pleural Space:
        Heart and Pericardium:
        Mediastinum and Hila:
        Thoracic Vessels:
        Osseous Structures and Chest Wall: 
        Upper Abdomen:
        Additional findings: 
        """,
        "CT spine": """
        Alignment :
        Bones :
        Intervertebral Discs :
        Spinal canal :
        Paraspinal soft tissues :
        Others :
        """,
        "X-ray chest": """
        Lungs :
        Heart :
        Mediastinum :
        """,
        "X-ray abdomen": """
        Bowel gas pattern :
        Abnormal calcifications :
        Bones :
        Others :
        """,
        "X-ray spine": """
        Alignment :
        Vertebral bodies :
        Intervertebral spaces :
        Soft Tissues :
        Others:
        """,
        "X-ray abdomen": """
        Bowel gas pattern :
        Abnormal calcifications :
        Bones :
        Others :
        """,
        "Echocardiography": """
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
    """

    }

    system_prompt = """"Everything above is a report template, and the text that follows is the actual radiology report.
    What I require from you is the following:
    Output according to the report template.
    - If an item from the template exists in the report, label it only as 'mentioned'.
    - If it doesn't exist, label it only as 'not mentioned'.
    - Never create anything outside of the items in the TEMPLATE.
    - Output format
    (template의 항목): mentioned/not mentioned"""

    def process_row(row):
        template = report_templates.get(mapping.get(row['Mapping_info_2'], ""), "")
        if not template:
            return "Invalid mapping or template missing"
        query = template + row['Value']
        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
            response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
            return response.choices[0].message.content
        except Exception as e:
            return None  # 재시도를 위해 None 반환

    # 재시도 로직 추가
    def retry_process_row(row, attempts=3):
        for _ in range(attempts):
            result = process_row(row)
            if result is not None:
                return result
        return "Failed after retries"  # 최대 재시도 횟수 후 실패

    # 진행 상태 바 추가
    tqdm.pandas(desc="Processing rows")
    df['fidelity_radiology'] = df.progress_apply(retry_process_row, axis=1)


    def radiology_fidelity_score(report_text):
        # If report_text is None, return a default value (for example, 0)
        if report_text is None:
            return 0

        # Convert report_text to lowercase for case-insensitive comparison
        report_text_lower = report_text.lower()

        # Split the report text into items using newline as separator
        findings_items = report_text_lower.split('\n')

        # Calculate the total number of items
        total_items = len([item for item in findings_items if item.strip() != ''])

        # Count the number of items marked as 'not mentioned'
        not_mentioned_items = sum('not mentioned' in item for item in findings_items)

        # Calculate and return the fidelity score
        return 1 - (not_mentioned_items / total_items) if total_items > 0 else 0

        # # If report_text is None, return a default value (for example, 0)
        # if report_text is None:
        #     return 0

        # # Convert report_text to lowercase for case-insensitive comparison
        # report_text_lower = report_text.lower()

        # # Determine the term used in the report, either "impression" or "conclusion"
        # if "impression" in report_text_lower:
        #     split_term = "impression"
        # elif "conclusion" in report_text_lower:
        #     split_term = "conclusion"
        # else:
        #     # If neither term is found, return a default value
        #     return 0

        # # Split the report text based on the identified term
        # findings_section, _ = report_text_lower.split(split_term, 1)

        # # Split "FINDINGS" section into items
        # findings_items = re.findall(r':\s*("[^"]+"|[^:\n]+)', findings_section)

        # # Calculate the total number of items
        # total_items = len(findings_items)

        # # Count the number of items marked as 'not mentioned'
        # not_mentioned_items = sum('not mentioned' in item for item in findings_items)

        # # Calculate and return the fidelity score
        # return 1 - (not_mentioned_items / total_items) if total_items > 0 else 0


    # Filter out rows where 'fidelity_radiology' is 'Invalid mapping or template missing'
    df_filtered_radiology = df[df['fidelity_radiology'] != 'Invalid mapping or template missing']

    # Apply report_fidelity_score to the filtered DataFrame
    df_filtered_radiology['fidelity_radiology_results'] = df_filtered_radiology['fidelity_radiology'].apply(radiology_fidelity_score)

    # Calculate mean and standard deviation of the fidelity scores in the filtered DataFrame
    mean_fidelity = df_filtered_radiology['fidelity_radiology_results'].mean()
    std_fidelity = df_filtered_radiology['fidelity_radiology_results'].std()

    # Print the results
    print(f"Fidelity score_radiology (mean, %): {mean_fidelity * 100:.2f}%")
    print(f"Fidelity score_radiology (standard deviation): {std_fidelity:.2f}")

    # Save the filtered and processed DataFrame to CSV
    df_filtered_radiology.to_csv(save_path_fidelity + "fidelity(unstructured)_radiology.csv", index=False, columns=['Primary_key', 'Variable_name', 'Mapping_info_2', 'Value', 'fidelity_radiology', 'fidelity_radiology_results'])




    def calculate_overall_fidelity(df_clinical, df_radiology):
        # Combine the fidelity scores from both dataframes
        combined_fidelity_scores = pd.concat([df_clinical['fidelity_clinical_results'], df_radiology['fidelity_radiology_results']])

        # Calculate mean and standard deviation
        mean_fidelity_total = combined_fidelity_scores.mean()
        std_fidelity_total = combined_fidelity_scores.std()

        # Print the results
        print(f"Fidelity score (mean, %): {mean_fidelity_total * 100:.2f}%")
        print(f"Fidelity score (standard deviation): {std_fidelity_total:.2f}")

        # Optionally, return these values if you need to use them elsewhere
        return mean_fidelity_total, std_fidelity_total

    mean_fidelity_total, std_fidelity_total = calculate_overall_fidelity(df_filtered_clinical, df_filtered_radiology)

    print("Overall Fidelity score (mean, %):", mean_fidelity_total * 100)
    print("Overall Fidelity score (standard deviation):", std_fidelity_total)

    with open(save_path_fidelity + 'fidelity_scores(mean).txt', 'w') as file:
        file.write(f"{mean_fidelity_total * 100:.3f}")
    with open(save_path_fidelity + 'fidelity_scores(std).txt', 'w') as file:
        file.write(f"{std_fidelity_total:.3f}")

    #시각화 코드 시작
    #불러와서 합치기 
    df1 = pd.read_csv(save_path_fidelity + 'fidelity(unstructured)_clinical.csv')
    df2 = pd.read_csv(save_path_fidelity + 'fidelity(unstructured)_radiology.csv')
    df1_renamed = df1.rename(columns={
        'fidelity_clinical': 'fidelity',
        'fidelity_clinical_results': 'fidelity_results'
    })
    df2_renamed = df2.rename(columns={
        'fidelity_radiology': 'fidelity',
        'fidelity_radiology_results': 'fidelity_results'
    })
    result_df = pd.concat([df1_renamed, df2_renamed], ignore_index=True)
    result_df.to_csv(save_path_fidelity + 'visualization.csv', index=False)

    #mapping_dict 정의
    mapping_dict = {
        "ACT": "CT abdomen",
        "BCT": "CT brain",
        "CCT": "CT chest",
        "SCT": "CT spine",
        "CXR": "X-ray chest",
        "AXR": "X-ray abdomen",
        "SXR": "X-ray spine",
        "ECH": "Echocardiography",
        "ADM": "Admission note",
        "DIS": "Discharge summary",
        "SUR": "Surgery note",
        "EME": "Emergency note"
    }

    #Box-plot 그리기
    df=pd.read_csv(save_path_fidelity + 'visualization.csv')
    valid_categories = ["ADM", "DIS", "SUR", "EME", "ACT", "BCT", "CCT", "SCT", "CXR", "AXR", "SXR", "ECH"]
    filtered_data = df[df['Mapping_info_2'].isin(valid_categories)]
    filtered_data['Mapping_info_2'] = filtered_data['Mapping_info_2'].map(mapping_dict)
    plt.figure(figsize=(16, 12))
    sns.boxplot(x='Mapping_info_2', y='fidelity_results', data=filtered_data)
    sns.stripplot(x='Mapping_info_2', y='fidelity_results', data=filtered_data, color='blue', jitter=True, alpha=0.5, size=12)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Fidelity Results', fontsize=12)
    plt.xticks(rotation=30)  # Rotate category labels for better visibility
    plt.ylim(0, 1)  # y축을 0부터 1까지 고정
    plt.grid(True)
    plt.savefig(save_path_fidelity + 'fidelity(unstructured)_box plot.png')

    #변수별 충실성 계산 dataframe 만들기
    df = pd.read_csv(save_path_fidelity + 'visualization.csv')
    summary_df = df.groupby('Mapping_info_2')['fidelity_results'].agg(
        Count='count',
        Fidelity_score_mean=lambda x: round(x.mean(), 3),
        Fidelity_score_std=lambda x: round(x.std(), 3)
    ).reset_index()
    summary_df['Mapping_info_2'] = summary_df['Mapping_info_2'].map(mapping_dict)
    #summary_df = summary_df.rename(columns={'Mapping_info_2': 'Category'})
    summary_df.to_csv(save_path_fidelity + 'fidelity(unstructured)_summary dataframe.csv', index=False)

    # 각 데이터프레임을 필터링하고 저장
    df=pd.read_csv(save_path_fidelity + 'visualization.csv')
    for keys, values in mapping_dict.items():
        filtered_df = df[df['Mapping_info_2'] == keys]
        filename = f"fidelity(unstructured)_{values} dataframe.csv"
        filtered_df.to_csv(save_path_fidelity + filename, index=False)
    
run()
