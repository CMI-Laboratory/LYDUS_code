import yaml
import argparse
import pandas as pd
import numpy as np
import openai
from tqdm import tqdm
import datetime
from dateutil.parser import parse
import gc
import sys
import os
import contextlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LYDUS_Completeness import get_completeness
from LYDUS_Date_Validity import get_date_validity
from LYDUS_Classification import get_classification
from LYDUS_Preciseness import get_preciseness
from LYDUS_Range_Validity import get_range_validity
from LYDUS_Format_Validity import get_code_validity
from LYDUS_Sequence_Validity import get_sequence_validity
from LYDUS_Cross_Sectional_Consistency import get_cross_sectional_consistency
from LYDUS_Time_Series_Consistency import get_time_series_consistency
from LYDUS_Fidelity import get_structured_fidelity
from LYDUS_Note_Fidelity import get_unstructured_fidelity
from LYDUS_Note_Accuracy import get_unstructured_accuracy
from LYDUS_Class_Diversity import get_class_diversity
from LYDUS_Instance_Diversity import get_instance_diversity
from LYDUS_Vocabulary_Diversity import get_vocabulary_diversity
from LYDUS_Sentence_Diversity import get_sentence_diversity
from LYDUS_Logical_Accuracy import get_logical_accuracy

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield

def make_var_list_text(var_list): # 그룹핑 위한 - 변수 리스트 txt화
    var_list_text = ''
    for var_name in var_list:
        var_list_text += f"{var_name}, "
    var_list_text = var_list_text[:-2]
    return var_list_text

def llm_ask_column(client, model_ver, var_list, target_concept): # 그룹핑 위한 - 그룹 기준 변수 질의
    system_prompt = f'''You are a medical data expert.
    A list of variable names will be provided.
    From the provided variables, select exactly one variable that is most relevant to **{target_concept}**.
    Respond with **only** the variable name, no additional explanation.
    And return it **exactly as it appears** in the provided list.
    If no appropriate variable is found, respond with 'None'.'''

    user_prompt = f'''List of variable names : {var_list}'''

    response = client.chat.completions.create(
        model = model_ver,
        messages = [{'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : user_prompt}],
        temperature = 0
    )

    var_name = response.choices[0].message.content
    if var_name == 'None':
        return None

    #var_name = var_name.replace("'", '')
    return var_name




def run_all_metrics(g, quiq, via, config): ####### 그룹별로 돌아감.
    """
    Runs all available metrics on the provided dataframe.
    Returns a dictionary of {metric_name: score}.
    """
    scores = {}

    print('<Range Validity>')
    gc.collect()
    try:
        with suppress_output():
          _, df_summary, _ = get_range_validity(quiq)
        total_num = df_summary['Total_num'].sum()
        outlier_num = df_summary['Outlier_total_num'].sum()

        scores[('Range_Validity', g)] = round((total_num - outlier_num) / total_num * 100, 2)
    except Exception as e:
        print(f"Error calculating Range Validity: {e}")
        scores[('Range_Validity', g)] = np.nan
    

    print('<Date Validity>')
    gc.collect()
    try:
        with suppress_output():
          _, df_summary = get_date_validity(quiq, config['model_ver'], config['api_key'])
        total_date = df_summary['Total_date'].sum()
        invalid_date = df_summary['Invalid_date'].sum()

        scores[('Date Validity', g)] = round((total_date - invalid_date) / total_date * 100, 2)
    except Exception as e:
        print(f"Error calculating Date Validity: {e}")
        scores[('Date Validity', g)] = np.nan


    print('<Format Validity>')
    gc.collect()
    try:
        with suppress_output():
          _, df_summary, _ = get_code_validity(quiq, via, config['model_ver'], config['api_key'])
        total_code = df_summary['Total_code'].sum()
        invalid_code = df_summary['Invalid_code'].sum()
        format_validity = (total_code - invalid_code) / total_code * 100
        format_validity = round(format_validity, 2)

        scores[('Format Validity', g)] = format_validity
    except Exception as e:
        print(f"Error calculating Format Validity: {e}")
        scores[('Format Validity', g)] = np.nan


    print('<Sequence Validity>')
    gc.collect()
    try:
        with suppress_output():
          _, df_summary = get_sequence_validity(quiq, config['model_ver'], config['api_key'])
        total_num = df_summary['Total_num'].sum()
        invalid_num = df_summary['Invalid_num'].sum()
        valid_num = total_num - invalid_num
        sequence_validity = valid_num / total_num * 100
        sequence_validity = round(sequence_validity, 2)

        scores[('Sequence Validity', g)] = sequence_validity
    except Exception as e:
        print(f"Error calculating Sequence Validity: {e}")
        scores[('Sequence Validity', g)] = np.nan


    print('<Completeness>')
    gc.collect()
    try:
        with suppress_output():
          df_summary = get_completeness(quiq)
        total_num = df_summary['Total_num'].sum()
        null_num = df_summary['Null_num'].sum()

        scores[('Completeness', g)] = round((total_num - null_num) / total_num * 100, 2)
    except Exception as e:
        print(f"Error calculating Completeness: {e}")
        scores[('Completeness', g)] = np.nan

    
    print('<Logical Accuracy>')
    gc.collect()
    try:
        with suppress_output():
          var_list_target, dict_total, dict_outlier = get_logical_accuracy(quiq, config['model_ver'], config['api_key'],
                                                                       config['operation_type_manual'], config['target_variable'],
                                                                       config['automatic_num'], config['recommend_num'])
        outlier_num = 0
        total_num = 0
        for idx, var_name_target in enumerate(var_list_target) :
            total_num += len(dict_total[var_name_target])
            outlier_num += len(dict_outlier[var_name_target])
        logical_accuracy = (total_num - outlier_num) / total_num * 100
        logical_accuracy = round(logical_accuracy, 2)

        scores[('Logical Accuracy', g)] = logical_accuracy
    except Exception as e:
        print(f"Error calculating Logical Accuracy: {e}")
        scores[('Logical Accuracy', g)] = np.nan

    
    print('<Cross Sectional Consistency>')
    gc.collect()
    try:
        with suppress_output():
          average_consistency, _, _ = get_cross_sectional_consistency(quiq, via, config['model_ver'], config['api_key'])
          average_consistency = round(average_consistency * 100, 2)

        scores[('Cross Sectional Consistency', g)] = average_consistency
    except Exception as e:
        print(f"Error calculating Cross Sectional Consistency: {e}")
        scores[('Cross Sectional Consistency', g)] = np.nan

  
    print('<Time Series Consistency>')
    gc.collect()
    try:
        with suppress_output():
          _, df_summary = get_time_series_consistency(config['save_path'], quiq)
        total_time_point = df_summary['Total_time_point'].sum()
        change_point = df_summary['Change_point'].sum()
        time_series_consistency = ((total_time_point - change_point) / total_time_point * 100)
        time_series_consistency = round(time_series_consistency, 2)

        scores[('Time Series Consistency', g)] = time_series_consistency
    except Exception as e:
        print(f"Error calculating Time Series Consistency: {e}")
        scores[('Time Series Consistency', g)] = np.nan
    

    print('<Class Diversity>')
    gc.collect()
    try:
        with suppress_output():
          df_summary, _ = get_class_diversity(quiq)
        total_num = df_summary['Total Number of Data'].sum()
        class_diversity_mul_num = (df_summary['Total Number of Data'] * df_summary['Class_diversity (%)']).sum()
        weighted_class_diversity = class_diversity_mul_num / total_num
        weighted_class_diversity = round(weighted_class_diversity, 2)
        
        scores[('Class Diversity', g)] = weighted_class_diversity
    except Exception as e:
        print(f"Error calculating Class Diversity: {e}")
        scores[('Class Diversity', g)] = np.nan


    print('<Instance Diversity>')
    gc.collect()
    try:
        with suppress_output():
          df_summary, _, weighted_simpson_diversity = get_instance_diversity(quiq)
        scores[('Instance Diversity', g)] = weighted_simpson_diversity
    except Exception as e:
        print(f"Error calculating Instance Diversity: {e}")
        scores[('Instance Diversity', g)] = np.nan


    print('<Fidelity>')
    gc.collect()
    try:
        with suppress_output():
          df_summary = get_structured_fidelity(quiq)
        total_num = df_summary['Patient_num'].sum()
        mul_sum = (df_summary['Patient_num'] * df_summary['Mean']).sum()
        weighted_fidelity = mul_sum / total_num
        weighted_fidelity = round(weighted_fidelity, 2)

        scores[('Fidelity', g)] = weighted_fidelity
    except Exception as e:
        print(f"Error calculating Fidelity: {e}")
        scores[('Fidelity', g)] = np.nan


    print('<Preciseness>')
    gc.collect()
    try:
        with suppress_output():
          df_summary, _ = get_preciseness(quiq)
        total_num = df_summary['Total_num'].sum()
        preciseness_mul_num_sum = (df_summary['Total_num'] * df_summary['Preciseness (%)']).sum()
        preciseness = round(preciseness_mul_num_sum / total_num, 2)

        scores[('Preciseness', g)] = preciseness
    except Exception as e:
        print(f"Error calculating Preciseness: {e}")
        scores[('Preciseness', g)] = np.nan


    print('<Classification>')
    gc.collect()
    try:
        with suppress_output():
          _, _, (weighted_accuracy, weighted_precision, weighted_recall, weighted_f1score, weighted_auroc) = get_classification(quiq)
        
        scores[('Accuracy', g)] = weighted_accuracy
        scores[('Precision', g)] = weighted_precision
        scores[('Recall', g)] = weighted_recall
        scores[('F1score', g)] = weighted_f1score
        scores[('AUROC', g)] = weighted_auroc
    except Exception as e:
        print(f"Error calculating Classification: {e}")
        scores[('Accuracy', g)] = np.nan
        scores[('Precision', g)] = np.nan
        scores[('Recall', g)] = np.nan
        scores[('F1score', g)] = np.nan
        scores[('AUROC', g)] = np.nan


    print('<Note Fidelity>')
    gc.collect()
    try:
        with suppress_output():
          _, _, df_results, _ = get_unstructured_fidelity(quiq, config['model_ver'], config['api_key'])
        
        fidelity_scores = df_results['Fidelity_results']
        mean_fidelity = round(fidelity_scores.mean(), 2)

        scores[('Note Fidelity', g)] = mean_fidelity
    except Exception as e:
        print(f"Error calculating Note Fidelity: {e}")
        scores[('Note Fidelity', g)] = np.nan


    print('<Note Accuracy>')
    gc.collect()
    try:
        with suppress_output():
          _, _, df_results, _ = get_unstructured_accuracy(quiq, config['model_ver'], config['api_key'])
        
        accuracy_scores = df_results['Accuracy_results']
        mean_accuracy = round(accuracy_scores.mean(), 2)

        scores[('Note Accuracy', g)] = mean_accuracy
    except Exception as e:
        print(f"Error calculating Note Accuracy: {e}")
        scores[('Note Accuracy', g)] = np.nan


    print('<Vocabulary Diversity>')
    gc.collect()
    try:
        with suppress_output():
          vocabulary_diversity, _, _, _ = get_vocabulary_diversity(quiq, top_n=10)
        
        scores[('Vocabulary Diversity', g)] = vocabulary_diversity
    except Exception as e:
        print(f"Error calculating Vocabulary Diversity: {e}")
        scores[('Vocabulary Diversity', g)] = np.nan


    print('<Sentence Diversity>')
    gc.collect()
    try:
        with suppress_output():
          sentence_diversity, _, _, _ = get_sentence_diversity(quiq, top_n=10)
        
        scores[('Sentence Diversity', g)] = sentence_diversity
    except Exception as e:
        print(f"Error calculating Sentence Diversity: {e}")
        scores[('Sentence Diversity', g)] = np.nan

    return scores





def get_bias_detection(quiq, via, config):


    # 사전 설정
    client = openai.OpenAI(api_key=config['api_key'])
    # quiq
    # via
    df_sex_results = pd.DataFrame()
    df_race_results = pd.DataFrame()
    df_age_results = pd.DataFrame()


    # Group별 컬럼명 인식
    print("Identify the variables needed for grouping")

    var_list = quiq['Variable_name'].unique()
    var_list_text = make_var_list_text(var_list)

    # Sex
    sex_col = llm_ask_column(client, model_ver, var_list_text, "biological sex")
    print(f"Sex column: {sex_col}")

    # Race
    race_col = llm_ask_column(client, model_ver, var_list_text, "race")
    print(f"Race column: {race_col}")

    # Birth Date
    birth_col = llm_ask_column(client, model_ver, var_list_text, "date of birth")
    print(f"Birth Date column: {birth_col}")


    # 개인의 Group 정보 수집
    print("Collect patient's demographics")
    # 아까 얻어낸 demographic 변수에 해당하는 column들 수집해서 pivot
    demo_df = quiq[quiq['Variable_name'].isin([sex_col, race_col, birth_col])] 
    demo_pivot = demo_df.pivot_table(index='Patient_id', columns='Variable_name', values='Value', aggfunc='first').reset_index()

    del demo_df
    gc.collect()

    # 컬럼명 rename
    cols_map = {} # dictionary
    if sex_col: cols_map[sex_col] = 'Sex'
    if race_col: cols_map[race_col] = 'Race'
    if birth_col: cols_map[birth_col] = 'BirthDate'
    demo_pivot.rename(columns=cols_map, inplace=True)

    # Demographic 정보 원래 QUIQ에 붙여주기 (환자 id 기준)
    df_merged = pd.merge(quiq, demo_pivot, on='Patient_id', how='left')
    
    del quiq
    del demo_pivot
    gc.collect()


    # Age 계산
    # 현재는 birthdate가 있는 경우에 대해서만 계산됨.
    print("Calculate age")
    if 'BirthDate' in df_merged.columns: #If BirthDate가 있으면 Age 계산
        df_merged['BirthDate'] = pd.to_datetime(df_merged['BirthDate'], errors='coerce')
        df_merged['Event_date'] = pd.to_datetime(df_merged['Event_date'], errors='coerce')
        df_merged['Age_Value'] = (df_merged['Event_date'].dt.year - df_merged['BirthDate'].dt.year # 기본
                                  - ((df_merged['Event_date'].dt.month < df_merged['BirthDate'].dt.month) | # 이벤트가 생일보다 빠름 - 월이 빠름 
                                    ((df_merged['Event_date'].dt.month == df_merged['BirthDate'].dt.month) # 이벤트가 생일보다 빠름 - 월 같고 일이 빠름
                                    & (df_merged['Event_date'].dt.day < df_merged['BirthDate'].dt.day))))

        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
        labels = [
            '0-9', '10-19', '20-29', '30-39',
            '40-49', '50-59', '60-69', '70-79', '80+'
        ]
        df_merged['Age_Group'] = pd.cut(
            df_merged['Age_Value'],
            bins=bins,
            labels=labels,
            right=False
        )


    # Sex, Race, Age에 대해 nulldrop
    print("Dropna")
    if 'Sex' in df_merged.columns :
      df_merged = df_merged.dropna(subset = ['Sex'])
    if 'Race' in df_merged.columns :
      df_merged = df_merged.dropna(subset = ['Race'])
    if 'Age_Group' in df_merged.columns :
      df_merged = df_merged.dropna(subset = ['Age_Group'])


    # group별 metric 값 계산
    # Sex Groups
    if 'Sex' in df_merged.columns :
      for g in df_merged['Sex'].unique(): # 그룹별로 돌아가면서
        print('\n########## Sex - ', g, '##########')
        gc.collect()
        temp_group = df_merged[df_merged['Sex'] == g]
        temp_results = pd.Series(run_all_metrics(g, temp_group, via, config)).unstack()
        df_sex_results = pd.concat([df_sex_results, temp_results], axis = 1)
      
      df_sex_results['Mean'] = df_sex_results.mean(axis = 1)
      df_sex_results['GDI'] = df_sex_results.sub(df_sex_results['Mean'], axis=0).abs().max(axis=1)

    # Race Groups
    if 'Race' in df_merged.columns :
      for g in df_merged['Race'].unique():
        print('\n########## Race - ', g, '##########')
        gc.collect()
        temp_group = df_merged[df_merged['Race'] == g]
        temp_results = pd.Series(run_all_metrics(g, temp_group, via, config)).unstack()
        df_race_results = pd.concat([df_race_results, temp_results], axis = 1)

      labels = sorted( sorted(df_merged['Race'].unique()))
      df_race_results = df_race_results[labels]
      df_race_results['Mean'] = df_race_results.mean(axis = 1)
      df_race_results['GDI'] = df_race_results.sub(df_race_results['Mean'], axis=0).abs().max(axis=1)

    # Age Groups
    if 'Age_Group' in df_merged.columns :
      for g in df_merged['Age_Group'].unique():
        print('\n########## Age - ', g, '##########')
        gc.collect()
        temp_group = df_merged[df_merged['Age_Group'] == g]
        temp_results = pd.Series(run_all_metrics(g, temp_group, via, config)).unstack()
        df_age_results = pd.concat([df_age_results, temp_results], axis = 1)
      
      labels = sorted(df_merged['Age_Group'].unique())
      df_age_results = df_age_results[labels]
      df_age_results['Mean'] = df_age_results.mean(axis = 1)
      df_age_results['GDI'] = df_age_results.sub(df_age_results['Mean'], axis=0).abs().max(axis=1)


    return df_sex_results, df_race_results, df_age_results






if __name__ == '__main__': # 아래 내용에 tab
    print('<LYDUS - Bias Detection>')
    
    # py 환경에서 아래 코드 구동
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    quiq_path = config.get('quiq_path')
    via_path = config.get('via_path')
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')
    api_key = config.get('api_key')
    
    operation_type_manual = config.get('operation_type_manual')
    target_variable = config.get('target_variable')
    automatic_num = config.get('automatic_num')
    recommend_num = config.get('recommend_num')
    
    quiq = pd.read_csv(quiq_path)
    via = pd.read_csv(via_path)
    
    config = {'model_ver' : model_ver,
              'api_key' : api_key,
              'operation_type_manual' : operation_type_manual,
              'target_variable' : target_variable,
              'automatic_num' : automatic_num,
              'recommend_num' : recommend_num,
              'save_path' : -1}

    df_sex_results, df_race_results, df_age_results = get_bias_detection(quiq, via, config)

    df_sex_results.to_csv(save_path + '/bias_sex_summary.csv', index=False)
    df_race_results.to_csv(save_path + '/bias_race_summary.csv', index=False)
    df_age_results.to_csv(save_path + '/bias_age_summary.csv', index=False)
    
    print('\n<SUCCESS>')
