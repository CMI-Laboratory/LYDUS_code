import yaml
import argparse
import datetime
import pandas as pd
import openai
from dateutil.parser import parse
from tqdm import tqdm
import gc


SYSTEM_CONTENT = """We will evaluate the quality of the medical data
I want to verify that the date given is valid.
Please answer with 'yes' or 'no'.
No other answer than 'yes' or 'no'.
"""

def _gpt_chat(client, model, user_content, temperature=0, max_tokens=1000, n=1):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": user_content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=n
    )
    return [choice.message.content for choice in response.choices]

def _valid_date_custom(date_string): 
    formats = ["%Y년 %m월 %d일", "%Y %m월 %d일", "%Y년 %m %d일", "%Y년 %m월 %d", "%Y년 %m월 %d", "%Y %m월 %d", "%Y %m %일"]
    for date_format in formats:
        try:
            datetime.datetime.strptime(date_string, date_format)
            return True
        except ValueError:
            pass
    return False
    
def _is_valid_date(date_string): 
    if isinstance(date_string, pd.Timestamp):
        return True
    try:
        parse(date_string)
        return True
    except ValueError:
        return False

'''def _save_data(df, variable_name, total_dates, results):
    invalid_dates = len(df)
    date_validation_percentage = 1-(invalid_dates / total_dates) if total_dates > 0 else 0
    date_validation_percentage = round(date_validation_percentage * 100, 2)
    print(f'{variable_name} Date validation: {date_validation_percentage}\n')
    results.append({
        'Variable_name': variable_name,
        'Total number of dates': total_dates,
        'Number of invalid dates': invalid_dates,
        'Date_validity': date_validation_percentage
        })
    return results'''

def _extract_date_data_mapping(df):
    record_dates_df = df[['Original_table_name', 'Event_date']].dropna() 
    record_dates_df['Variable_name'] = 'Event_date'
    record_dates_df.rename(columns={'Event_date': 'Date_value'}, inplace=True)
    
    date_mapping_df = df[df['Mapping_info_1'].str.contains('date', case=False, na=False)]
    date_mapping_df = date_mapping_df[['Original_table_name', 'Variable_name', 'Value']].dropna() # dropna
    date_mapping_df.rename(columns={'Value': 'Date_value'}, inplace=True)

    combined_date_df = pd.concat([record_dates_df, date_mapping_df])

    return combined_date_df[['Original_table_name', 'Variable_name', 'Date_value']]

def _validate_date_entry(date_string, client, model):
                         
    if _is_valid_date(date_string) :
        return True  
    elif _valid_date_custom(date_string):
        return True 
                         
    else:
        print('gpt')
        user_content = f"date : {date_string}"
        try:
            result = _gpt_chat(client, model, user_content)
            if "no" in result:
                return False
            elif 'yes' in result:
                return True
            else:
                return None 
        except Exception as e:
            print("Failed to call GPT service:", e) 
            return None
                         

def get_date_validity(
        quiq:pd.DataFrame,
        model:str,
        api_key:str) :
   
    client = openai.OpenAI(api_key=api_key)
    
    df = quiq.copy()
    df['Mapping_info_1'] = df['Mapping_info_1'].astype(str)

    final_date_df = _extract_date_data_mapping(df)
    assert len(final_date_df) > 0, 'FAIL : No date values.'
    
    grouped_df_idx = final_date_df.groupby(['Original_table_name', 'Variable_name']).count().index
    
    summary_df = pd.DataFrame(columns = ['Original_table_name', 'Variable_name', 'Total_date', 'Invalid_date', 'Date_Validity_(%)'])
    
    tqdm.pandas()
    valid_results_df = pd.DataFrame()
    for n, idx in enumerate(grouped_df_idx) :

        table_name = idx[0]
        variable_name = idx[1]
        
        print(f'\n{table_name} - {variable_name}')
        temp = final_date_df[(final_date_df['Original_table_name'] == table_name) & (final_date_df['Variable_name'] == variable_name)]
        
        temp['Is_valid'] = temp['Date_value'].progress_apply(lambda x : _validate_date_entry(x, client, model))
        
        total_date = len(temp)
        valid_date = temp['Is_valid'].sum()
        invalid_date = total_date - valid_date
        date_validity = valid_date / total_date * 100
        date_validity = round(date_validity, 2)
                         
        summary_df.at[n, 'Original_table_name'] = table_name
        summary_df.at[n, 'Variable_name'] = variable_name
        summary_df.at[n, 'Total_date'] = total_date  
        summary_df.at[n, 'Invalid_date'] = invalid_date
        summary_df.at[n, 'Date_Validity_(%)'] = date_validity
        
        valid_results_df = pd.concat([valid_results_df, temp], axis = 0)
        
        gc.collect()
        

    return valid_results_df, summary_df
   
if __name__ == '__main__' :
    print('<LYDUS - Date Validity>')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')
    api_key = config.get('api_key')
    
    valid_results_df, summary_df = get_date_validity(quiq, model_ver, api_key)

    total_date = summary_df['Total_date'].sum()
    invalid_date = summary_df['Invalid_date'].sum()
    valid_date = total_date - invalid_date
    date_validity = valid_date / total_date * 100
    date_validity = round(date_validity, 2)
    
    with open(save_path + '/date_validity_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Date Validity (%) = {date_validity}\n')
        file.write(f'Total dates = {total_date}\n')
        file.write(f'Invalid dates = {invalid_date}\n')
    
    valid_results_df.to_csv(save_path + '/date_validity_detail.csv', index = False)
    
    summary_df.to_csv(save_path + '/date_validity_summary.csv', index = False)
    
    print('\n<SUCCESS>')
