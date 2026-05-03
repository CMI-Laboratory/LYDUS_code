import gc
import yaml
import argparse
import pandas as pd
import openai

def match_code_regex (target_name, target_desc) : 

    regex_cache = {'ICD-9' : '^[0-9]{3}(\.[0-9]{1,2})?$',
                   'ICD-10' : '^[A-Z]{1}[0-9]{2}(\.[0-9]{1,2})?$',
                   'ICD-11' : '^[A-Z0-9]{1}[A-Z]{1}[0-9]{1}[A-Z0-9]{1}(\.[A-Z0-9]{1,2})?$',
                   'SNOMED-CT' : '^[0-9]{6,18}$',
                   'RxNorm' :'^[0-9]{5,9}$',
                   'LOINC' : '^[0-9]{1,6}\-[0-9]{1}$',
                   'ATC' : '^[A-Z]{1}[0-9]{2}[A-Z]{2}[0-9]{2}$'}

    target_name = target_name.item().lower()
    target_desc = target_desc.item().lower()
    
    if (('icd' in target_name) and ('9' in target_name)) or (('icd' in target_desc) and ('9' in target_desc)) :
        code_name = 'ICD-9'
    elif (('icd' in target_name) and ('10' in target_name)) or (('icd' in target_desc) and ('10' in target_desc)):
        code_name = 'ICD-10'
    elif (('icd' in target_name) and ('11' in target_name)) or (('icd' in target_desc) and ('11' in target_desc)):
        code_name = 'ICD-11'
    elif (('snomed' in target_name) and ('ct' in target_name)) or (('snomed' in target_desc) and ('ct' in target_desc)):
        code_name = 'SNOMED-CT'
    elif ('rxnorm' in target_name) or ('rxnorm' in target_desc) :
        code_name = 'RxNorm'
    elif ('loinc' in target_name) or ('loinc' in target_desc) :
        code_name = 'LOINC'
    elif ('atc' in target_name) or ('atc' in target_desc) :
        code_name = 'ATC'
    else :
        code_name = None
    
    if code_name is not None : 
        regex_for_target = regex_cache.get(code_name) 
        return code_name, regex_for_target
    else : 
        return None, None



def llm_define_regex (client, model, target_name, target_description) : 

    # GPT response 1 - for code name
    system_prompt = f"""You are a medical coding assistant.
    You will be given a name and a description of a variable.
    Your task is to identify and return **exactly one standardized medical code category** (e.g. ICD-9, SNOMED-CT)
    that best corresponds to the description of a variable.
    Respond with only the name of the code category, no additional explanation.
    If the description does not clearly correspond to any known code category, respond with 'None'"""

    user_prompt = f"""Name of a variable : {target_name}
    Description of a variable : {target_description}"""

    response = client.chat.completions.create(
        model = model,
        messages = [{'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : user_prompt}],
        temperature = 0
    )

    code_name = response.choices[0].message.content

    if code_name == 'None' :
        return None, None
    
    
    
    # GPT response 2 - for Regular Expression
    system_prompt = f"""You are medical coding expert.
    You will be given the name of a standardized medical code category (e.g. ICD-9, SNOMED-CT).
    Your task is to return a regular expression
    that accurately captures the **typical format** of codes within the specified category.
    Respond with only the regular expression, no additional explanation.
    If the format of the given category is unknown or cannot be generalized, respond with 'None'"""

    user_prompt = f"""Medical code category : {code_name}"""

    response = client.chat.completions.create(
        model = model,
        messages = [{'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : user_prompt}],
        temperature = 0
    )

    regex_for_target = response.choices[0].message.content
    
    if regex_for_target == 'None' :
        return code_name, None


    return code_name, regex_for_target


def llm_validate_code (validation_target, regex_for_target) :

    validation_target['Is_valid'] = validation_target['Value'].str.match(regex_for_target)

    return validation_target


def get_code_validity(quiq: pd.DataFrame,
                      via: pd.DataFrame, 
                      model:str,
                      api_key:str) :

    client = openai.OpenAI(api_key=api_key)

    quiq_df = quiq.copy()
    via_df = via.copy()
    quiq_df['Mapping_info_1'] = quiq_df['Mapping_info_1'].astype(str)
    
    filtered_quiq_df = quiq_df[quiq_df['Mapping_info_1'].str.contains('medical_code', case = False, na = False)] # medical code에 대해서만
    filtered_quiq_df = filtered_quiq_df.dropna(subset = ['Value']) # Value가 비어있는 경우 drop
    filtered_quiq_df['Value'] = filtered_quiq_df['Value'].apply(lambda x : str(int(x)) if (isinstance(x, float) and x.is_integer()) else str(x)) # 모든 medical code를 string으로 변환
    filtered_quiq_df = filtered_quiq_df[['Original_table_name', 'Variable_name', 'Value']] # 필요한 column만 남김
    
    assert len(filtered_quiq_df) > 0, 'FAIL : No value related to medical code'
    
    gc.collect()
    
    validation_df = pd.DataFrame() 
    
    filtered_quiq_df['Identifier'] = filtered_quiq_df['Original_table_name'] + ' - ' + filtered_quiq_df['Variable_name'] 
    via_df['Identifier'] = via_df['Original_table_name'] + ' - ' + via_df['Variable_name']
    
    unique_identifiers = filtered_quiq_df['Identifier'].unique()
    
    for current_identifier in unique_identifiers : 

        validation_target = filtered_quiq_df.loc[filtered_quiq_df['Identifier'] == current_identifier]
        validation_target['Regex'] = ''

        target_name = via_df.loc[via_df['Identifier'] == current_identifier]['Variable_name']
        target_description = via_df.loc[via_df['Identifier'] == current_identifier]['Description']

        if len(target_name) == 0 :
            print(f'FAIL - \'{current_identifier}\' could not be found in the VIA table.')
            print(f'       Please check the QUIQ and VIA tables.')
            print()

            validation_target['Is_valid'] = False 
                      
            validation_df = pd.concat([validation_df, validation_target], axis = 0)
            continue


        code_name, regex_for_target = match_code_regex(target_name, target_description) 
                   
        if regex_for_target is None : 
            code_name, regex_for_target = llm_define_regex(client, model, target_name, target_description) 

        if regex_for_target is None :
            if code_name is None :
                print(f'FAIL - Unable to identify an appropriate code for \'{current_identifier}\'')
                print(f'       Please provide a more detailed VIA description.')
                print()
            elif regex_for_target is None : 
                print(f'FAIL - A medical code category ({code_name}) was detected.')
                print(f'       But an appropriate regular rexpression could not be defined.')
                print()
            
            validation_target['Is_valid'] = False 
                      
            validation_df = pd.concat([validation_df, validation_target], axis = 0)
            continue
        ############################
        
        print(f'SUCCESS - Identified \'{code_name}\' with respect to \'{current_identifier}\'')
        print(f'The following regular expression is used : {regex_for_target}')
        print()
            
        validation_target['Regex'] = regex_for_target
        validation_target = llm_validate_code(validation_target, regex_for_target)
        
        validation_df = pd.concat([validation_df, validation_target], axis = 0)
    
        gc.collect()
    
    error_summary = validation_df.groupby(['Original_table_name', 'Variable_name']).agg(
        Total_code = ('Value', 'count'),
        Invalid_code = ('Is_valid', 'sum'),
        Format_Validity = ('Value', lambda x : None),
        Regular_Expression = ('Regex', 'first')
    ).reset_index()
    error_summary = error_summary.rename(columns = {'Format_Validity' : 'Format Validity (%)'})
    error_summary['Format Validity (%)'] = (error_summary['Invalid_code'] / error_summary['Total_code'] * 100).round(2)
    error_summary['Invalid_code'] = error_summary['Total_code'] - error_summary['Invalid_code']
               
    variable_vs_cases = {}
    for current_identifier in unique_identifiers :
        temp = validation_df.loc[validation_df['Identifier'] == current_identifier]
        variable_vs_cases[current_identifier] = temp[['Original_table_name', 'Variable_name', 'Value', 'Is_valid']]
    
    validation_df = validation_df[['Original_table_name', 'Variable_name', 'Value', 'Is_valid']]
    
    gc.collect()

    return validation_df, error_summary, variable_vs_cases
    

if __name__  == '__main__' :
    print('<LYDUS - Format Validity>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    via_path = config.get('via_path')
    via = pd.read_csv(via_path)
    save_path = config.get('save_path')
    model_ver = config.get('model_ver')
    api_key = config.get('api_key')
    
    validation_df, error_summary, variable_vs_cases = get_code_validity(quiq, via, model_ver, api_key)

    error_summary.to_csv(save_path + '/format_validity_summary.csv', index = False)
    
    validation_df.to_csv(save_path + '/format_validity_detail.csv', index = False)
    
    total_code = error_summary['Total_code'].sum()
    invalid_code = error_summary['Invalid_code'].sum()
    format_validity = (total_code - invalid_code) / total_code * 100
    format_validity = round(format_validity, 2)
    
    with open(save_path + '/format_validity_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Format Validity (%) = {format_validity}\n')
        file.write(f'Total Code = {total_code}\n')
        file.write(f'Invalid Code = {invalid_code}\n')
        
    print('<SUCCESS>')
