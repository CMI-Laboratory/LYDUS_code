import yaml
import argparse
import pandas as pd
import openai
import re

SYSTEM_CONTENT = f"""
    Goal: To check whether terms that represent the **same real-world meaning** are expressed consistently across the dataset.
    You must classify them based on their true meaning, not just their linguistic similarity.
     - If terms do not have exactly the same meaning, they do not need to be grouped. They SHOULD be left as SEPERATE GROUPS.
     - If all terms have different meanings, assign each term to its own group.
     - Especially for medical info, **DO NOT group similar-looking words unless they refer to the same real-world meaning.**
    
    
    #### ✅ *1. DO NOT remove or replace underscores (`_`) or blank in any terms.  
    - **DO NOT change uppercase/lowercase. Even if a word looks like a proper noun, DO NOT capitalize it.
    
    #### ✅ **2. Never merge different medical info(e.g., blood_type), Ethnicity and Nationality. They Must Always Be Separate** 
    - Don't ever group another countries. 

    **Examples:**
    Example 1:
    Input: ['F', 'M']
    Output:
    category1: 'F'
    category2: 'M'

    Example 2:
    Input: ['F', 'M', '여', 'Female', '남자', 'Male', '여자', '남', 'female']
    Output:
    category1: 'F', 'Female', '여', '여자', 'female'
    category2: '남', '남자', 'M', 'Male'

    Example 3:
    Input: ['ENG', 'english', 'KOR', 'korean', 'SPN']
    Output:
    category1: 'ENG', 'english'
    category2: 'KOR', 'korean'
    category3: 'SPN'

    Example 4 (Nationality or Ethnicity Issue):
    Input: ['French', 'Dominican', 'American', 'Canadian', 'american', 'French_canadian']
    Output:
    category1: 'French'
    category2: 'Dominican'
    category3: 'American', 'american'
    category4: 'Canadian'
    category5: 'French_canadian'

    **"Return the exact same capitalization as given. No other comment except lists. Output format should be a format like below.**
    **Required Output Format:(STRICT)**
    category1: 'A1','A2'
    category2: 'B1','B2'
    category3: 'C1'
    """


def _filter_dataframe(df):
    
    df['Mapping_info_1'] = df['Mapping_info_1'].fillna("").astype(str)
    df['Mapping_info_2'] = df['Mapping_info_2'].fillna("").astype(str)
    df_filtered = df[~df['Value'].isna()]
    
    df_filtered = df_filtered[df_filtered['Variable_type'].str.contains('string|str', case=False, na=False)]
    
    df_filtered = df_filtered[~df_filtered['Mapping_info_1'].str.contains('note|code|date', case=False, na=False)]
    
    df_filtered = df_filtered[df_filtered['Is_categorical'] == 1]
    
    return df_filtered


def _llm_chat(client, model, variable_name, description, user_content, temperature=0, max_tokens=1000, n=1):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": f"""Variable name:{variable_name}
                 Description: {description}
                 Only group the values below
                 Values:{user_content}
                 """}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n
        )
        
        choice = response.choices[0]
        
        if choice.message.content is None:
            reason = choice.finish_reason
            print(f"❌ Invalid LLM Response (Variable: {variable_name}) → ❌ Cause: {reason}")
            return None

        return [choice.message.content for choice in response.choices]
        
    
    except Exception as e:
        print(f"❌ LLM call failed (Variable: {variable_name}) → {type(e).__name__}: {e}")
        return None

def normalize(text: str) -> str:
    return ' '.join(text.strip().split())

def _parse_llm_response_by_line(response, unique_array):

    try:
        
        lines = response.strip().split('\n')
        array_2d = []

        normalized_unique = set(normalize(str(u)) for u in unique_array)

        for line in lines:
            if ':' in line:
                try:
                    matches = re.findall(r"'(.*?)'", line)
                    if not matches:
                        matches = [line.split(':', 1)[1].strip()]

                    filtered = [normalize(m) for m in matches if normalize(m) in normalized_unique]

                    if filtered:
                        array_2d.append(filtered)
                    
                except Exception as inner_e:
                    print(f"⚠️ Failed to parse line: {line} → {inner_e}")
                    continue

        return array_2d

    except Exception as e:
        print(f"❌ Error while parsing LLM response: {type(e).__name__}: {e}")
        if e.__cause__:
            print(f"➡️ Inner cause: {type(e.__cause__).__name__}: {e.__cause__}")
        return []


def _get_category_counts(df, current_identifier, unique_array, description, openai_client, model, attempt=1):
    unique_array = df[df['identifier'] == current_identifier]['Value'].dropna().unique()
    
    user_content = ', '.join(unique_array.tolist())

    result = _llm_chat(openai_client, model, current_identifier, description, user_content)
    
    if result is None:
        print(f"⛔ No LLM response, excluded from evaluation → {current_identifier}")
        return [], {}, False 
    
    response_text = result[0] if result else ""
    
    array_2d = _parse_llm_response_by_line(response_text, unique_array)
    
    category_counts = {}
    for category in array_2d:
        category_counts[category[0]] = len(category)

    flattened_llm = [normalize(item) for sublist in array_2d for item in sublist]
    df_var = df[df['identifier'] == current_identifier]
    flattened_df = df_var['Value'].astype(str).apply(normalize)

    valid = any(flattened_df.isin(flattened_llm))
    
    if not valid and attempt < 5:  
        print("No valid data matched with LLM categories, retrying classification... ",)
        return _get_category_counts(df, current_identifier, unique_array, description, openai_client, model, attempt + 1)
    
    return array_2d, category_counts, valid


def _calculate_inner_consistency(array_2d, df):
    total_weighted_percentage = 0
    total_weight = 0
    
    for line in array_2d:
        count_by_col = df['Value'].astype(str).str.strip().apply(
            lambda x: x if x in [item.strip() for item in line] else None
            ).value_counts().dropna()
        
        total_counts = count_by_col.sum()
        if total_counts > 0:
            for value, count in count_by_col.items():
                max_percentage = count / total_counts
                weighted_percentage = max_percentage * count
                total_weighted_percentage += weighted_percentage
            
            total_weight += total_counts
        else:
            print(f"⛔ Warning: No data matched for line items {line} in df['Value'].")
            
            return None  

    if total_weight == 0:
        print("⛔ Total weight is zero, which indicates no valid data was processed for any categories.")
        return None  

    return round(total_weighted_percentage / total_weight, 3)


def get_cross_sectional_consistency(
        quiq:pd.DataFrame, 
        via: pd.DataFrame,  
        model:str,
        api_key:str) :
    
    client = openai.OpenAI(api_key=api_key)

    df = quiq.copy()
    df['Variable_type'] = df['Variable_type'].astype(str)
    
    target_df = _filter_dataframe(df)
    targets = target_df.groupby(['Original_table_name', 'Variable_name']).count().index
    #target_vars = np.random.choice(target_vars, size=100, replace=False)
    
    total_consistency = 0
    count_vars = 0
    consistency_results = []
    consistency_detail = {}
    consistency_values = []
    
    target_df['identifier'] = target_df['Original_table_name'] + ' - ' + target_df['Variable_name']
    via['identifier'] = via['Original_table_name'] + ' - ' + via['Variable_name']
    
    description_map = dict(
        via[['identifier', "Description"]].drop_duplicates().values)
    
    for target in targets:
        table_name = target[0]
        TARGET_VARIABLE = target[1]
        current_identifier = f'{table_name} - {TARGET_VARIABLE}'
        print(current_identifier)
        
        unique_array = target_df[target_df['identifier'] == current_identifier]['Value'].dropna().unique()
        if len(unique_array) == 1 : 
            continue 
        
        description = description_map.get(current_identifier, "No description available")

        array_2d, category_counts, valid = _get_category_counts(target_df, current_identifier, unique_array, description, client, model)
        if not valid:  
            continue

        consistency_value = _calculate_inner_consistency(array_2d, target_df[target_df['identifier'] == current_identifier])
        if consistency_value is None:  
            continue
        
        consistency_results.append({
            'Original_table_name' : table_name,
            'Variable_name': TARGET_VARIABLE,
            'Cross-sectional consistency (%)': round(consistency_value * 100, 2)
        })
        
        consistency_detail[current_identifier] = []
        for category in array_2d:
            category_name = category[0]
            count = category_counts[category_name]
            items = ', '.join(category)
            line = f"{category_name}: {count} items - Includes: {items}"
            consistency_detail[current_identifier].append(line)

        total_consistency += consistency_value
        count_vars += 1

    if count_vars > 0:
        average_consistency = (total_consistency / count_vars)
        summary = f'Total average consistency: {(average_consistency)}\n'          
    else:
        no_vars_msg = 'No variables to calculate.\n'
        print(no_vars_msg)


    consistency_results_df = pd.DataFrame(consistency_results)

    return average_consistency, consistency_results_df, consistency_detail


if __name__ == '__main__' :
    print('<LYDUS - Cross Sectional Consistency>\n')
    
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
    
    average_consistency, consistency_results_df, consistency_detail = get_cross_sectional_consistency(quiq, via, model_ver, api_key)

    with open(save_path + '/cross_sectional_consistency_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Average Cross Sectional Consistency (%) = {round(average_consistency * 100, 2)}\n')
    
    consistency_results_df.to_csv(save_path + '/cross_sectional_consistency_summary.csv', index = False)
    
    consistency_results_df['identifier'] = consistency_results_df['Original_table_name'] + ' - ' + consistency_results_df['Variable_name']
    with open(save_path + '/cross_sectional_consistency_detail.txt', 'w', encoding = 'utf-8') as file :
        
        for current_identifier in consistency_results_df['identifier'] :

            file.write(f'<{current_identifier}>\n')
            lines = consistency_detail[current_identifier]

            for line in lines :
                file.write(f'{line}\n')

            file.write('\n')
    
    print('\n<SUCCESS>')
