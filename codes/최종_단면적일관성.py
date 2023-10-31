import os
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

def initialize_openai(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
    return llm

def get_category_counts(llm, unique_array):
    result = llm(f"{unique_array}, classify this into minimum groups in forms like 'A / B' according to meanings without any comments. For example, ['F', 'M', 'Female'] into 'F, Female / M'.").strip()
    lines = result.strip().split('\n')
    #print(lines)
    category_counts = {}
    for line in lines:
        categories = line.strip('[]').replace("'", "").split(' / ')
        for cat in categories:
            items = [item.strip() for item in cat.split(',')]
            for item in items:
                if item not in category_counts:
                    category_counts[item] = 1
                else:
                    category_counts[item] += 1
    
    return categories, category_counts


def calculate_inner_consistency(category, df, TARGET_VARIABLE):
    total_weighted_percentage = 0
    total_weight = 0
    for line in lines:
        line_values = eval(line)  # Assuming line is something like "['F', 'M']"
        
        count_by_col = df[TARGET_VARIABLE].apply(lambda x: x if x in line_values else None).value_counts().dropna()
        
        for value, count in count_by_col.items():
            max_percentage = count / count_by_col.sum()
            weighted_percentage = max_percentage * count
            total_weighted_percentage += weighted_percentage
        
        total_weight += count_by_col.sum()
        
    return total_weighted_percentage / total_weight




def calculate_consistency(csv_file_name):
    OPENAI_API_KEY = "sk-"  # 여기에 실제 OpenAI API 키를 입력하십시오.
    llm = initialize_openai(OPENAI_API_KEY)
    TARGET_VARIABLE = '성별'
    df = pd.read_csv(csv_file_name)
    unique_array = df[TARGET_VARIABLE].dropna().unique()
    
    categories, category_counts = get_category_counts(llm, unique_array)
    
    ''''
    for category, count in category_counts.items():
        print(f"{category}: {count}개")
    ''' 
     
    consistency_value = calculate_inner_consistency(lines, df, TARGET_VARIABLE)
    print('일관성:', consistency_value)
