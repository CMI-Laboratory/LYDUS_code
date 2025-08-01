import yaml
import argparse
import pandas as pd
from math import log
from collections import Counter


def _filter_categorical(df:pd.DataFrame) -> pd.DataFrame:
     
    df1 = df[df['Mapping_info_1'].str.contains('diagnosis', case = False, na = False)]
    df2 = df[df['Mapping_info_1'].str.contains('prescription', case = False, na = False) & df['Mapping_info_2'].str.contains('drug', case = False, na = False)]
    df3 = df[df['Mapping_info_1'].str.contains('procedure', case = False, na = False)]
    df = pd.concat([df1, df2, df3], axis = 0)
    
    df = df[df['Is_categorical'] == 1]
    
    df = df.dropna(subset = 'Value')
    
    df['Value'] = df['Value'].astype(str)
    
    return df


def _calculate_diversity(class_values:pd.Series):
    filtered_values = class_values.dropna()
    class_counter = Counter(filtered_values).most_common()
    class_num = len(class_counter)
    total_num = len(filtered_values)

    probabilities = [count / total_num for _, count in class_counter]
    shannon_diversity = -sum(prob * log(prob) for prob in probabilities)
    simpson_diversity = sum(prob**2 for prob in probabilities)
    class_diversity = class_num / total_num
    
    simpson_diversity_score = 1 if class_diversity == 1 else (1 - simpson_diversity)
    
    return class_diversity, shannon_diversity,simpson_diversity_score, class_counter, class_num, total_num
    

def _calculate_and_plot_diversity(df:pd.DataFrame) :

    diversity_results = []
    
    df_grouped_idx = df.groupby(['Original_table_name', 'Variable_name']).count().sort_values(by = 'Value', ascending = False).index

    for idx in df_grouped_idx :
        table_name = idx[0]
        variable_name = idx[1]

        df['identifier'] = df['Original_table_name'] + ' - ' + df['Variable_name']
        current_identifier = f'{table_name} - {variable_name}'
        
        filtered_df = df[df['identifier'] == current_identifier]

        class_values = filtered_df['Value'] 
        class_diversity, shannon_diversity, simpson_diversity_score, class_counts, class_num, total_num = _calculate_diversity(class_values)
        
        diversity_results.append({
            'Original_table_name': filtered_df['Original_table_name'].unique()[0],
            'Variable_name': filtered_df['Variable_name'].unique()[0],
            'Total Number of Data': total_num,
            'Number of Classes': class_num,
            'Class_diversity (%)': round(simpson_diversity_score * 100, 2)
        })
        
    return diversity_results

def get_class_diversity(quiq:pd.DataFrame) -> pd.DataFrame:
    
    quiq['Mapping_info_1'] = quiq['Mapping_info_1'].astype(str)
    quiq['Mapping_info_2'] = quiq['Mapping_info_2'].astype(str)
    quiq['Is_categorical'] = pd.to_numeric(quiq['Is_categorical'], errors = 'coerce')

    df_categorical = _filter_categorical(quiq)

    assert len(df_categorical) > 0, 'FAIL : No data matching the criteria found.'

    diversity_by_variable_name = _calculate_and_plot_diversity(df_categorical)

    results_df = pd.DataFrame(diversity_by_variable_name)

    return results_df


if __name__ == '__main__' :
    print('<LYDUS - Classs Diversity>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    
    df_summary = get_class_diversity(quiq)

    total_num = df_summary['Total Number of Data'].sum()
    class_diversity_mul_num = (df_summary['Total Number of Data'] * df_summary['Class_diversity (%)']).sum()

    weighted_class_diversity = class_diversity_mul_num / total_num
    weighted_class_diversity = round(weighted_class_diversity, 2)

    with open(save_path + '/class_diversity_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Weighted Class Diversity (%) = {weighted_class_diversity}\n')
    
    df_summary.to_csv(save_path + '/class_diversity_summary.csv', index = False)
    
    print('<SUCCESS>')
