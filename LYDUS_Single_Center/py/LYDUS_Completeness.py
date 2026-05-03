import yaml
import argparse
import numpy as np
import pandas as pd

def get_completeness(quiq:pd.DataFrame) -> pd.DataFrame:

    df = quiq.copy()

    non_null_counts_per_variable = df.groupby(['Original_table_name', 'Variable_name'])['Value'].count()
    total_counts_per_variable = df.groupby(['Original_table_name', 'Variable_name'])['Value'].size()
    completeness_ratio_per_variable = np.round(non_null_counts_per_variable.values / total_counts_per_variable.values * 100, 2)

    completeness_df = pd.DataFrame({
        'Original_table_name' : [multiindex[0] for multiindex in total_counts_per_variable.index],
        'Variable_name': [multiindex[1] for multiindex in total_counts_per_variable.index],
        'Total_num' : total_counts_per_variable.values,
        'Null_num' :total_counts_per_variable.values - non_null_counts_per_variable.values,
        'Completeness (%)': completeness_ratio_per_variable
    })

    return completeness_df

if __name__ == '__main__' :
    print('<LYDUS - Completeness>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    
    df_summary = get_completeness(quiq)

    total_num = df_summary['Total_num'].sum()
    null_num = df_summary['Null_num'].sum()
    completeness = (total_num - null_num) / total_num * 100
    completeness = round(completeness, 2)

    with open(save_path + '/completeness_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Completeness (%) = {completeness}\n')
        file.write(f'Total Num = {total_num}\n')
        file.write(f'Null Num = {null_num}\n')
    
    df_summary.to_csv(save_path + '/completeness_summary.csv', index = False)

    print('\n<SUCCESS>')
