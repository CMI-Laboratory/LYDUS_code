import yaml
import argparse
import os
import re
import math
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def draw_histogram (save_path, idx, identifier, hist_values) :
    table_name, variable_name = identifier.split(' - ')
    plt.title(identifier)
    plt.hist(hist_values)
    table_name = re.sub(r'[\\/:*?"<>|]', ' ', table_name)
    variable_name = re.sub(r'[\\/:*?"<>|]', ' ', variable_name)
    plt.savefig(save_path + f'/preciseness_histograms/{idx}_{table_name}_{variable_name}.png')
    plt.close()
    
def _gini_simpson(data):
    total_count = len(data)
    counter = Counter(data)
    gini_index = 1.0
    diversity = round(len(counter)/total_count, 3)

    for count in counter.values():

        probability = count / total_count
        gini_index -= probability**2

    gini_index = round(gini_index*100, 2)

    return gini_index, diversity

def _round_at_3(x):
    x = round(float(x), 3)
    return x

def _multiply_by(x, a):
    x = x * (10**a)
    return x

def get_preciseness(quiq:pd.DataFrame):    
    
    df = quiq.copy()
    df['Variable_type'] = df['Variable_type'].astype(str)
    df['Is_categorical'] = pd.to_numeric(df['Is_categorical'], errors = 'coerce')
    
    df = df[df['Variable_type'].str.contains('numeric', case = False, na = False)]
    df = df[df['Is_categorical'] == 0]
    df['Value'] = pd.to_numeric(df['Value'], errors = 'coerce')
    df = df.dropna(subset = 'Value')
    df.reset_index(inplace=True,drop=True)

    assert len(df) > 0, 'No numeric values.'

    filtered_groups = df.groupby('Variable_name').filter(lambda x: len(x) > 1000)

    assert len(filtered_groups) > 0, 'FAIL : No variables with more than 1000 data items.'

    labellist = list(
        filtered_groups.groupby(['Original_table_name', 'Variable_name'])
        .count()
        .sort_values('Value', ascending=False)
        .index
        ) 

    #totaltable = pd.DataFrame({lab_table_item_column_name:itemlist, lab_table_label_column_name:labellist})
    totaltable = pd.DataFrame({'Original_table_name' : [x[0] for x in labellist], 'Variable_name':[x[1] for x in labellist]})
    totaltable['Total_num'] = np.nan
    totaltable['Decimal_num'] = np.nan
    totaltable['Preciseness (%)'] = np.nan

    histogram_values = {}

    for i, names in enumerate(labellist):
        tablename, labelname = names
        print(f'{tablename} - {labelname}')
        dftemp = df[df['Original_table_name']==tablename]
        dftemp = dftemp[dftemp['Variable_name']==labelname]
        dftemp.reset_index(inplace=True, drop=True)
        
        columns_to_extract = ['Original_table_name', 'Variable_name', 'Patient_id', 'Event_date', 'Value']
        dftemp = dftemp[columns_to_extract]
        dftemp.dropna(subset = ['Value'], inplace=True)
        dftemp.reset_index(inplace=True, drop=True)

        dftemp['Value'] = dftemp['Value'].apply(_round_at_3)

        decimalnum = -4

        for iterr in range(5):
            dftemp2 = dftemp.copy()
            dftemp2['Value'] = dftemp2['Value'].apply(_multiply_by, args=(3-iterr,))
            tempnum = 0
            for j in range(len(dftemp2)):

                if math.isclose(dftemp2['Value'][j]%10, 0, abs_tol=0.001) == True or math.isclose(-dftemp2['Value'][j]%10, 0, abs_tol=0.001) == True:
                    tempnum += 1
            if tempnum/len(dftemp2) > 0.99:
                pass
            else:
                decimalnum = 3-iterr
                break

        dftemp2 = dftemp.copy()
        dftemp2['Value'] = dftemp2['Value'].apply(_multiply_by, args=(3-iterr,))
        values = []
        for j in range(len(dftemp2)):
            values.append(int(dftemp2['Value'][j]%10))
        a, b = _gini_simpson(values)
        totaltable['Total_num'][i] = len(dftemp2)
        totaltable['Decimal_num'][i] = decimalnum
        totaltable['Preciseness (%)'][i] = a

        histogram_values[f'{tablename} - {labelname}'] = values
    
    totaltable['Total_num'] = totaltable['Total_num'].astype(int)
    totaltable['Decimal_num'] = totaltable['Decimal_num'].astype(int)

    return totaltable, histogram_values

if __name__ == '__main__' :
    print('<LYDUS - Preciseness>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    
    df_summary, histogram_values = get_preciseness(quiq)

    print('\nSave Results...')
    total_num = df_summary['Total_num'].sum()
    preciseness_mul_num_sum = (df_summary['Total_num'] * df_summary['Preciseness (%)']).sum()
    preciseness = round(preciseness_mul_num_sum / total_num, 2)

    with open(save_path + '/preciseness_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Preciseness (%) = {preciseness}\n')
    
    df_summary.to_csv(save_path + '/preciseness_summary.csv', index = False)
    
    os.makedirs(save_path + '/preciseness_histograms', exist_ok = True)
    
    df_summary['Identifier'] = df_summary['Original_table_name'] + ' - ' + df_summary['Variable_name']
    
    for idx, current_identifier in enumerate(df_summary['Identifier']) : 
        draw_histogram(save_path, idx, current_identifier, histogram_values[current_identifier])
    
    print('\n<SUCCESS>')

