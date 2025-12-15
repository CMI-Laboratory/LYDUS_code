import gc
import yaml
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt


def draw_auroc_plot (save_path, category, category_results) :

    plt.figure(figsize = (8,5))
    plt.plot(category_results['Year'].to_numpy(), category_results['AUROC'].to_numpy())
    plt.axhline(y = 0.8, color = 'gray', linestyle = '--')
    change_point = category_results[category_results['Is_change_point'] == 1]
    plt.scatter(change_point['Year'].to_numpy(), change_point['AUROC'].to_numpy(), color = 'red', marker = 'o')

    plt.title(f'{category} - AUROC plot')
    plt.xlabel('Year')
    plt.ylabel('AUROC')
    plt.ylim(-0.1, 1)

    plt.tight_layout()

    if save_path is not None :
        plt.savefig(save_path + f'/{category} - AUROC plot.png', facecolor = 'white')


def detect_change_point(category, save_path, df_pivot, df_years) : # Year # AUROC # Is_change_point # Shap result
    
    for target_year in df_years['Year'] :
        print()
        print(target_year, end = ' ')
        
        idx = df_years.loc[df_years['Year'] == target_year].index[0]
       
        len_before_target = len(df_pivot.loc[df_pivot['Event_date'].dt.year == target_year-1])
        len_after_target = len(df_pivot.loc[df_pivot['Event_date'].dt.year == target_year])
        
        if (len_before_target + len_after_target) == 0 :
            df_years.at[idx, 'AUROC'] = -0.1
            print('FAIL - No data to process', end = ' ')
            continue
        
        if (min(len_before_target, len_after_target) / (len_before_target + len_after_target)) < 0.25 : 
            df_years.at[idx, 'AUROC'] = -0.1
            print('FAIL - Data imbalance detected', end = ' ')
            continue
            
        temp = df_pivot.loc[(df_pivot['Event_date'].dt.year == target_year-1) | (df_pivot['Event_date'].dt.year == target_year)]

        temp.loc[(temp['Event_date'].dt.year == target_year-1), 'Label'] = 0
        temp.loc[(temp['Event_date'].dt.year == target_year), 'Label'] = 1

        X = temp.drop(columns = ['Patient_id', 'Event_date', 'Year', 'Label'])
        y = temp['Label']

        if (y.value_counts().min() < 2) :
            df_years.at[idx, 'AUROC'] = -0.1
            print('FAIL - Not enough data to process', end = ' ')
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify = y)

        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        target_auroc = roc_auc_score(y_test, y_pred_proba)
        df_years.at[idx, 'AUROC'] = target_auroc


        if target_auroc >= 0.8 : 
            df_years.at[idx, 'Is_change_point'] = 1
            print('CHANGE POINT!', end = ' ')
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            year = df_years.at[idx, 'Year']
            plt.figure()
            shap.summary_plot(shap_values, X_test, plot_type = 'bar', show=False)
            if save_path is not None :
                plt.savefig(save_path+f'/{category} ({year}) - SHAP Plot.png', facecolor = 'white', bbox_inches = 'tight')


        gc.collect()
        
    print()
    return df_years


def get_time_series_consistency (save_path, quiq) :
    
    quiq_df = quiq.copy()
    quiq_df['Event_date'] = pd.to_datetime(quiq_df['Event_date'], errors = 'coerce')
    quiq_df = quiq_df.dropna(subset = 'Event_date')
    quiq_df['Mapping_info_1'] = quiq_df['Mapping_info_1'].astype(str)
    quiq_df['Mapping_info_2'] = quiq_df['Mapping_info_2'].astype(str)
    
    total_results = {}
    summary = pd.DataFrame(columns = ['Category', 'Total_time_point', 'Change_point', 'Time Series Consistency (%)'])
    summary['Category'] = ['Event', 'Diagnosis', 'Prescription', 'Procedure']
     
    
    ############ Event ############
    
    print('CATEGORY: Event')
    df_event_quiq = quiq_df[quiq_df['Mapping_info_1'].str.contains('event', case = False, na = False)] 
    df_event_quiq = df_event_quiq[df_event_quiq['Mapping_info_2'].str.contains('lab_event', case = False, na = False)] 
    
    if len(df_event_quiq) == 0 :
        print('\nFAIL - Unable to calculate: No data to related to event')
    else :
        df_event_quiq['Dummy'] = df_event_quiq['Variable_name'].copy()
        df_event_pivot = df_event_quiq.pivot_table(index = ['Patient_id', 'Event_date'],
                                                   columns = 'Variable_name',
                                                   values = 'Dummy',
                                                   aggfunc = lambda x : 1, 
                                                   fill_value = 0).reset_index()
        
        df_event_pivot['Year'] = df_event_pivot['Event_date'].dt.year

        df_event_results = pd.DataFrame()
        min_year_for_model = df_event_pivot['Year'].min() + 1
        max_year_for_model = df_event_pivot['Year'].max() - 1
        df_event_results['Year'] = np.arange(min_year_for_model, max_year_for_model)
        df_event_results['AUROC'] = np.nan
        df_event_results['Is_change_point'] = 0

        if len(df_event_results) == 0 :
            print('FAIL - Unable to calculate: not enough historical data.')
            print()  
        else :
            df_event_results = detect_change_point(summary['Category'][0], save_path, df_event_pivot, df_event_results)
            total_results['event'] = df_event_results 
            
            total_time_point = len(df_event_results)
            change_point = df_event_results['Is_change_point'].sum()
            summary.at[0, 'Total_time_point'] = total_time_point
            summary.at[0, 'Change_point'] = change_point
            summary.at[0, 'Time Series Consistency (%)'] = ((total_time_point - change_point) / total_time_point * 100).round(2)
            
    
    gc.collect()
        
    
    ############ Diagnosis ############
    
    print()
    print('CATEGORY: Diagnosis')
    df_diagnosis_quiq = quiq_df[quiq_df['Mapping_info_1'].str.contains('diagnosis', case = False, na = False)]
    
    if len(df_diagnosis_quiq) == 0 :
        print('\nFAIL - Unable to calculate: No data to related to diagnosis')
    else : 
        df_diagnosis_quiq['Dummy'] = df_diagnosis_quiq['Value'].copy()
        df_diagnosis_pivot = df_diagnosis_quiq.pivot_table(index = ['Patient_id', 'Event_date'],
                                      columns = 'Value',
                                      values = 'Dummy',
                                      aggfunc = lambda x : 1,
                                      fill_value = 0).reset_index()

        df_diagnosis_pivot['Year'] = df_diagnosis_pivot['Event_date'].dt.year

        df_diagnosis_results = pd.DataFrame()
        min_year_for_model = df_diagnosis_pivot['Year'].min() + 1
        max_year_for_model = df_diagnosis_pivot['Year'].max() - 1
        df_diagnosis_results['Year'] = np.arange(min_year_for_model, max_year_for_model)
        df_diagnosis_results['AUROC'] = np.nan
        df_diagnosis_results['Is_change_point'] = 0
        
        if len(df_diagnosis_results) == 0 :
            print('FAIL - Unable to calculate: Not enough historical data.')
            print()  
        else :
            df_diagnosis_results = detect_change_point(summary['Category'][1], save_path, df_diagnosis_pivot, df_diagnosis_results)
            total_results['diagnosis'] = df_diagnosis_results
            
            total_time_point = len(df_diagnosis_results)
            change_point = df_diagnosis_results['Is_change_point'].sum()
            summary.at[1, 'Total_time_point'] = total_time_point
            summary.at[1, 'Change_point'] = change_point
            summary.at[1, 'Time Series Consistency (%)'] = ((total_time_point - change_point) / total_time_point * 100).round(2)
    

    gc.collect()
    
    
    ############ Prescription ############
    
    print()
    print('CATEGORY: Prescription')
    df_prescription_quiq = quiq_df[quiq_df['Mapping_info_1'].str.contains('prescription', case = False, na = False)] 
    df_prescription_quiq = df_prescription_quiq[df_prescription_quiq['Mapping_info_1'].str.contains('drug', case = False, na = False)]
    
    if len(df_prescription_quiq) == 0 :
        print('\nFAIL - Unable to calculate: No data related to prescription')
    else :
        df_prescription_quiq['Dummy'] = df_prescription_quiq['Value'].copy()
        df_prescription_pivot = df_prescription_quiq.pivot_table(index = ['Patient_id', 'Event_date'],
                                      columns = 'Value',
                                      values = 'Dummy',
                                      aggfunc = lambda x : 1,
                                      fill_value = 0).reset_index()
        
        df_prescription_pivot['Year'] = df_prescription_pivot['Event_date'].dt.year

        df_prescription_results = pd.DataFrame()
        min_year_for_model = df_prescription_pivot['Year'].min() + 1
        max_year_for_model = df_prescription_pivot['Year'].max() - 1
        df_prescription_results['Year'] = np.arange(min_year_for_model, max_year_for_model)
        df_prescription_results['AUROC'] = np.nan
        df_prescription_results['Is_change_point'] = 0
        
        if len(df_prescription_results) == 0 :
            print('FAIL - Unable to calculate: not enough historical data.')
            print()  
        else :
            df_prescription_results = detect_change_point(summary['Category'][2], save_path, df_prescription_pivot, df_prescription_results)
            total_results['prescription'] = df_prescription_results
            
            total_time_point = len(df_prescription_results)
            change_point = df_prescription_results['Is_change_point'].sum()
            summary.at[2, 'Total_time_point'] = total_time_point
            summary.at[2, 'Change_point'] = change_point
            summary.at[2, 'Time Series Consistency (%)'] = ((total_time_point - change_point) / total_time_point * 100).round(2)
     
    gc.collect()
    
    
    
    ############ Procedure ############
    print()
    print('CATEGORY: Procedure')
    df_procedure_quiq = quiq_df[quiq_df['Mapping_info_1'].str.contains('procedure', case = False, na = False)]
    
    if len(df_procedure_quiq) == 0 :
        print('\nFAIL - Unable to calculate: No data related to procedure')
    else :
        df_procedure_quiq['Dummy'] = df_procedure_quiq['Value'].copy()
        df_procedure_pivot = df_procedure_quiq.pivot_table(index = ['Patient_id', 'Event_date'],
                                      columns = 'Value',
                                      values = 'Dummy',
                                      aggfunc = lambda x : 1,
                                      fill_value = 0).reset_index()
        
        df_procedure_pivot['Year'] = df_procedure_pivot['Event_date'].dt.year

        df_procedure_results = pd.DataFrame()
        min_year_for_model = df_procedure_pivot['Year'].min() + 1
        max_year_for_model = df_procedure_pivot['Year'].max() - 1
        df_procedure_results['Year'] = np.arange(min_year_for_model, max_year_for_model)
        df_procedure_results['AUROC'] = np.nan
        df_procedure_results['Is_change_point'] = 0
        
        if len(df_procedure_results) == 0 :
            print('FAIL - Unable to calculate: not enough historical data.')
            print()  
        else :
            df_procedure_results = detect_change_point(summary['Category'][3], save_path, df_procedure_pivot, df_procedure_results)
            total_results['procedure'] = df_procedure_results
            
            total_time_point = len(df_procedure_results)
            change_point = df_procedure_results['Is_change_point'].sum()
            summary.at[3, 'Total_time_point'] = total_time_point
            summary.at[3, 'Change_point'] = change_point
            summary.at[3, 'Time Series Consistency (%)'] = ((total_time_point - change_point) / total_time_point * 100).round(2)
  
    gc.collect()
    
    return total_results, summary


if __name__ == '__main__' :
    print('<LYDUS - Time Series Consistency>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    
    dict_total_results, df_summary = get_time_series_consistency(save_path, quiq)
    
    total_time_point = df_summary['Total_time_point'].sum()
    change_point = df_summary['Change_point'].sum()
    time_series_consistency = ((total_time_point - change_point) / total_time_point * 100)
    time_series_consistency = round(time_series_consistency, 2)
    
    with open(save_path + '/time_series_consistency_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Time Series Consistency (%) = {time_series_consistency}\n')
        file.write(f'Total Time Points = {total_time_point}\n')
        file.write(f'Change Points = {change_point}\n')

    df_summary.to_csv(save_path + '/time_series_consistency_summary.csv', index = False)
    
    for idx in range(len(df_summary)) :
        if not np.isnan(df_summary.at[idx, 'Time Series Consistency (%)']) : 
       
            category = df_summary.at[idx, 'Category'].lower()
            category_results = dict_total_results[category]
        
            draw_auroc_plot(save_path, category, category_results) 
    
    print('\n<SUCCESS>')
