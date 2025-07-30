import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_box_plot(save_path, order, label_vs_boxplot, box_plot_num):

    labels = order
    n = min(len(labels), box_plot_num)
    
    #print(labels)
    fig, axs = plt.subplots(1, n, figsize=(4 * max(1, n), 7))
    fig.set_tight_layout(True)
    flierprops = dict(marker='o', markerfacecolor='green', markersize=2,
                      linestyle='none', alpha=0.2)
    for i in range(n):
        label = order[i]
        ax = axs[i]
        box_data = label_vs_boxplot.get(label, [])

        if len(box_data) == 0:
            ax.set_title(f"{label}\n(No data)", fontsize=14, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.boxplot(box_data, flierprops=flierprops)
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.tick_params(axis='y', labelsize=14)

    plt.savefig(save_path + '/range_validity_boxplot.png', facecolor = 'white')

'''def draw_range_validity_box_plot(ax:Axes, summary_df:pd.DataFrame):
    rangevalidity_list = list(summary_df['outlier_total_proportion'])
    rangevalidity_list = [1-x for x in rangevalidity_list]
    cleaned_list = [x*100 for x in rangevalidity_list if not np.isnan(x)]
    ax.clear()
    ax.boxplot(cleaned_list)'''

def _get_outlier(df=None, column=None, weight=1.5):
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR*weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    outlier_idx_under = df[column][ df[column] < lowest ].index
    outlier_idx_upper = df[column][ df[column] > highest ].index
    return quantile_25, quantile_75, lowest,highest, outlier_idx_under, outlier_idx_upper

def get_range_validity(quiq:pd.DataFrame) :
    
    df = quiq.copy()
    df['Variable_type'] = df['Variable_type'].astype(str)
    df['Is_categorical'] = pd.to_numeric(df['Is_categorical'], errors = 'coerce')
    
    df = df[df['Variable_type'].str.contains('numeric', case = False, na = False)]
    df = df[df['Is_categorical'] == 0]
    df['Value'] = pd.to_numeric(df['Value'], errors = 'coerce')
    df = df.dropna(subset = 'Value')
    df.reset_index(inplace=True,drop=True)

    filtered_groups = df.groupby('Variable_name').filter(lambda x: len(x) > 1000)

    assert len(filtered_groups) > 0, 'FAIL : No variables with more than 1000 data items.'

    
    labellist = list(
        filtered_groups.groupby(['Original_table_name', 'Variable_name'])
        .count()
        .sort_values('Value', ascending=False).index
        )

    itemlist = np.arange(0, len(labellist))
    
    columns_to_extract = ['Patient_id', 'Value', 'Event_date']

    total_outlier_df = pd.DataFrame()
    summary_df = pd.DataFrame(
        columns =[ 
            'Original_table_name', 'Variable_name', 'Total_num', 'Outlier_under_num', 'Outlier_upper_num', 'Outlier_total_num', 
            'Outlier_under_proportion', 'Outlier_upper_proportion', 'Outlier_total_proportion', 'Range Validity (%)'
            ]
        )
    errorlist = []

    dict_dynamic = {}
    
    for i, names in enumerate(labellist):
        tablename, labelname = names
        itemidname = itemlist[i]

        dict_dynamic[itemidname] = df[df['Variable_name']==labelname]
        dict_dynamic[itemidname] = dict_dynamic[itemidname][columns_to_extract]
        dict_dynamic[itemidname].dropna(subset = ['Value'], inplace=True)
        dict_dynamic[itemidname].reset_index(inplace=True,drop=True)
        firstlength = len(dict_dynamic[itemidname])
        try:
            dict_dynamic[itemidname] = dict_dynamic[itemidname].astype({'Value':'float'})
            
            q25, q75, lowest, highest, outlier_idx_under, outlier_idx_upper = _get_outlier(df=dict_dynamic[itemidname], column='Value', weight=1.5)

            outlier_df_under = dict_dynamic[itemidname].iloc[outlier_idx_under]
            outlier_df_under.reset_index(inplace=True,drop=True)
            outlier_df_under['Original_table_name'] = tablename
            outlier_df_under['Variable_name']=labelname
            outlier_df_under['Direction']='under'
            outlier_df_under = outlier_df_under[['Original_table_name', 'Variable_name', 'Patient_id', 'Event_date', 'Value', 'Direction']]

            outlier_df_upper = dict_dynamic[itemidname].iloc[outlier_idx_upper]
            outlier_df_upper.reset_index(inplace=True,drop=True)
            outlier_df_upper['Original_table_name'] = tablename
            outlier_df_upper['Variable_name'] = labelname
            outlier_df_upper['Direction'] = 'upper'
            outlier_df_upper = outlier_df_upper[['Original_table_name', 'Variable_name', 'Patient_id', 'Event_date', 'Value', 'Direction']]

            total_outlier_df = pd.concat([total_outlier_df, outlier_df_under])
            total_outlier_df = pd.concat([total_outlier_df, outlier_df_upper])

            secondlength_under = len(outlier_df_under)
            secondlength_upper = len(outlier_df_upper)
            secondlength_total = secondlength_under + secondlength_upper

            proportion_under = round(secondlength_under/firstlength, 3)
            proportion_upper = round(secondlength_upper/firstlength, 3)
            proportion_total = round(secondlength_total/firstlength, 3)

            range_validity = (firstlength - secondlength_total) / firstlength * 100
            range_validity = round(range_validity, 2)
            
            summary = [
                tablename, labelname, firstlength, secondlength_under, secondlength_upper, secondlength_total, 
                proportion_under, proportion_upper, proportion_total, range_validity
                ]
            summary_df.loc[len(summary_df)] = summary

            print(labelname, "| outlier_under:", secondlength_under, "outlier_upper:", secondlength_upper, "outlier_total:", secondlength_total)

        except:
            errorlist.append(labelname)

    label_vs_boxplot = {}
    for i in range(len(itemlist)):
        itemidname = itemlist[i]
        tablename = labellist[i][0]
        labelname = labellist[i][1]
        label_vs_boxplot[f'{tablename} - {labelname}'] = dict_dynamic[itemidname]['Value'].to_list()


    return total_outlier_df, summary_df, label_vs_boxplot


if __name__ == '__main__' :
    print('<LYDUS - Range Validity>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    top_n = config.get('top_n', 10)
    
    df_total_outlier, df_summary, label_vs_boxplot = get_range_validity(quiq)

    total_num = df_summary['Total_num'].sum()
    outlier_num = df_summary['Outlier_total_num'].sum()
    range_validity = (total_num - outlier_num) / total_num * 100
    range_validity = round(range_validity, 2)

    with open(save_path + '/range_validity_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Range Validity (%) = {range_validity}\n')
        file.write(f'Total Num = {total_num}\n')
        file.write(f'Outlier Num = {outlier_num}\n')
    
    df_summary.to_csv(save_path + '/range_validity_summary.csv', index = False)
    df_total_outlier.to_csv(save_path + '/range_validity_outlier_total.csv', index = False)
    
    df_summary['Identifier'] = df_summary['Original_table_name'] + ' - ' + df_summary['Variable_name']
    target_list = df_summary['Identifier'][:top_n]
    draw_box_plot(save_path, target_list, label_vs_boxplot, len(target_list))

    
    print('\n<SUCCESS>')
