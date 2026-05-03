import yaml
import argparse
import pandas as pd
from math import log
from collections import Counter

def _calculation(instance_values):
    filtered_values = [value for value in instance_values if pd.notnull(value) and value != 0]
    if not filtered_values:
        return None, None, None, Counter()  # Avoid division by zero
    
    total_num = sum(filtered_values)
    if total_num == 0:
        return 0, 0, 0, Counter()  # Handling divide by zero
    
    instance_counter = Counter(filtered_values)
    
    probabilities = [count / total_num for _, count in instance_counter.items()]
    shannon_diversity = -sum(prob * log(prob) for prob in probabilities if prob > 0)
    simpson_diversity = sum(prob**2 for prob in probabilities)
    instance_diversity = len(filtered_values) / total_num
    simpson_diversity_score = 1.0 if instance_diversity==1 else (1 - simpson_diversity)

    instance_diversity = round(instance_diversity * 100, 2)
    simpson_diversity_score = round(simpson_diversity_score * 100, 2)
    return instance_diversity, shannon_diversity, simpson_diversity_score, instance_counter

def _calculate_class_instance_counts(class_values, instance_values):
    df = pd.DataFrame({"Class": class_values, "Instance": instance_values})
    class_instance_counts = df.groupby(["Class", "Instance"]).size().reset_index(name="Instance_Count")
    class_counts = df.groupby("Class").size().reset_index(name="Total_Count")
    
    weighted_instance_diversities = []
    weighted_simpson_diversities = []
    total_items = len(class_values)

    results_data = []
    
    for _, row in class_counts.iterrows():
        class_value = row["Class"]
        total_count = row["Total_Count"]
        instance_diversity, shannon_diversity, simpson_diversity_score, instance_counter = _calculation(
            class_instance_counts[class_instance_counts["Class"] == class_value]["Instance_Count"]
        )
        results_data.append({
            'Original_table_name': class_value[0],
            'Variable_name': class_value[1],
            'Value' : class_value[2],
            'Instance Diversity (%)': instance_diversity,
            'Gini-Simpson Index (%)': simpson_diversity_score
        })
        weighted_instance_diversities.append(instance_diversity * total_count)
        weighted_simpson_diversities.append(simpson_diversity_score * total_count)
        
    results_df = pd.DataFrame(results_data)
    
    weighted_avg_instance_diversity = round(sum(weighted_instance_diversities) / total_items, 2)
    weighted_avg_simpson_diversity = round(sum(weighted_simpson_diversities) / total_items, 2)

    return results_df, weighted_avg_instance_diversity, weighted_avg_simpson_diversity
    
def get_instance_diversity(quiq:pd.DataFrame):

    df = quiq.copy()
    df['Variable_name'] = df['Variable_name'].astype(str)
    df['Mapping_info_1'] = df['Mapping_info_1'].astype(str)
    df['Mapping_info_2'] = df['Mapping_info_2'].astype(str)
    df['Is_categorical'] = pd.to_numeric(df['Is_categorical'], errors = 'coerce')
     
    df0 = df[df['Mapping_info_1'].str.contains('event', case = False, na = False)]
    df1 = df[df['Mapping_info_1'].str.contains('diagnosis', case = False, na = False)]
    df2 = df[df['Mapping_info_1'].str.contains('prescription', case = False, na = False) & df['Mapping_info_2'].str.contains('drug', case = False, na = False)]
    df3 = df[df['Mapping_info_1'].str.contains('procedure', case = False, na = False)]
    df = pd.concat([df0, df1, df2, df3], axis = 0)
    
    df = df[df['Is_categorical'] == 1]
    
    df_filtered = df.dropna(subset = ['Patient_id', 'Variable_name', 'Value'])
    
    df_filtered['ClassKey'] = list(zip(df_filtered['Original_table_name'], df_filtered['Variable_name'], df_filtered['Value']))

    class_values = df_filtered['ClassKey']
    instance_values = df_filtered['Patient_id']

    results_df, weighted_avg_instance_diversity, weighted_avg_simpson_diversity = _calculate_class_instance_counts(class_values, instance_values)

    
    return results_df, weighted_avg_instance_diversity, weighted_avg_simpson_diversity        

if __name__ == '__main__' :
    print('<LYDUS - Instance Diversity>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding = 'utf-8') as file :
        config = yaml.safe_load(file)
        
    quiq_path = config.get('quiq_path')
    quiq = pd.read_csv(quiq_path)
    save_path = config.get('save_path')
    
    df_results, weighted_avg_instance_diversity, weighted_avg_simpson_diversity = get_instance_diversity(quiq)

    with open(save_path + '/instance_diversity_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Weighted Instance Diveristy (%) = {weighted_avg_instance_diversity}\n')
        file.write(f'Weighted Gini-Simpson Index (%) = {weighted_avg_simpson_diversity}\n')
    
    df_results.to_csv(save_path + '/instance_diversity_summary.csv', index = False)
    
    print('<SUCCESS>')
