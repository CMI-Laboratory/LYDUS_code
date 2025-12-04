import gc
import re
import yaml
import argparse
import pandas as pd
import numpy as np
import openai
import statsmodels.formula.api as smf
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset, DataLoader

def make_var_list_text(var_list) :
    var_list_text = ''
    
    for var_name in var_list :
        var_list_text += f"'{var_name}', "
    
    var_list_text = var_list_text[:-2]
    
    return var_list_text

def llm_ask_sex(client, model_ver, var_list_sex) :
    system_prompt = '''You are a medical data expert.
    A list of variable names will be provided.
    From the provided variables, select exactly one variable that is most relevant to **biological sex**.
    Respond with **only** the variable name, no additional explanation.
    And return it **exactly as it appears** in the provided list.
    If no appropriate variable is found, respond with 'None'.'''
    
    user_prompt = f'''List of variable names : {var_list_sex}'''

    response = client.chat.completions.create(
        model = model_ver,
        messages = [{'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : user_prompt}],
        temperature = 0
    )
    
    var_name_sex = response.choices[0].message.content
    if var_name_sex == 'None' :
        return None
    
    var_name_sex = var_name_sex.replace("'", '')
    return var_name_sex

def llm_ask_birthdate(client, model_ver, var_list_birthdate) :
    system_prompt = '''You are a medical data expert.
    A list of variable names will be provided.
    From the provided variables, select exactly one variable that is most relevant to **date of birth**.
    Respond with **only** the variable name, no additional explanation.
    And return it **exactly as it appears** in the provided list.
    If no appropriate variable is found, respond with 'None'.'''

    user_prompt = f'''List of variable names : {var_list_birthdate}'''
    
    response = client.chat.completions.create(
        model = model_ver,
        messages = [{'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : user_prompt}],
        temperature = 0
    )
    
    var_name_birthdate = response.choices[0].message.content
    if var_name_birthdate == 'None' :
        return None
    
    var_name_birthdate = var_name_birthdate.replace("'", '')
    return var_name_birthdate

def llm_ask_recommend(client, model_ver, var_name_target, n, var_list_candidate) :
    system_prompt = f'''You are a medical data expert.
    
    Your will be provided with :
    - A target variable
    - A list of variable names
    
    Your task it to :
    select the **top {n}** variables from the list that are **most relevant** to the target variable.
    
    Important Rules :
    1. Do **not include** the target variable itself in the output.
    2. Return exactly {n} variable names, seperate them by **!**.
    3. Do **not repeat** any variable name - all must be unique.
    4. Return the variable names **exactly as it appears** in the provided list.
    5. Do **not include** any additional explanation.
    '''
    
    user_prompt = f'''Target variable : {var_name_target}
    List of variable names : {var_list_candidate}'''
    
    response = client.chat.completions.create(
        model = model_ver,
        messages = [{'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : user_prompt}],
        temperature = 0
    )
    
    var_list_recommended_text = response.choices[0].message.content
    var_list_recommended_text = var_list_recommended_text.replace("'", '')
    var_list_recommended = var_list_recommended_text.split('!')
    print(f'RECOMMENDED VARIABLES : {var_list_recommended}')
    return var_list_recommended
    
    
class Autoencoder(nn.Module) :
    def __init__(self, input_dim) :
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim // 1.3)),
            nn.Tanh(),
            nn.Linear(int(input_dim // 1.3), int(input_dim // 2))
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(input_dim // 2), int(input_dim // 1.3)),
            nn.Tanh(),
            nn.Linear(int(input_dim // 1.3), input_dim)
        )
    
    def forward(self, x) :
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_logical_accuracy(quiq:pd.DataFrame,
                         model_ver:str,
                         api_key:str,
                         operation_type_manual:bool, 
                         target_variable:str, 
                         automatic_num:int, 
                         recommend_num:int) :

    client = openai.OpenAI(api_key=api_key)

    gc.collect()

    df_quiq = quiq.copy()
    df_quiq['Event_date'] = pd.to_datetime(df_quiq['Event_date'], errors = 'coerce')
    df_quiq['Mapping_info_1'] = df_quiq['Mapping_info_1'].astype(str)
    df_quiq['Mapping_info_2'] = df_quiq['Mapping_info_2'].astype(str)
    df_quiq['Variable_type'] = df_quiq['Variable_type'].astype(str)
    df_quiq['Is_categorical'] = pd.to_numeric(df_quiq['Is_categorical'], errors = 'coerce')

    
    df_sex_quiq = df_quiq[df_quiq['Is_categorical'] == 1]

    var_list_sex = df_sex_quiq['Variable_name'].unique()
    if len(var_list_sex) > 0 :
        var_list_sex_text = make_var_list_text(var_list_sex)

        var_name_sex = llm_ask_sex(client, model_ver, var_list_sex_text) # gpt 호출
        print(f'FIND SEX VARIABLE : {var_name_sex}\n')
    else :
        var_name_sex = None
        print('FIND SEX VARIABLE : None\n')

    df_sex_essential = df_sex_quiq[df_sex_quiq['Variable_name'] == var_name_sex] # None이면 길이 0일것임.
    df_sex_essential = df_sex_essential[['Value', 'Patient_id']]
    df_sex_essential = df_sex_essential.dropna()


    df_birthdate_quiq = df_quiq[df_quiq['Mapping_info_1'] == 'date']

    var_list_birthdate = df_birthdate_quiq['Variable_name'].unique()
    if len(var_list_birthdate) > 0 :
        var_list_birthdate_text = make_var_list_text(var_list_birthdate)

        var_name_birthdate = llm_ask_birthdate(client, model_ver, var_list_birthdate_text) # gpt 호출
        print(f'FIND BIRTHDATE VARIABLE : {var_name_birthdate}\n')
    else :
        var_name_birthdate = None
        print('FIND BIRTHDATE VARIABLE : None\n')

    df_birthdate_essential = df_birthdate_quiq[df_birthdate_quiq['Variable_name'] == var_name_birthdate] # None이면 길이 0일것임.
    df_birthdate_essential = df_birthdate_essential[['Value', 'Patient_id']]
    df_birthdate_essential['Value'] = pd.to_datetime(df_birthdate_essential['Value'], errors = 'coerce')
    df_birthdate_essential = df_birthdate_essential.dropna()

    gc.collect()
                             

    print('FILTER CATEGORY VALUES\n')
    # Event filtering (event 변수명)
    df_event_quiq = df_quiq[df_quiq['Mapping_info_1'].str.contains('event', case = False, na = False)] # event 이면서
    df_event_quiq = df_event_quiq[df_event_quiq['Variable_type'].str.contains('numeric', case = False, na = False)] # numeric 인 값만 남김
    df_event_quiq = df_event_quiq[df_event_quiq['Is_categorical'] == 0]    
    df_event_quiq['Value'] = pd.to_numeric(df_event_quiq['Value'], errors = 'coerce') # 실제로 숫자로 바뀌는 값만 남기고
    df_event_quiq = df_event_quiq.dropna(subset = ['Value', 'Event_date']) # na drop 

    df_diagnosis_quiq = df_quiq[df_quiq['Mapping_info_1'].str.contains('diagnosis', case = False, na = False)]
    df_diagnosis_quiq = df_diagnosis_quiq[df_diagnosis_quiq['Is_categorical'] == 1]
    df_diagnosis_quiq = df_diagnosis_quiq.dropna(subset = ['Value', 'Event_date']) # na drop 
    
    df_prescription_quiq = df_quiq[df_quiq['Mapping_info_1'].str.contains('prescription', case = False, na = False)]
    df_prescription_quiq = df_prescription_quiq[df_prescription_quiq['Mapping_info_2'].str.contains('drug', case = False, na = False)]
    df_prescription_quiq = df_prescription_quiq[df_prescription_quiq['Is_categorical'] == 1]
    df_prescription_quiq = df_prescription_quiq.dropna(subset = ['Value', 'Event_date']) # na drop 
    
    df_procedure_quiq = df_quiq[df_quiq['Mapping_info_1'].str.contains('procedure', case = False, na = False)] 
    df_procedure_quiq = df_procedure_quiq[df_procedure_quiq['Is_categorical'] == 1]
    df_procedure_quiq = df_procedure_quiq.dropna(subset = ['Value', 'Event_date']) # na drop 

    df_others_quiq = pd.concat([df_diagnosis_quiq, df_prescription_quiq, df_procedure_quiq], axis = 0)

    assert len(df_event_quiq) + len(df_others_quiq) > 0, \
    'FAIL - No available data related to event, diagnosis, prescription, procedure.' # assertion error
    
    del df_diagnosis_quiq
    del df_prescription_quiq
    del df_procedure_quiq
    
    gc.collect()
    
    
    if operation_type_manual == True : 
        evaluate_mode = [-1]
        if target_variable in df_event_quiq['Variable_name'].unique() : 
            var_list_target = [target_variable]
            evaluate_mode[0] = 0
        elif target_variable in df_others_quiq['Value'].unique() :
            var_list_target = [target_variable]
            evaluate_mode[0] = 1
        else :
            assert False, 'FAIL - Invalid target variable name. Please check and try again.' 
    else : # automatic mode면
        evaluate_mode = [-1] * automatic_num
        var_list_target = [''] * automatic_num
        
        df_others_quiq['Dummy'] = df_others_quiq['Value'].copy()
        count_event = df_event_quiq.groupby('Variable_name').agg(Count = ('Value', 'count'), Category = ('Mapping_info_1', 'first')).reset_index()
        count_event = count_event[['Variable_name', 'Count', 'Category']]
        count_others = df_others_quiq.groupby('Value').agg(Count = ('Dummy', 'count'), Category = ('Mapping_info_1', 'first')).reset_index()
        count_others = count_others[['Value', 'Count', 'Category']]
        count_others = count_others.rename(columns = {'Value' : 'Variable_name'})
        
        count_all = pd.concat([count_event, count_others], axis = 0).sort_values(by = 'Count', ascending = False).reset_index(drop = True)
        

        del count_event
        del count_others
        
        gc.collect()
        
        for idx in range(automatic_num) :
            var_list_target[idx] = count_all.at[idx, 'Variable_name']
            if 'event' in count_all.at[idx, 'Category'].lower() :
                evaluate_mode[idx] = 0
            else :
                evaluate_mode[idx] = 1
    
    print(f'SET TARGET VARIABLES : {var_list_target}')
    #print(f'SET EVALUATE MODE : {evaluate_mode}\n')


    dict_total = {}
    dict_outlier = {}

    for loop, var_name_target in enumerate(var_list_target) :
        flag = 0
        print(f'\n# LOOP {loop+1} - Target Variable : {var_name_target}\n')
          
        var_evaluate_mode = evaluate_mode[loop]
    
        if var_evaluate_mode == 0 : # event면
            df_target_essential = df_event_quiq[df_event_quiq['Variable_name'].isin([var_name_target])]
            df_target_essential = df_target_essential[['Original_table_name', 'Variable_name', 'Event_date', 'Value', 'Patient_id']]
              
            df_event_quiq = df_event_quiq[~df_event_quiq['Variable_name'].isin([var_name_target])] 
        else : 
            df_target_essential = df_others_quiq[df_others_quiq['Value'].isin([var_name_target])]
            df_target_essential = df_target_essential[['Original_table_name', 'Variable_name', 'Event_date', 'Value', 'Patient_id']]
          
            df_others_quiq = df_others_quiq[~df_others_quiq['Value'].isin([var_name_target])] 
            

        df_event_essential = df_event_quiq[['Original_table_name', 'Variable_name', 'Event_date', 'Value', 'Patient_id']]
        var_list_event = df_event_essential.groupby('Variable_name').count().sort_values(by = 'Value', ascending = False).index[:100]
        df_event_essential = df_event_essential[df_event_essential['Variable_name'].isin(var_list_event)]
        
        df_others_quiq['Dummy'] = df_others_quiq['Value'].copy()
        df_others_essential = df_others_quiq[['Original_table_name', 'Variable_name', 'Event_date', 'Value', 'Patient_id', 'Dummy', 'Mapping_info_1']]
        var_list_diagnosis = df_others_essential[df_others_essential['Mapping_info_1'].str.contains('diagnosis', case = False, na = False)]\
          .groupby('Value').count().sort_values(by = 'Dummy', ascending = False).index[:100]
        var_list_prescription = df_others_essential[df_others_essential['Mapping_info_1'].str.contains('prescription', case = False, na = False)]\
          .groupby('Value').count().sort_values(by = 'Dummy', ascending = False).index[:100]
        var_list_procedure = df_others_essential[df_others_essential['Mapping_info_1'].str.contains('procedure', case = False, na = False)]\
          .groupby('Value').count().sort_values(by = 'Dummy', ascending = False).index[:100]
        df_others_essential = df_others_essential[df_others_essential['Value'].isin(var_list_diagnosis.tolist() + var_list_prescription.tolist() + var_list_procedure.tolist())]
    
        gc.collect()
        

        var_list_candidate = list(df_event_essential['Variable_name'].unique()) + list(df_others_essential['Value'].unique())
        var_list_candidate_text = make_var_list_text(var_list_candidate)# text로 만들어
        var_list_recommended = llm_ask_recommend(client, model_ver, var_name_target, recommend_num, var_list_candidate_text)


        print('MAKE CLINICAL CONTEXT VECTOR')
        dict_dynamic = {} 

        if var_evaluate_mode == 0 : 
            dict_dynamic[var_name_target] = df_target_essential[['Event_date', 'Patient_id', 'Value']] 
            dict_dynamic[var_name_target] = dict_dynamic[var_name_target].groupby(['Patient_id', 'Event_date']).agg('median').reset_index()
            dict_dynamic[var_name_target] = dict_dynamic[var_name_target].rename(columns = {'Event_date' : 'Target_date', 'Value' : f'{var_name_target}_val'})
        else :
            dict_dynamic[var_name_target] = df_target_essential[['Event_date', 'Patient_id', 'Value']] 
            dict_dynamic[var_name_target]['Value'] = 1 
            dict_dynamic[var_name_target] = dict_dynamic[var_name_target].rename(columns = {'Event_date' : 'Target_date', 'Value' : f'{var_name_target}_val'})


        for var_name_recommended in var_list_recommended :
            if var_name_recommended in df_event_essential['Variable_name'].unique() : 
                dict_dynamic[var_name_recommended] = df_event_essential[df_event_essential['Variable_name'] == var_name_recommended]
                dict_dynamic[var_name_recommended] = dict_dynamic[var_name_recommended][['Event_date', 'Patient_id', 'Value']]
                dict_dynamic[var_name_recommended] = dict_dynamic[var_name_recommended].groupby(['Patient_id', 'Event_date']).agg('median').reset_index()
                dict_dynamic[var_name_recommended] = dict_dynamic[var_name_recommended].rename(columns = {'Value' : f'{var_name_recommended}_val'})
            else : 
                dict_dynamic[var_name_recommended] = df_others_essential[df_others_essential['Value'] == var_name_recommended]
                dict_dynamic[var_name_recommended] = dict_dynamic[var_name_recommended][['Event_date', 'Patient_id', 'Value']]
                dict_dynamic[var_name_recommended] = dict_dynamic[var_name_recommended].drop_duplicates()
                dict_dynamic[var_name_recommended] = dict_dynamic[var_name_recommended].rename(columns = {'Value' : f'{var_name_recommended}_val'})

            gc.collect()

        df_merged = dict_dynamic[var_name_target]


        for var_name_recommended in var_list_recommended :
            if var_name_recommended in df_event_essential['Variable_name'].unique() :
                df_merged = pd.merge(df_merged, dict_dynamic[var_name_recommended], on = 'Patient_id', how = 'left') 
                df_merged = df_merged.dropna() 

                df_merged = df_merged[(df_merged['Event_date'] >= df_merged['Target_date'] - pd.Timedelta(days = 7)) & (df_merged['Event_date'] <= df_merged['Target_date'])]
                df_merged = df_merged.reset_index(drop = True)
                df_merged['Time_diff'] = (df_merged['Target_date'] - df_merged['Event_date']).dt.total_seconds()
                idx = df_merged.groupby(['Patient_id', 'Target_date', f'{var_name_target}_val'])['Time_diff'].idxmin() 
                df_merged = df_merged.iloc[idx].reset_index(drop = True)

                df_merged = df_merged.drop(['Event_date', 'Time_diff'], axis = 1)

            elif var_name_recommended in df_others_essential['Value'].unique() : 
                dict_dynamic[var_name_recommended][f'{var_name_recommended}_val'] = 1 
                df_merged = pd.merge(df_merged, dict_dynamic[var_name_recommended], on = 'Patient_id', how = 'left')
                df_merged[f'{var_name_recommended}_val'] = df_merged[f'{var_name_recommended}_val'].fillna(0)

                df_merged_split_wdate = df_merged[df_merged[f'{var_name_recommended}_val'] == 1]
                df_merged_split_ndate = df_merged[df_merged[f'{var_name_recommended}_val'] == 0]

                df_merged_split_wdate = df_merged_split_wdate[(df_merged_split_wdate['Event_date'] >= df_merged_split_wdate['Target_date'] - pd.Timedelta(days = 7)) & (df_merged_split_wdate['Event_date'] <= df_merged_split_wdate['Target_date'])]
                df_merged_split_wdate = df_merged_split_wdate.reset_index(drop = True)
                df_merged_split_wdate['Time_diff'] = (df_merged_split_wdate['Target_date'] - df_merged_split_wdate['Event_date']).dt.total_seconds()
                idx = df_merged_split_wdate.groupby(['Patient_id', 'Target_date', f'{var_name_target}_val'])['Time_diff'].idxmin()
                df_merged_split_wdate = df_merged_split_wdate.iloc[idx].reset_index(drop = True)

                df_merged = pd.concat([df_merged_split_wdate, df_merged_split_ndate], axis = 0).reset_index(drop = True)
                df_merged = df_merged.drop(['Event_date', 'Time_diff'], axis = 1)

            else :
                print('FAIL - Variable name mismatch detected.')
                flag = 1
                break

        if flag == 1 :
            continue

            gc.collect()

        if len(df_sex_essential) > 0 :
            onehot_sex = pd.get_dummies(df_sex_essential['Value'], prefix = 'Sex').iloc[:, 1:].astype(int)
            df_sex_essential_concat = pd.concat([df_sex_essential, onehot_sex], axis = 1)
            df_merged = pd.merge(df_merged, df_sex_essential_concat, on = 'Patient_id', how = 'left')
            df_merged = df_merged.dropna() 
            df_merged = df_merged.drop(['Value'], axis = 1)

        if len(df_birthdate_essential) > 0 :
            df_merged = pd.merge(df_merged, df_birthdate_essential, on = 'Patient_id', how = 'left')
            df_merged = df_merged.dropna() 
            df_merged = df_merged[df_merged['Target_date'] > df_merged['Value']]
            df_merged['Age'] = (df_merged['Target_date'].dt.to_pydatetime() - df_merged['Value'].dt.to_pydatetime())
            df_merged['Age'] = [timedelta.days / 365.25 for timedelta in df_merged['Age']]
            df_merged = df_merged.drop(['Value'], axis = 1)

        if len(df_merged) == 0 :
            print('FAIL - Failed to construct the clinical context vector.')
            continue
          
        df_result = df_merged.copy() 

        if var_evaluate_mode == 0 :


            ################## 1. Linear Regression ###################
            print('LINEAR REGRESSION')
            X = df_merged.drop(['Patient_id', 'Target_date', f'{var_name_target}_val'], axis = 1)
            y = df_merged[f'{var_name_target}_val']

            X.columns = X.columns.str.replace(r'\W', '_', regex = True) 
            y.name = re.sub(r'\W', '_', y.name)

            formula = y.name
            formula += ' ~ '
            formula += ' + '.join(X.columns)

            Xy = pd.concat([X, y], axis = 1)

            for quantile in [0.01, 0.99] :
                print(quantile)
                model = smf.quantreg(formula, Xy)
                model = model.fit(q = quantile)
                pred = model.predict(X)
                df_result[f'LR_{quantile}'] = pred

            gc.collect()


            ################## 2. Gradient Boosting ###################
            print('GRADIENT BOOSTING')
            X = df_merged.drop(['Patient_id', 'Target_date', f'{var_name_target}_val'], axis = 1)
            y = df_merged[f'{var_name_target}_val']

            for quantile in [0.01, 0.99] :
                print(quantile)
                model = GBR(loss = 'quantile', alpha = quantile, min_samples_leaf = 5, min_samples_split = 5)
                model = model.fit(X, y)
                pred = model.predict(X)
                df_result[f'GB_{quantile}'] = pred

            gc.collect()


            ################## 3. Auto Encoder ###################
            print('AUTOENCODER')

            Input_unscaled = df_merged.drop(['Patient_id', 'Target_date'], axis = 1) 
            scaler = RobustScaler()
            Input_scaled = scaler.fit_transform(Input_unscaled)
            Input_scaled = torch.tensor(Input_scaled, dtype = torch.float32)

            Input_dataset = TensorDataset(Input_scaled)
            Input_loader = DataLoader(Input_dataset, batch_size = 64, shuffle = True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = Autoencoder(input_dim = Input_scaled.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
            criterion = nn.MSELoss()

            best_loss = float('inf')
            counter = 0
            patience = 5
            min_delta = 1e-3

            epoch = 1
            while True :
                model.train()

                total_loss = 0
                total_sample = 0

                for batch in Input_loader :
                    inputs = batch[0].to(device)
                    outputs = model(inputs)

                    loss = criterion(outputs, inputs) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * inputs.shape[0] 
                    total_sample += inputs.shape[0]

                now_loss = total_loss / total_sample 

                if best_loss - now_loss > min_delta : 
                    best_loss = now_loss
                    counter = 0
                else :
                    counter += 1

                print(f'Epoch {epoch} | Loss : {round(now_loss, 3)}')
                epoch += 1

                if counter >= patience :
                    break
                if (epoch > 500) and (now_loss < 1) :
                    break
            print('STOP TRAINING')

            model.eval()
            with torch.no_grad() :
                outputs = model(Input_scaled.to(device)).cpu().numpy()
                df_result[f'AE_output'] = scaler.inverse_transform(outputs)[:, 0] 

            recon_error =  abs(df_result[f'AE_output'] - df_result[f'{var_name_target}_val'])
            df_result[f'AE_0.98'] = np.quantile(recon_error, 0.98)

            gc.collect()

            dict_total[var_name_target] = df_result.copy()

            df_upper = df_result[df_result[f'{var_name_target}_val'] > df_result['LR_0.99']]
            df_upper = df_upper[df_upper[f'{var_name_target}_val'] > df_upper['GB_0.99']]
            df_upper = df_upper[abs(df_upper[f'AE_output'] - df_upper[f'{var_name_target}_val']) > df_upper['AE_0.98']]
            df_upper['Direction'] = 'Upper'

            df_under = df_result[df_result[f'{var_name_target}_val'] < df_result['LR_0.01']]
            df_under = df_under[df_under[f'{var_name_target}_val'] < df_under['GB_0.01']]
            df_under = df_under[abs(df_under[f'AE_output'] - df_under[f'{var_name_target}_val']) > df_under['AE_0.98']]
            df_under['Direction'] = 'Under'

            df_outlier = pd.concat([df_upper, df_under], axis = 0)

            dict_outlier[var_name_target] = df_outlier.copy()

            gc.collect()
        
        else : 
            ################## 1. Support vector Machine ##########
            
            print('SUPPORT VECTOR MACHINE')
            Input_unscaled = df_merged.drop(['Patient_id', 'Target_date',  f'{var_name_target}_val'], axis = 1) 
            scaler = RobustScaler()
            Input_scaled = scaler.fit_transform(Input_unscaled)
            
            model = OneClassSVM(kernel = 'rbf', nu = 0.02, gamma = 'scale')
            model.fit(Input_scaled)
            
            print(0.98)
            df_result[f'SVM_score'] = - model.decision_function(Input_scaled)
            df_result[f'SVM_0.98'] = np.quantile(df_result[f'SVM_score'], 0.98)
            
            gc.collect()
            
            ################## 2. Isolation Forest ##########
            print('ISOLATION FOREST')
            Input_unscaled = df_merged.drop(['Patient_id', 'Target_date',  f'{var_name_target}_val'], axis = 1) 
            scaler = RobustScaler()
            Input_scaled = scaler.fit_transform(Input_unscaled)
            
            print(0.98)
            model = IsolationForest(contamination = 0.02)
            model.fit(Input_scaled)
            
            df_result[f'IF_score'] = - model.decision_function(Input_scaled)
            df_result[f'IF_0.98'] = np.quantile(df_result[f'IF_score'], 0.98)
            
            gc.collect()
            
            ################## 3. Auto Encoder ###################
            print('AUTOENCODER')
            Input_unscaled = df_merged.drop(['Patient_id', 'Target_date',  f'{var_name_target}_val'], axis = 1) 
            scaler = RobustScaler()
            Input_scaled = scaler.fit_transform(Input_unscaled)
            Input_scaled = torch.tensor(Input_scaled, dtype = torch.float32)

            Input_dataset = TensorDataset(Input_scaled)
            Input_loader = DataLoader(Input_dataset, batch_size = 64, shuffle = True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = Autoencoder(input_dim = Input_scaled.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
            criterion = nn.MSELoss()

            best_loss = float('inf')
            counter = 0
            patience = 5
            min_delta = 1e-3

            epoch = 1
            while True :
                model.train()

                total_loss = 0
                total_sample = 0

                for batch in Input_loader :
                    inputs = batch[0].to(device)
                    outputs = model(inputs)

                    loss = criterion(outputs, inputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * inputs.shape[0] 
                    total_sample += inputs.shape[0]

                now_loss = total_loss / total_sample 

                if best_loss - now_loss > min_delta : 
                    best_loss = now_loss
                    counter = 0
                else :
                    counter += 1

                print(f'Epoch {epoch} | Loss : {round(now_loss, 3)}')
                epoch += 1

                if counter >= patience :
                    break
                if (epoch > 500) and (now_loss < 1) :
                    break
            print('STOP TRAINING')

            model.eval()
            with torch.no_grad() :
                outputs = model(Input_scaled.to(device)).cpu().numpy()
                Input_scaled = Input_scaled.cpu().numpy()

            df_result[f'AE_error'] = np.mean(abs(outputs - Input_scaled), axis = 1)
            df_result[f'AE_0.98'] = np.quantile(df_result[f'AE_error'], 0.98)

            gc.collect()

            dict_total[var_name_target] = df_result.copy()
            
            df_outlier = df_result[df_result[f'SVM_score'] > df_result[f'SVM_0.98']]
            df_outlier = df_outlier[df_outlier[f'IF_score'] > df_outlier[f'IF_0.98']]
            df_outlier = df_outlier[df_outlier[f'AE_error'] > df_outlier[f'AE_0.98']]

            dict_outlier[var_name_target] = df_outlier.copy()

            gc.collect()
          
        print()
    
    return var_list_target, dict_total, dict_outlier

if __name__ == '__main__' :
    print('<LYDUS - Logical Accuracy>\n')

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
    operation_type_manual = bool(config.get('operation_type_manual'))
    target_variable = config.get('target_variable')
    automatic_num = config.get('automatic_num')
    recommend_num = config.get('recommend_num')

    var_list_target, dict_total, dict_outlier = get_logical_accuracy(quiq, model_ver, api_key,\
                                                                        operation_type_manual, target_variable, automatic_num, recommend_num)

    outlier_num = 0
    total_num = 0

    df_summary = pd.DataFrame(columns = ['Target Variable', 'Total Num', 'Outlier Num' 'Logical Accuracy (%)'])
    for idx, var_name_target in enumerate(var_list_target) :
        df_summary.at[idx, 'Target Variable'] = var_name_target
        try :
            df_summary.at[idx, 'Total Num'] = len(dict_total[var_name_target])
            total_num += len(dict_total[var_name_target])
            df_summary.at[idx, 'Outlier Num'] = len(dict_outlier[var_name_target])
            outlier_num += len(dict_outlier[var_name_target])
            df_summary.at[idx, 'Logical Accuracy (%)'] = (df_summary.at[idx, 'Total Num'] - df_summary.at[idx, 'Outlier Num']) / df_summary.at[idx, 'Total Num'] * 100
            df_summary.at[idx, 'Logical Accuracy (%)'] = round(df_summary.at[idx, 'Logical Accuracy (%)'], 2)

            if len(dict_outlier[var_name_target]) > 0 :
                dict_outlier[var_name_target].to_csv(save_path + f'/outlier_{idx}_{var_name_target}.csv', index = False)
        except :
            df_summary.at[idx, 'Total Num'] = np.nan
            df_summary.at[idx, 'Outlier Num'] = np.nan
            df_summary.at[idx, 'Logical Accuracy (%)'] = np.nan
        
    df_summary.to_csv(save_path + '/logical_accuracy_summary.csv', index = False)

    logical_accuracy = (total_num - outlier_num) / total_num * 100
    logical_accuracy = round(logical_accuracy, 2)

    with open(save_path + '/logical_accuracy_total.txt', 'w', encoding = 'utf-8') as file :
        file.write(f'Logical Accuracy (%) = {logical_accuracy}\n')
        file.write(f'Total Num = {total_num}\n')
        file.write(f'Outlier Num = {outlier_num}\n')

    print('<SUCCESS>')








