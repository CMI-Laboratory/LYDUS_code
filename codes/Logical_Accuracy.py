import torch
import json
import yaml
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import time
import torch.nn.functional as F
import pandas as pd
import numpy as np
import datetime
from mlinsights.mlmodel import QuantileLinearRegression as QLR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import os
import sys
import glob
from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI
import random
import matplotlib.pyplot as plt
import warnings
import re
warnings.filterwarnings('ignore')
print(torch.__version__)

now_temp1= datetime.datetime.now()
config_yml_name = sys.argv[1]
with open(f'{config_yml_name}',encoding='utf-8') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

operation_type = config_dict['Reliability_operation_type']
automatic_num = config_dict['Reliability_automatic_num']
os.environ["OPENAI_API_KEY"] = config_dict['open_api_key']

target_variable = config_dict['Reliability_target_variable']
gpt_model_type = config_dict['Reliability_gpt_model_type']
recommended_variable_num = config_dict['Reliability_recommended_variable_num']

table_name = config_dict['csv_path']
session_id = str(1)

table_item_column_name = 'Variable_ID'
table_label_column_name = 'Variable_name'
patient_id = 'Patient_number'
value_column_name = 'Value'
charttime_column_name = 'Record_datetime'

def query_gpt(target_variable, gpt_model_type, recommended_variable_num, labeltext):    
    usercontent='''다음 중 
    select * from (select ITEMID, count(*) as cnt from AAA group by ITEMID) v order by cnt desc
    '''

    #usercontent='다음 중 '
    #usercontent+= ('"' + target_variable + '"' + ' ')
    #usercontent+= ('와 연관성 및 상관성이 높다고 알려진 변수들을 정확히 ' + str(recommended_variable_num) + '개만 추천해줘. 그리고 무조건 단답으로만 답해주고 구분자는 ;를 써주고 다른 그 어떠한 말도 덧붙이지 말아줘. 리스트에 있는 용어와 정확히 같은 용어로 답해줘.\n')
    usercontent+= ('Among the following, please recommend exactly ' + str(recommended_variable_num) + ' variables that are known to be highly correlated with ')
    usercontent+= ('"' + target_variable + '"' + '. ')
    usercontent+= ('Do not answer anything other than the variable, and use ; as a delimiter. Please answer in exactly the same terms as in the list.')
    usercontent+=labeltext


    messages=[
            #{"role": "system", "content": usercontent_global3},
            {"role": "user", "content": usercontent}
        ]

    #completion=openai.ChatCompletion.create(model=gpt_model_type, messages=messages)
    client = OpenAI()
    completion=client.chat.completions.create(model=gpt_model_type, messages=messages)


    aa=completion.choices[0].message.content
    return aa

df=pd.read_csv(table_name,low_memory=False)
newcolumns=['Primary_key','Variable_ID','Variable_category','Variable_name','Record_datetime','Value','Unit','Variable_type','Recorder','Recorder_position','Recorder_affiliation','Patient_number','Admission_number','Annotation_value','Mapping_info_1','Mapping_info_2']
df.columns=newcolumns
df = df[df['Mapping_info_1'].fillna('').str.contains('event', case=False)]
#df = df[df['Mapping_info_2'].isin(['lab_events'])]
df.reset_index(inplace=True,drop=True)


df['temp_value_float'] = pd.to_numeric(df[value_column_name], errors='coerce')

# Step 2: Filter out rows where the temporary column is NaN
df_filtered_all_columns = df.dropna(subset=['temp_value_float'])

# Step 3: Replace the original '값' column with the float-converted values and drop the temporary column
df_filtered_all_columns[value_column_name] = df_filtered_all_columns['temp_value_float']
df = df_filtered_all_columns.drop(columns=['temp_value_float'])

df.reset_index(inplace=True,drop=True)

labellist=list(df.groupby(table_label_column_name).count().sort_values(value_column_name,ascending=False).index[:200])

labeltext=''
for i in range(len(labellist)):
    labeltext+="'"
    labeltext+=labellist[i]
    labeltext+="', "


dfdf1=pd.read_csv(table_name,low_memory=False)
newcolumns=['Primary_key','Variable_ID','Variable_category','Variable_name','Record_datetime','Value','Unit','Variable_type','Recorder','Recorder_position','Recorder_affiliation','Patient_number','Admission_number','Golden_label','Mapping_info_1','Mapping_info_2']
dfdf1.columns=newcolumns
df_dx1=dfdf1[dfdf1['Mapping_info_1']=='ICD9_Dx']
df_dx2=dfdf1[dfdf1['Mapping_info_1']=='ICD10_Dx']
df_dx3=pd.concat([df_dx1,df_dx2])

labellist_dx=list(df_dx3.groupby(table_label_column_name).count().sort_values(value_column_name,ascending=False).index[:100])

if len(labellist_dx)>0:
    labellist_dx=[x.replace(',','') for x in labellist_dx]

    for i in range(len(labellist_dx)):
        labeltext+="'"
        labeltext+=labellist_dx[i]
        labeltext+="', "

labeltext=labeltext[:-2]


if operation_type=='automatic':
    iterations=automatic_num
    print('operation type is automatic')
elif operation_type=='manual':
    iterations=1
    print('operation type is manual')
else:
    print('operation type is wrong')


individual_results_df=pd.DataFrame(columns=['Variable Name','Reliability','Lower Anomaly','Upper Anomaly']) ##20240723

totalcount=0
totalfrac=0
for iterr in range(iterations):
    total_answer_list=[]
    count=0
    
    if operation_type=='automatic':
        target_variable=labellist[iterr]
    print('target variable is',target_variable)
    
    for j in range(20):

        answer=query_gpt(target_variable, gpt_model_type,recommended_variable_num,labeltext)
        answer=answer.replace("'",'')
        answer_list=answer.split('; ')
        answer_list2=[]
        for i in range(len(answer_list)):
            if answer_list[i] in labellist:
                answer_list2.append(answer_list[i])
        if len(answer_list2)==recommended_variable_num:
            total_answer_list+=answer_list2
            count+=1
            print('gpt recommendation count+1', answer_list2)
        else:
            answer_list=answer.split(';')
            answer_list2=[]
            for i in range(len(answer_list)):
                if answer_list[i] in labellist:
                    answer_list2.append(answer_list[i])
            if len(answer_list2)==recommended_variable_num:
                total_answer_list+=answer_list2
                count+=1
                print('gpt recommendation count+1', answer_list2)

        if count==5:
            break

    tempdf=pd.DataFrame({'recommended_variables':total_answer_list})
    tempdf['nothing']=1
    aa=pd.DataFrame(tempdf.groupby('recommended_variables')['nothing'].count())  

    try:
        recommended_variable_list=list(aa.sort_values(by=['nothing'],ascending=False).drop([target_variable],axis=0)[:recommended_variable_num-1].index)
    except:
        recommended_variable_list=list(aa.sort_values(by=['nothing'],ascending=False)[:recommended_variable_num-1].index)

    print('recommended variables:' ,recommended_variable_list)

    locals()['{}'.format(target_variable)] = df[df[table_label_column_name]==target_variable]
    locals()['{}'.format(target_variable)].reset_index(inplace=True,drop=True)
    #print(locals()['{}'.format(target_variable)])
    distinct_target_variable_category=list(set(locals()['{}'.format(target_variable)]['Variable_category']))
    for varcat in distinct_target_variable_category:
        locals()['{}'.format(target_variable+varcat)]= locals()['{}'.format(target_variable)][locals()['{}'.format(target_variable)]['Variable_category']==varcat]
        locals()['{}'.format(target_variable+varcat)][charttime_column_name]=pd.to_datetime(locals()['{}'.format(target_variable+varcat)][charttime_column_name])
        locals()['{}'.format(target_variable+varcat)][charttime_column_name]=locals()['{}'.format(target_variable+varcat)][charttime_column_name].dt.tz_localize(None)
    if len(distinct_target_variable_category)==1:
        locals()['{}'.format(target_variable)]=locals()['{}'.format(target_variable+varcat)]
    else:
        for varcat in distinct_target_variable_category:
            locals()['{}'.format(target_variable)]=locals()['{}'.format(target_variable+varcat)]
            break
        for vv in range(1,len(distinct_target_variable_category)):
            locals()['{}'.format(target_variable)]=pd.concat([locals()['{}'.format(target_variable)],locals()['{}'.format(target_variable+varcat)]])
    #print('target length:',len(locals()['{}'.format(target_variable)]))
    locals()['{}'.format(target_variable)].reset_index(inplace=True,drop=True)
    #print(locals()['{}'.format(target_variable)])

    for i in range(len(recommended_variable_list)):
        rec_var=recommended_variable_list[i]
        locals()['{}'.format(rec_var)] = df[df[table_label_column_name]==rec_var]
        locals()['{}'.format(rec_var)].reset_index(inplace=True,drop=True)
        
        distinct_rec_var_category=list(set(locals()['{}'.format(rec_var)]['Variable_category']))
        for varcat in distinct_rec_var_category:
            locals()['{}'.format(rec_var+varcat)]= locals()['{}'.format(rec_var)][locals()['{}'.format(rec_var)]['Variable_category']==varcat]
            locals()['{}'.format(rec_var+varcat)][charttime_column_name]=pd.to_datetime(locals()['{}'.format(rec_var+varcat)][charttime_column_name])
            locals()['{}'.format(rec_var+varcat)][charttime_column_name]=locals()['{}'.format(rec_var+varcat)][charttime_column_name].dt.tz_localize(None)
        if len(distinct_rec_var_category)==1:
            locals()['{}'.format(rec_var)]=locals()['{}'.format(rec_var+varcat)]
        else:
            for varcat in distinct_rec_var_category:
                locals()['{}'.format(rec_var)]=locals()['{}'.format(rec_var+varcat)]
                break
            for vv in range(1,len(distinct_rec_var_category)):
                locals()['{}'.format(rec_var)]=pd.concat([locals()['{}'.format(rec_var)],locals()['{}'.format(rec_var+varcat)]])
        #print(f'rec var {i+1}:',rec_var, ', length:',len(locals()['{}'.format(rec_var)]))

    columns_to_extract=[patient_id, value_column_name,charttime_column_name]
    tempcolumns=columns_to_extract.copy()
    tempcolumns[1]=target_variable
    locals()['{}'.format(target_variable)]=locals()['{}'.format(target_variable)][columns_to_extract]
    locals()['{}'.format(target_variable)].columns=tempcolumns
    locals()['{}'.format(target_variable)].dropna(inplace=True)
    locals()['{}'.format(target_variable)].reset_index(inplace=True,drop=True)
    locals()['{}'.format(target_variable)].columns=[patient_id,target_variable,charttime_column_name+'_target']
    print('target length:',len(locals()['{}'.format(target_variable)]))
    
    
    for i in range(len(recommended_variable_list)):
        rec_var=recommended_variable_list[i]

        columns_to_extract=[patient_id, value_column_name,charttime_column_name]
        tempcolumns=columns_to_extract.copy()
        tempcolumns[1]=rec_var
        locals()['{}'.format(rec_var)]=locals()['{}'.format(rec_var)][columns_to_extract]
        locals()['{}'.format(rec_var)].columns=tempcolumns
        locals()['{}'.format(rec_var)].dropna(inplace=True)
        locals()['{}'.format(rec_var)].reset_index(inplace=True,drop=True)
        print(f'rec var {i+1}:',rec_var, ', length:',len(locals()['{}'.format(rec_var)]))

    merge1=locals()['{}'.format(target_variable)].copy()
    merge1_final=locals()['{}'.format(target_variable)].copy()

    for i in range(len(recommended_variable_list)):
    #for i in range(1):
        rec_var=recommended_variable_list[i]
        charttime_column_name_target=charttime_column_name+'_target'
        
        if rec_var not in labellist_dx:
            merge1=pd.merge(merge1,locals()['{}'.format(rec_var)],on=[patient_id],how='left')
            merge1['timediff']=abs(merge1[charttime_column_name_target]-merge1[charttime_column_name])
            merge1['timediff2']= merge1['timediff'].dt.days*86400 + merge1['timediff'].dt.seconds
            merge1['rank']=merge1.groupby([patient_id,charttime_column_name_target])['timediff2'].rank(method='first',na_option='bottom')
            merge1=merge1[merge1['rank']==1]
            #print(merge1)
            merge1_temp=merge1[merge1['timediff2']<=86400][[patient_id,target_variable,charttime_column_name_target,rec_var]]
            merge1_final=pd.merge(merge1_final,merge1_temp,on=[patient_id,target_variable,charttime_column_name_target],how='left')
            merge1_final.reset_index(inplace=True,drop=True)
            #print(merge1_final)
            merge1=merge1_final.copy()
        else:
            dxlabel=[]
            for mm in range(len(merge1)):
                pidtemp=merge1['Patient_number'][mm]
                if pidtemp in list(locals()['{}'.format(rec_var)]['Patient_number']):
                    dxlabel.append(1)
                else:
                    dxlabel.append(0)
            merge1[rec_var]=dxlabel

    merge1.drop([charttime_column_name_target],axis=1,inplace=True)
    merge1.drop([patient_id],axis=1,inplace=True)
    print('merged table length: ', len(merge1))

    #now = datetime.datetime.now()
    #nowStr2 = "{:%Y%m%d%H%M%S}".format(now)
    #os.mkdir(nowStr2)
    try:
        os.mkdir('LYDUS_results')
    except:
        pass
    try:
        os.mkdir('LYDUS_results/'+str(session_id))
    except:
        pass
    try:
        os.mkdir('LYDUS_results/'+str(session_id)+'/'+'reliability')
    except:
        pass 

    gpu_n =0
    device = torch.device('cuda:{}'.format(gpu_n)if torch.cuda.is_available() else 'cpu')

    #merged_table=pd.read_csv(merged_table_name)
    merged_table=merge1
    merged_table=merged_table.dropna()
    merged_table.reset_index(inplace=True,drop=True)

    merged_table_columns=list(merged_table.columns)
    merged_table_columns2=[]
    for i in range(len(merged_table_columns)):
        merged_table_columns2.append(merged_table_columns[i].replace(' ',''))
    merged_table_columns2
    merged_table.columns=merged_table_columns2
    
    tempcolumns=merged_table.columns
    tempcolumns2=[]
    for col in tempcolumns:
        newcol=re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", col)
        tempcolumns2.append(newcol)
    merged_table.columns=tempcolumns2    

    target_variable=target_variable.replace(' ','')

    kf = KFold(n_splits=2,shuffle=True) # Define the split - into 2 folds 

    totaldf=pd.DataFrame()

    for train_index, test_index in kf.split(merged_table):

        now = datetime.datetime.now()
        nowStr = "{:%Y%m%d%H%M%S}".format(now)
        #filepathsave=nowStr2+'/'+nowStr
        filepathsave='LYDUS_results/'+str(session_id)+'/'+'reliability/'+nowStr+'/'
        try:
            os.mkdir(filepathsave)
        except:
            pass   
        
        #os.mkdir(filepathsave)
        #filepathsave+='/'

        X_train, X_test = merged_table.iloc[train_index], merged_table.iloc[test_index]

        X_train_results=X_train.copy()
        X_test_results=X_test.copy()

        y_train=X_train[target_variable]
        y_test=X_test[target_variable]
        X_train_regression=X_train.drop(target_variable,axis=1)
        X_test_regression=X_test.drop(target_variable,axis=1)

        X_columns=list(X_train.columns)
        X_columns2=[]
        for i in range(len(X_columns)):
            X_columns2.append(X_columns[i]+'_scaled')

        regression_independent_variables=list(X_train_regression.columns)
        regression_formula=''
        regression_formula+=target_variable+' ~ '
        for i in range(len(regression_independent_variables)):
            regression_formula+=(regression_independent_variables[i] + ' + ')

        regression_formula=regression_formula[:-3]

        all_models = {}
        common_params = dict(
            learning_rate=0.05,
            n_estimators=200,
            max_depth=2,
            min_samples_leaf=9,
            min_samples_split=9,
        )

        for alpha in [0.01, 0.99]:
            gbr = GBR(loss="quantile", alpha=alpha, **common_params)
            all_models["q %1.2f" % alpha] = gbr.fit(X_train_regression, y_train)
            print('==')

        y_lower = all_models["q 0.01"].predict(X_test_regression)
        y_upper = all_models["q 0.99"].predict(X_test_regression)
        X_test_results['GBR_0.01']=y_lower
        X_test_results['GBR_0.99']=y_upper

        for quantile in [0.01, 0.99]:
            quantile_reg = smf.quantreg(regression_formula, X_train).fit(q = quantile)
            pred = quantile_reg.predict(X_test_regression)
            X_test_results['LR_'+str(quantile)] = list(pred)

        sc=StandardScaler()
        sc.fit(X_train)
        X_train2=sc.transform(X_train)
        X_test2=sc.transform(X_test)

        X_train2=pd.DataFrame(X_train2,columns=X_columns2)
        X_test2=pd.DataFrame(X_test2,columns=X_columns2)

        X_train3=pd.concat([X_train,X_train2],axis=1)
        X_test3=pd.concat([X_test,X_test2],axis=1)

        class LYDUS_AE_train(Dataset):

            def __init__(self):
                self.len=len(X_train)
                self.x=torch.tensor(X_train2.values,dtype=torch.float32)

            def __len__(self):
                return self.len

            def __getitem__(self,index):
                values=self.x[index]

                return values

        class LYDUS_AE_test(Dataset):

            def __init__(self):
                self.len=len(X_test)
                self.x=torch.tensor(X_test2.values,dtype=torch.float32)

            def __len__(self):
                return self.len

            def __getitem__(self,index):
                values=self.x[index]

                return values

        train_dataset = LYDUS_AE_train()
        test_dataset = LYDUS_AE_test()

        bs=8

        train_dataset_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=bs, drop_last=False)
        test_dataset_dataloader = DataLoader(test_dataset,batch_size=bs)

        class LYDUS_AE_model(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.encoder_hidden_layer = nn.Linear(
                    in_features=len(X_columns), out_features=int((recommended_variable_num+1)//1.5)
                )
                self.encoder_output_layer = nn.Linear(
                    in_features=int((recommended_variable_num+1)//1.5), out_features=int((recommended_variable_num+1)//2)
                )
                self.decoder_hidden_layer = nn.Linear(
                    in_features=int((recommended_variable_num+1)//2), out_features=int((recommended_variable_num+1)//1.5)
                )
                self.decoder_output_layer = nn.Linear(
                    in_features=int((recommended_variable_num+1)//1.5), out_features=len(X_columns)
                )

            def forward(self, features):
                activation = self.encoder_hidden_layer(features)
                #activation = torch.relu(activation)
                activation = torch.tanh(activation)
                code = self.encoder_output_layer(activation)
                #code = torch.relu(code)
                code = torch.tanh(code)
                activation = self.decoder_hidden_layer(code)
                #activation = torch.relu(activation)
                activation = torch.tanh(activation)
                activation = self.decoder_output_layer(activation)
                #reconstructed = torch.relu(activation)
                reconstructed = torch.tanh(activation)
                return reconstructed

        model=LYDUS_AE_model().to(device)
        criterion = nn.MSELoss().to(device)

        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=0.5)

        count=0
        togglecount=0
        time1=datetime.datetime.now()
        for epoch in range(500):
            epoch3digits=('000'+str(epoch))[-3:]

            if togglecount==10:
                break

            for phase in ['training','test']:
                if phase=='training':
                    model.train()
                    running_loss=0.0

                    for param_group in optimizer.param_groups:
                        print(str(epoch),'learning rate:',param_group['lr'])
                    for i,data in enumerate(train_dataset_dataloader):
                        data=data.to(device).float()
                        optimizer.zero_grad()
                        outputs=model(data)
                        loss=criterion(outputs,data)
                        loss.backward()
                        optimizer.step()
                        running_loss+=loss.item()

                    print('    train loss:',round(running_loss,2))     


                if phase=='test':
                    model.eval()
                    running_loss=0.0

                    with torch.no_grad():
                        for i,data in enumerate(test_dataset_dataloader):
                            data=data.to(device).float()
                            outputs=model(data)
                            loss=criterion(outputs,data)
                            running_loss+=loss.item()

                    scheduler.step(running_loss)        
                    print('    test loss:',round(running_loss,2)) 

                    if count==0:
                        bestloss=running_loss
                        torch.save(model.state_dict(),'./'+filepathsave+str(epoch3digits)+'_'+str(round(bestloss,2))+'.pt')
                    else:
                        if running_loss < bestloss*0.999:
                            bestloss=running_loss
                            togglecount=0
                            torch.save(model.state_dict(),'./'+filepathsave+str(epoch3digits)+'_'+str(round(bestloss,2))+'.pt')
                        else:
                            togglecount+=1
                    count+=1
        time2=datetime.datetime.now()
        print('time elapsed',time2-time1)

        model=LYDUS_AE_model().to(device)

        modelfiles=os.listdir(filepathsave)
        modelfiles.sort()
        themodel=modelfiles[-1]


        model.load_state_dict(torch.load('./'+filepathsave+themodel))
        criterion = nn.MSELoss().to(device)


        predictions=[]

        model.eval()
        with torch.no_grad():
            for i,data in enumerate(test_dataset_dataloader):
                data=data.to(device).float()
                outputs=model(data)
                for j in range(len(outputs)):
                    predictions.append(outputs[j].cpu().numpy())

        output=pd.DataFrame(predictions,columns=X_columns)
        output=pd.DataFrame(sc.inverse_transform(output),columns=X_columns)

        X_test_results['AE']=list(output[target_variable])

        totaldf=pd.concat([totaldf,X_test_results])


    totaldf_upper=totaldf[totaldf[target_variable] > totaldf['GBR_0.99']]
    totaldf_upper=totaldf_upper[totaldf_upper[target_variable] > totaldf_upper['LR_0.99']]
    totaldf_upper=totaldf_upper[totaldf_upper[target_variable] > totaldf_upper['AE']*1.5]
    totaldf_under=totaldf[totaldf[target_variable] < totaldf['GBR_0.01']]
    totaldf_under=totaldf_under[totaldf_under[target_variable] < totaldf_under['LR_0.01']]
    totaldf_under=totaldf_under[totaldf_under[target_variable] < totaldf_under['AE']*0.5]
    print('==========================')
    #print('total count',len(totaldf))
    print('upper count',len(totaldf_upper))
    print('under count',len(totaldf_under))
    filepathsave2='LYDUS_results/'+str(session_id)+'/'+'reliability/'
    totaldf.to_csv(filepathsave2 + str(iterr)+target_variable+ 'total.csv')
    totaldf_upper.to_csv(filepathsave2+ str(iterr)+target_variable+ 'upper.csv')
    totaldf_under.to_csv(filepathsave2+ str(iterr)+target_variable+ 'under.csv')
    
    totalcount += len(totaldf)
    totalfrac +=  ( ( len(totaldf) - len(totaldf_upper) - len(totaldf_under) )/len(totaldf) ) * len(totaldf)
    
    individual_results_list=[  target_variable , round(( ( len(totaldf) - len(totaldf_upper) - len(totaldf_under) )/len(totaldf) ),4), round(( ( len(totaldf_under) )/len(totaldf) ),4), round(( ( len(totaldf_upper) )/len(totaldf) ),4)  ] #20240723
    individual_results_df.loc[len(individual_results_df)]=individual_results_list #20240723
                  
totalresult=round(totalfrac/totalcount,4)
text1='total_result='+str(totalresult)
file=open(filepathsave2+'total_result.txt',"w")
file.write(text1)
file.close()        

individual_results_df.to_csv(filepathsave2+'individual_results.csv',index=False) #20240723
plt.boxplot( individual_results_df['Reliability']) #20240723
plt.savefig(filepathsave2+'total_boxplot.png') #20240723
    
now_temp2= datetime.datetime.now()
print(now_temp2-now_temp1)
