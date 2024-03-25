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
import openai
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')
print(torch.__version__)

config_yml_name = sys.argv[1]


with open(f'{config_yml_name}',encoding='utf-8') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

session_id = config_dict['General_session_id']
boxplotnum = config_dict['Rangevalidity_boxplotnum']

lab_table_name = config_dict['General_table_name']

lab_table_item_column_name = 'Variable_ID'
lab_table_label_column_name = 'Variable_name'
patient_id = 'Patient_number'
value_column_name = 'Value'
charttime_column_name = 'Record_datetime'


df=pd.read_csv(lab_table_name,low_memory=False)
newcolumns=['Primary_key','Variable_ID','Variable_category','Variable_name','Record_datetime','Value','Unit','Variable_type','Recorder','Recorder_position','Recorder_affiliation','Patient_number','Admission_number','Annotation_value','Mapping_info_1','Mapping_info_2']
df.columns=newcolumns

df['temp_value_float'] = pd.to_numeric(df['Value'], errors='coerce')

# Step 2: Filter out rows where the temporary column is NaN
df_filtered_all_columns = df.dropna(subset=['temp_value_float'])

# Step 3: Replace the original '값' column with the float-converted values and drop the temporary column
df_filtered_all_columns['Value'] = df_filtered_all_columns['temp_value_float']
df = df_filtered_all_columns.drop(columns=['temp_value_float'])

df.reset_index(inplace=True,drop=True)

filtered_groups = df.groupby(lab_table_label_column_name).filter(lambda x: len(x) > 1000)

labellist = list(filtered_groups.groupby(lab_table_label_column_name).count().sort_values(value_column_name, ascending=False).index)

#filtered_groups = df.groupby(lab_table_item_column_name).filter(lambda x: len(x) > 500)

#itemlist = list(filtered_groups.groupby(lab_table_item_column_name).count().sort_values(value_column_name, ascending=False).index)

itemlist=list(df.groupby(lab_table_item_column_name).count().sort_values(value_column_name,ascending=False).index[:len(labellist)])

def get_outlier(df=None, column=None, weight=1.5):
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR*weight
  
    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight
  
    outlier_idx_under = df[column][ df[column] < lowest ].index
    outlier_idx_upper = df[column][ df[column] > highest ].index
    return quantile_25,quantile_75,lowest,highest,outlier_idx_under,outlier_idx_upper

now1=datetime.datetime.now()
columns_to_extract=[patient_id, value_column_name,charttime_column_name]

total_outlier_df=pd.DataFrame()
summary_df=pd.DataFrame(columns=['변수명','변수 ID','total_num','outlier_under_num','outlier_upper_num','outlier_total_num','outlier_under_proportion','outlier_upper_proportion','outlier_total_proportion'])
errorlist=[]

for i in range(len(labellist)):
    labelname=labellist[i]
    itemidname=itemlist[i]
    
    locals()['{}'.format(itemidname)]=df[df[lab_table_label_column_name]==labelname]
    locals()['{}'.format(itemidname)]=locals()['{}'.format(itemidname)][columns_to_extract]
    locals()['{}'.format(itemidname)].dropna(inplace=True)
    locals()['{}'.format(itemidname)].reset_index(inplace=True,drop=True)
    firstlength=len(locals()['{}'.format(itemidname)])
    try:
        locals()['{}'.format(itemidname)]=locals()['{}'.format(itemidname)].astype({value_column_name:'float'})
        
        q25,q75,lowest,highest,outlier_idx_under,outlier_idx_upper = get_outlier(df=locals()['{}'.format(itemidname)], column=value_column_name, weight=1.5)

        outlier_df_under=locals()['{}'.format(itemidname)].iloc[outlier_idx_under]
        outlier_df_under.reset_index(inplace=True,drop=True)
        outlier_df_under['ITEMID']=itemidname
        outlier_df_under['LABEL']=labelname
        outlier_df_under['direction']='under'

        outlier_df_upper=locals()['{}'.format(itemidname)].iloc[outlier_idx_upper]
        outlier_df_upper.reset_index(inplace=True,drop=True)
        outlier_df_upper['ITEMID']=itemidname
        outlier_df_upper['LABEL']=labelname
        outlier_df_upper['direction']='upper'

        total_outlier_df=pd.concat([total_outlier_df,outlier_df_under])
        total_outlier_df=pd.concat([total_outlier_df,outlier_df_upper])

        secondlength_under=len(outlier_df_under)
        secondlength_upper=len(outlier_df_upper)
        secondlength_total=secondlength_under+secondlength_upper

        proportion_under=round(secondlength_under/firstlength,5)
        proportion_upper=round(secondlength_upper/firstlength,5)
        proportion_total=round(secondlength_total/firstlength,5)

        summary=[labelname,itemidname,firstlength,secondlength_under,secondlength_upper,secondlength_total,proportion_under,proportion_upper,proportion_total]
        summary_df.loc[len(summary_df)]=summary

        print(labelname,itemidname,"outlier_under:",proportion_under,"outlier_upper:",proportion_upper,"outlier_total:",proportion_total)
    except:
        print('########## something wrong ############')
        print(labelname,itemidname)
        errorlist.append(labelname)

try:
    os.mkdir('LYDUS_results')
except:
    pass
try:
    os.mkdir('LYDUS_results/'+str(session_id))
except:
    pass
try:
    os.mkdir('LYDUS_results/'+str(session_id)+'/'+'rangevalidity')
except:
    pass   

summary_df.to_csv('LYDUS_results/'+str(session_id)+'/'+'rangevalidity'+'/'+'summary_df.csv',index=False, encoding='cp949')
total_outlier_df.to_csv('LYDUS_results/'+str(session_id)+'/'+'rangevalidity'+'/'+'total_outlier_df.csv',index=False,encoding='cp949')

totalcount=0
frac_total=0
for i in range(len(summary_df)):
    totalcount+=summary_df['total_num'][i]
    frac_total+=((summary_df['total_num'][i]-summary_df['outlier_total_num'][i])/summary_df['total_num'][i])*summary_df['total_num'][i]
totalresult=round(frac_total/totalcount,4)
text1='total_result='+str(totalresult)
file=open('LYDUS_results/'+str(session_id)+'/rangevalidity/total_result.txt',"w")
file.write(text1)
file.close()

fig,axs=plt.subplots(1,boxplotnum,figsize=(boxplotnum*4,10))
flierprops = dict(marker='o', markerfacecolor='green', markersize=2,
                  linestyle='none',alpha=0.2)
for i, ax in enumerate(axs.flat):
    itemidname=itemlist[i]
    labelname=labellist[i]
    locals()['{}'.format(itemidname)]
    ax.boxplot(locals()['{}'.format(itemidname)]['Value'],flierprops=flierprops)
    ax.set_title(labelname,fontsize=20,fontweight='bold')
    ax.tick_params(axis='y',labelsize=14)
plt.tight_layout()
plt.savefig('LYDUS_results/'+str(session_id)+'/rangevalidity/botplots.png')


now2=datetime.datetime.now()
print(now2-now1)
