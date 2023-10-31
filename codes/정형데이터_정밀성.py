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
import math
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from decimal import Decimal
from collections import Counter
import math
import re
import warnings
warnings.filterwarnings('ignore')
print(torch.__version__)

now1= datetime.datetime.now()

config_yml_name = sys.argv[1]
with open(f'{config_yml_name}',encoding='utf-8') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

lab_table_name = config_dict['General_table_name']
session_id = config_dict['General_session_id']

lab_table_item_column_name = '변수 ID'
lab_table_label_column_name = '변수명'
patient_id = '환자 번호'
value_column_name = '값'
charttime_column_name = '기록 날짜'

def gini_simpson(data):
    total_count = len(data)
    counter = Counter(data)
    gini_index = 1.0
    diversity = len(counter)/total_count

    for count in counter.values():
        probability = count / total_count
        gini_index -= probability**2

    return gini_index, diversity

def round_at_3(x):
    x=round(x,3)
    return x

def multiply_by(x,a):
    x=x*(10**a)
    return x

try:
    os.mkdir('LYDUS_results')
except:
    pass
try:
    os.mkdir('LYDUS_results/'+str(session_id))
except:
    pass
try:
    os.mkdir('LYDUS_results/'+str(session_id)+'/'+'granularity')
except:
    pass   
try:
    os.mkdir('LYDUS_results/'+str(session_id)+'/'+'granularity/histograms')
except:
    pass   

savepath1='LYDUS_results/'+str(session_id)+'/'+'granularity/'
savepath2='LYDUS_results/'+str(session_id)+'/'+'granularity/histograms/'

df=pd.read_csv(lab_table_name,low_memory=False)

#df=df.sample(frac=0.01,replace=False)

labellist=list(df.groupby(lab_table_label_column_name).count().sort_values(value_column_name,ascending=False).index[:100])
itemlist=list(df.groupby(lab_table_item_column_name).count().sort_values(value_column_name,ascending=False).index[:100])

totaltable=pd.DataFrame({lab_table_item_column_name:itemlist,lab_table_label_column_name:labellist})
totaltable['count']=np.nan
totaltable['decimalnum']=np.nan
totaltable['Granularity']=np.nan

for i in range(len(itemlist)):
    dftemp=df[df['변수명']==labellist[i]]
    dftemp.reset_index(inplace=True,drop=True)
    columns_to_extract=[patient_id, value_column_name,charttime_column_name]
    dftemp=dftemp[columns_to_extract]
    dftemp.dropna(inplace=True)
    dftemp.reset_index(inplace=True,drop=True)

    dftemp[value_column_name]=dftemp[value_column_name].apply(round_at_3)

    cnt=len(dftemp)

    decimalnum=-4

    for iterr in range(5):
        dftemp2=dftemp.copy()
        dftemp2[value_column_name]=dftemp2[value_column_name].apply(multiply_by,args=(3-iterr,))
        tempnum=0
        for j in range(len(dftemp2)):
            if math.isclose(dftemp2[value_column_name][j]%10, 0, abs_tol=0.001)==True or math.isclose(-dftemp2[value_column_name][j]%10, 0, abs_tol=0.001)==True:
                tempnum+=1
        if tempnum/len(dftemp2)>0.99:
            pass
        else:
            decimalnum=3-iterr
            break

    dftemp2=dftemp.copy()
    dftemp2[value_column_name]=dftemp2[value_column_name].apply(multiply_by,args=(3-iterr,))
    values=[]
    for j in range(len(dftemp2)):
        values.append(int(dftemp2[value_column_name][j]%10))
    a,b=gini_simpson(values)
    print(labellist[i],itemlist[i],round(a,4))
    totaltable['count'][i]=len(dftemp2)
    totaltable['decimalnum'][i]=decimalnum
    totaltable['Granularity'][i]=a

    plt.hist(values)
    plt.title(labellist[i])
    label_temp=re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", labellist[i])

    plt.savefig(savepath2+str(itemlist[i])+ ' ' +label_temp+'.png')
    plt.close()

totalcount=np.sum(totaltable['count'])
totalgranularity=np.sum(totaltable['count']*totaltable['Granularity'])/totalcount

totaltable.to_csv(savepath1+'totalresults.csv')

text1='total_result='+str(round(totalgranularity,4))
file=open( savepath1+'totalgranularity.txt' ,"w")
file.write(text1)
file.close()

now2=datetime.datetime.now()
print(now2-now1)