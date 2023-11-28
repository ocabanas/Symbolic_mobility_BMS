#!/usr/bin/env python
# coding: utf-8

# In[1]:


sample_size = 1000
mcmc_resets = 5
mcmc_steps = 12000
mcmc_ens_avg=[10000,100]
log_flows = True
new_train_test=False
#"""
XLABS = [
    'd',
    'm_o',
    'm_d']
params = 6


# In[2]:


import pandas as pd
import numpy as np
import sys
import warnings
import gc
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from copy import deepcopy,copy
from ipywidgets import IntProgress
from itertools import chain
from IPython.display import display
from datetime import datetime
import pickle
import os
# Since the 'user' column do not have relevant information will not be read

# Import Machine Scientist
from importlib.machinery import SourceFileLoader
path = './rguimera-machine-scientist/machinescientist.py'
ms = SourceFileLoader('ms', path).load_module()

# Read data
states=['New York','Massachusetts','California','Florida','Washington','Texas']


# In[3]:


print(np.__version__)


# In[4]:


print(np.__version__)


# In[5]:


def get_train_test(features):
    import random
    name_unique = np.unique(list(features.name_o)+list(features.name_d))
    random.shuffle(name_unique)
    split = len(name_unique) // 2
    name_train = name_unique[:split]
    name_test = name_unique[split:]
    
    train_data = features[(features.name_o.isin(name_train))&(features.name_d.isin(name_train))]
    test_data = features[(features.name_o.isin(name_test))&(features.name_d.isin(name_test))]
    return train_data,test_data
def delete_nan(a,b):
    a_new=[i for i,j in zip(a,b) if i>0. and i<np.inf and j>0. and j<np.inf and i!=np.nan and j!=np.nan]
    b_new=[j for i,j in zip(a,b) if i>0. and i<np.inf and j>0. and j<np.inf and i!=np.nan and j!=np.nan]
    return a_new,b_new
def common_part_of_commuters(values1, values2):
    values1,values2=delete_nan(values1,values2)
    tot = np.sum(values1) + np.sum(values2)
    return 2.0 * np.sum(np.minimum(values1, values2)) / tot
def common_part_of_commuters_accuracy(real, predicted):
    real,predicted=delete_nan(real,predicted)
    tot = 2.*np.sum(real)
    return 2.0 * np.sum(np.minimum(real,predicted)) / tot
def RMSE(real,predicted):
    #from sklearn.metrics import mean_squared_error
    #return mean_squared_error(real,predicted, squared=False)
    real,predicted=delete_nan(real,predicted)
    return np.sqrt(np.square(np.subtract(real,predicted)).mean())
def RE(real,predicted):
    real,predicted=delete_nan(real,predicted)
    return np.array(list(np.abs((y-y1)/y1) for y,y1 in zip(predicted,real))).mean()
def ensemble_prediction_median_np(list_models,data):
    ens_pred=[]
    for i,x in data.iterrows():
        res=[]
        for model in list_models:
            df=pd.DataFrame(x).transpose()
            res1=model.predict(df)
            if np.iscomplex(res1.values[0])==False:
                res.append(res1.values[0])
        if log_flows==True:
            median=np.nanmedian(np.exp(res))
        else:
            median=np.nanmedian(res)
        ens_pred.append(median)
    return ens_pred
def ensemble_prediction_median(list_models,data):
    import multiprocessing
    num_cores = max(int(multiprocessing.cpu_count()/2.),1)  #half of available cores or one
    #print('Debug, num cores:',num_cores)
    num_cores=5
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(data, num_partitions)
    global func_median
    def func_median(data):
        ens_pred=[]
        for i,x in data.iterrows():
            res=[]
            for model in list_models:
                df=pd.DataFrame(x).transpose()
                res1=model.predict(df)
                if np.iscomplex(res1.values[0])==False:
                    res.append(res1.values[0])
            if log_flows==True:
                median=np.nanmedian(np.exp(res))
            else:
                median=np.nanmedian(res)
            ens_pred.append(median)
        return ens_pred
    pool = multiprocessing.Pool(num_cores)
    #df = pd.concat(pool.map(func_median, df_split))
    df=list(chain(*pool.map(func_median, df_split)))
    pool.close()
    pool.join()
    return df


# ### Train/Test

# In[6]:


if new_train_test==True:
    list_states_dataframes = [
        pd.read_pickle('data/NewYork/city_trips_features_2022_10_28-04_02_13.pkl'),#.sample(machinescientist_sample_size),
        pd.read_pickle('data/Massachusetts/city_trips_features_2022_10_28-09_28_48.pkl'),#.sample(machinescientist_sample_size),
        pd.read_pickle('data/California/city_trips_features_2022_10_30-03_06_27.pkl'),#.sample(machinescientist_sample_size),
        pd.read_pickle('data/Florida/city_trips_features_2022_10_31-12_54_59.pkl'),#.sample(machinescientist_sample_size),
        pd.read_pickle('data/Washington/city_trips_features_2022_10_31-07_50_13.pkl'),#ample(machinescientist_sample_size),
        pd.read_pickle('data/Texas/city_trips_features_2022_11_01-04_52_48.pkl'),#ample(machinescientist_sample_size),
    ]

    train_list_dataframes={}
    test_list_dataframes={}
    train_list_sample={}
    test_list_sample={}
    for i,df in enumerate(list_states_dataframes):
        print(df.columns)
        # Get list of all cities
        df=df[df.total_pop_flow>0.0]
        train,test=get_train_test(df)
        #
        train_list_dataframes[states[i]]=train
        train_list_sample[states[i]]=train.sample(sample_size)
        test_list_dataframes[states[i]]=test
        test_list_sample[states[i]]=test.sample(sample_size)

    #
    name=open('./data/checkpoints/'+f'list_states_dataframes_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl', "wb")
    pickle.dump(list_states_dataframes, name)
    print(name)
    name.close()
    #
    name=open('./data/checkpoints/'+f'fold1_dataframes_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl', "wb")
    pickle.dump(train_list_dataframes, name)
    print(name)
    name.close()
    #
    name=open('./data/checkpoints/'+f'fold1_sample_dataframes_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl', "wb")
    pickle.dump(train_list_sample, name)
    print(name)
    name.close()
    #
    name=open('./data/checkpoints/'+f'fold2_dataframes_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl', "wb")
    pickle.dump(test_list_dataframes, name)
    print(name)
    name.close()
    #
    name=open('./data/checkpoints/'+f'fold2_sample_dataframes_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl', "wb")
    pickle.dump(test_list_sample, name)
    print(name)
    name.close()
else:
    fold='fold1'
    name=open(f'./data/checkpoints/{fold}_sample_dataframes_2022_11_02-02_54_03.pkl', "rb")
    train_list_sample=pickle.load(name)
    name.close()
    


# ### Computing natural logarithm of population flow

# In[7]:


if log_flows==True:
    for i,key in enumerate(states):
        train_list_sample[states[i]]['total_pop_flow']=train_list_sample[states[i]]['total_pop_flow'].apply(lambda x : np.log(x))
        print(train_list_sample[states[i]].head())
    log_scale=False
else:
    log_scale=True


# # Machine Scientist

# ## BMS C: One model, multiDataFrame

# In[ ]:


res={}



best_model, state_ensemble, fig = ms.machinescientist(x=train_list_sample,
                                               y={key:i['total_pop_flow'] for key,i in train_list_sample.items()},
                                               XLABS=XLABS,n_params=params,
                                               resets=mcmc_resets,
                                               steps_prod=mcmc_steps,
                                               log_scale_prediction=log_scale,
                                                ensemble_avg=mcmc_ens_avg
                                              )

name=f'./data/checkpoints/{fold}_state_model_C_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl'
with open(name, 'wb') as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(name)
name=f'./data/checkpoints/{fold}_ensemble_model_C_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl'
with open(name, 'wb') as handle:
    pickle.dump(state_ensemble, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(name)


# In[14]:


del best_model, state_ensemble


# In[ ]:


import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
import gc
gc.collect()


# ## BMS A: One model

# In[1]:


res={}
from copy import deepcopy
from ipywidgets import IntProgress
from IPython.display import display

x_data=pd.concat(list(train_list_sample.values()))
y_data=pd.concat(list(i['total_pop_flow'] for i in train_list_sample.values()))

best_model, state_ensemble = ms.machinescientist(x=x_data,y=y_data,
                                                   XLABS=XLABS,n_params=params,
                                                   resets=mcmc_resets,
                                                   steps_prod=mcmc_steps,
                                                   log_scale_prediction=log_scale,
                                                   ensemble_avg=mcmc_ens_avg
                                                  )
name=f'./data/checkpoints/{fold}_state_model_A_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl'
with open(name, 'wb') as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(name)
name=f'./data/checkpoints/{fold}_ensemble_model_A_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl'
with open(name, 'wb') as handle:
    pickle.dump(state_ensemble, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(name)


# In[ ]:


del best_model, state_ensemble


# In[ ]:


import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
import gc
gc.collect()


# ## BMS B: Multiple models

# In[ ]:


#from memory_profiler import profile
res={}
state_model={}
state_ensemble={}
for key,frame in train_list_sample.items():

    best_model_train, list_ensemble_train = ms.machinescientist(x=train_list_sample[key],
                                                               y=train_list_sample[key]['total_pop_flow'],
                                                               XLABS=XLABS,n_params=params,
                                                               resets=mcmc_resets,
                                                               steps_prod=mcmc_steps,
                                                               log_scale_prediction=log_scale,
                                                               ensemble_avg=mcmc_ens_avg
                                                              )
    state_model[key]=copy(best_model_train)
    state_ensemble[key]=copy(list_ensemble_train)
    gc.collect()
name=f'./data/checkpoints/{fold}_state_model_B_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl'
with open(name, 'wb') as handle:
    pickle.dump(state_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(name)
name=f'./data/checkpoints/{fold}_ensemble_model_B_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl'
with open(name, 'wb') as handle:
    pickle.dump(state_ensemble, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(name)


# In[ ]:


del state_model, state_ensemble


# In[ ]:


import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
import gc
gc.collect()

