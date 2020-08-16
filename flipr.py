# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 10:22:34 2020

@author: saurav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train=pd.read_excel('Train_dataset.xlsx')
df_test=pd.read_excel('Test_dataset.xlsx')
df_test2=pd.read_excel('Test_dataset.xlsx','Put-Call_TS',skiprows=1)

indexes=df_test2.iloc[:,0]

df_test1=df_test2.iloc[:,1:]


corrmat=df_train.corr()

df_train.iloc[:,:].isnull().sum()

from sklearn.preprocessing import LabelEncoder

lb1=LabelEncoder()
df_train.iloc[:,1]=lb1.fit_transform(df_train.iloc[:,1])
df_test.iloc[:,1]=lb1.transform(df_test.iloc[:,1])

lb2=LabelEncoder()
df_train.iloc[:,2]=lb2.fit_transform(df_train.iloc[:,2])
df_test.iloc[:,2]=lb2.transform(df_test.iloc[:,2])

plt.hist(df_train.iloc[:,13])

#4,9,11

X=df_train.iloc[:,1:14].values
y=df_train.iloc[:,14].values

df_test=df_test.iloc[:,1:].values

from sklearn.impute import SimpleImputer
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:,[3,8,10]]=imp_mode.fit_transform(X[:,[3,8,10]])
df_test[:,[3,8,10]]=imp_mode.transform(df_test[:,[3,8,10]])

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,[0,1,2,4,5,6,7,9,11,12]]=imp_mean.fit_transform(X[:,[0,1,2,4,5,6,7,9,11,12]])
df_test[:,[0,1,2,4,5,6,7,9,11,12]]=imp_mean.transform(df_test[:,[0,1,2,4,5,6,7,9,11,12]])
df_test1=imp_mean.fit_transform(df_test1)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
mms=MinMaxScaler()
st=StandardScaler()
#X=st.fit_transform(X)
#df_test=st.transform(df_test)

X=mms.fit_transform(X)
df_test=mms.transform(df_test)
st1=StandardScaler()
#y=st1.fit_transform(y.reshape(len(y),1))



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.25, random_state=0)


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
#from sklearn.svm import SVR

#reg=SVR(C=1000)min_samples_split=5,min_samples_leaf=4,max_features='auto',max_depth=80,bootstrap=True
reg= RandomForestRegressor(n_estimators=200,criterion='mse',min_samples_split=5,min_samples_leaf=4,max_features='auto',max_depth=80,bootstrap=True)
reg.fit(X,y)

y_tr_pred=reg.predict(X_train)


y_pred=reg.predict(X_test)

y_pred1=reg.predict(df_test)



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


ret=rmse(y_pred,y_test)
retr=rmse(y_tr_pred,y_train)


#randomsearchcv

'''from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)

rf_random.best_params_'''

#Time Series

from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime

pc_16=[]

for i in range(len(df_test2)):
    #print(i)
    date=pd.date_range(start='08/10/2020', end='08/15/2020')
    date=pd.DataFrame(date)
    pc=df_test1[i,:]
    pc=pd.DataFrame(pc)
    pc=pc.values
    pc=pc.reshape(6,)
    pcd=pd.Series(pc,index=pd.to_datetime(date.iloc[:,0],format='%Y-%m'))
    #pcd_log=np.log(pcd)
    ##pcd_diff=pcd-pcd.shift()
    #pcd_diff.dropna(inplace=True)
    model=ARIMA(pcd,(1,0,0))
    model_fit=model.fit(disp=0)
    
    start_index='2020-08-10'
    end_index='2020-08-16'
    pred=model_fit.predict(start=start_index, end=end_index)
    '''predictions_ARIMA_diff = pd.Series(pred, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(pcd.ix[0])
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,
                                                      fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)'''
    pc_16.append(pred.iloc[6])
    
pc_16=np.asarray(pc_16)
pc_16=pc_16.reshape(len(pc_16),1)
pc_16=mms.fit_transform(pc_16)
pc_16=pc_16.reshape(len(pc_16),)
#pc_16=pd.DataFrame(pc_16)


df_test3=df_test[:,:]

df_test3[:,11]=pc_16

y_pred2=reg.predict(df_test3)


StackingSubmission1 = pd.DataFrame({ 'Stock Index': indexes,
                            'Stock Price': y_pred1 })
    
StackingSubmission2 = pd.DataFrame({ 'Stock Index': indexes,
                            'Stock Price': y_pred2 })
    
StackingSubmission1.to_csv("Part1.csv",index=False)
StackingSubmission2.to_csv("Part2.csv",index=False)




































































