#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor


# In[2]:


import matplotlib.pyplot as plt
from xgboost import plot_importance


# In[3]:


import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from sklearn import utils
import warnings
warnings.filterwarnings('ignore')


# In[4]:


#toy example of data engineering function
def engineerData(df, engineeringParam1='defaultValue', engineeringParam2='anotherValue', isTrain=True, hasY=True):
    #...
    #do some processing of df in here.
    #as this is a toy e.g., I just return data without any changes
    newEngineeredData = df
    #...
    if hasY:
        return newEngineeredData.iloc[:,:-1], newEngineeredData.iloc[:,-1]
    else:
        return newEngineeredData


# ## Read-in date and take a look

# In[5]:


data = pd.read_csv('./export.csv')


# In[6]:


data.head()


# In[7]:


plt.plot(np.arange(data.shape[0]), data.iloc[:,-1])
plt.show()


# We remark that target possede certain pattern, so we want to re-shuffle the input.

# ### Reshuffle

# In[8]:


data = data.sample(frac = 1).reset_index(drop = True)
plt.plot(np.arange(data.shape[0]), data.iloc[:,-1])
plt.show()


# ## Drop unamed column
# We remark that  XGBoost take user id as information, which is apparently overfitting.

# In[9]:


data = data.drop('Unnamed: 0', 1)


# ## Grid search for xgboost parameters

# In[10]:


#model parameters
# param = { 'objective':'mse','max_depth':7,'learning_rate':.01,'max_bin':250, 'seed':15, 'verbose': -1}
# param['metric'] = ['mean_squared_error']

# a parameter grid for xgboost
param = { #'objective':'mse',
         'max_depth':[7,10,4],
         'learning_rate':[.01, .005, .1],
         'max_bin':[250], 
         'max_depth':[3,5,7],
         'gamma':[0,0.5]
         #'seed':15, 
         #'verbose': -1
        }


# In[11]:


#training loop
samplesize = 5000
trainValidSplit = int(0.75*samplesize)
# roundPerBatch = 50


# In[12]:


engineerData(data.sample(samplesize).iloc[:trainValidSplit,:])[1]


# In[13]:


# Take a sample for  hyperparameter tuning
sampleX,sampleY = engineerData(data.sample(samplesize).iloc[:trainValidSplit,:])


# In[14]:


# sampleX = sampleX.drop('ContractFeature_Schedule,EndDate_ENCODED',1)
sampleX.head()


# In[15]:


sampleY


# In[16]:


# apply gridSearch to this sample
xgb = XGBRegressor(nthread = -1)
grid = GridSearchCV(xgb, param)
grid.fit(sampleX, sampleY)


# In[17]:


grid.best_estimator_


# In[18]:


# best_xgb = XGBRegressor(nthread = -1, learning_rate = 0.1, max_depth = 3)
best_xgb = grid.best_estimator_


# In[19]:


grid.best_estimator_


# In[20]:


# best_xgb = XGBRegressor(nthread = -1, learning_rate = 0.1, max_depth = 3, gamma = 0, max_bin = 300)


# In[21]:


#for each batch, we'll train on 75% of data and validate with the rest.
trainValidIndex = int(0.75*len(data))
#engineer features. Here, our toy function
trainX, trainY = engineerData(data.iloc[:trainValidIndex,:])


# In[22]:


trainY


# In[23]:


validX, validY = engineerData(data.iloc[trainValidIndex:,:], isTrain=False)
#build training and validation gbm dataset objects
# fitted = best_xgb.fit(trainX.drop('ContractFeature_Schedule,EndDate_ENCODED',1), trainY)
fitted = best_xgb.fit(trainX, trainY)

# dtrainX = best_xgb.DMatrix(data = trainX.values, feature_names = trainX.columns, label = trainY.values)


# In[24]:


predY = fitted.predict(validX)
# predY = fitted.predict(validX.drop('ContractFeature_Schedule,EndDate_ENCODED',1))


# In[25]:


np.sum((predY-validY)**2/len(validY))


# In[26]:


best_xgb.evals_result


# In[27]:


fitted.feature_importances_


# In[28]:


# plot
plt.bar(range(len(best_xgb.feature_importances_)), best_xgb.feature_importances_)
plt.show()


# In[29]:


# plot feature importance
plot_importance(best_xgb)
plt.show()


# In[30]:


plt.figure()
plt.plot(np.arange(len(validX)), validY, alpha = 0.7, label = 'real-value')
plt.plot(np.arange(len(validX)), predY, alpha = 0.7, label = 'pred-value')
plt.legend()
plt.show()


# ## Plot intepretation
# 

# In[40]:


# # requires graphviz and python-graphviz conda packages
import graphviz
import xgboost

xgboost.plot_importance(best_xgb)

# plot the output tree via matplotlib, specifying the ordinal number of the target tree
# xgb.plot_tree(xgb_model, num_trees=xgb_model.best_iteration)

# converts the target tree to a graphviz instance
# xgboost.to_graphviz(best_xgb, num_trees=best_xgb.best_iteration) // best_iteration ?


# * https://www.kaggle.com/zj0512/using-xgboost-with-scikit-learn
# example at the end of this link

# In[ ]:




