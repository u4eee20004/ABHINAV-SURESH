#!/usr/bin/env python
# coding: utf-8

# # ABHINAV SURESH - PROGRAM 2 - EEE-20004

# In[ ]:


import numpy as np # library used in python 
import pandas as pd # another kind of library 
import matplotlib.pyplot as plt # for plotting the graph from the library
from sklearn.linear_model import LinearRegression # for modeling and prediction of data we use linear regression


# In[3]:


data=pd.read_csv('ex1data1.txt',header=None)
plt.scatter(data[0],data[1]) 
data_n=data.values
m=len(data_n[:,0])
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)

regressor=LinearRegression()
regressor.fit(X,y)

theta0=regressor.intercept_
theta1=regressor.coef_

plt.plot(X,theta1*X+theta0)
plt.show()

