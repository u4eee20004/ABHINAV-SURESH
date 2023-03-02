#!/usr/bin/env python
# coding: utf-8

# # ABHINAV SURESH - PROGRAM 1 - EEE - 20004

# In[ ]:


import numpy as np # it is library used in python 
import pandas as pd # it is another kind of library 
import matplotlib.pyplot as plt # to import the graph from the library


# In[2]:


def ComputeCost(X,y,theta): #Python function is defines as keyword "def"
    n=len(y) # to find the lenght 
    h=X.dot(theta) # formula to find the h
    square_err=(h-y)**2 # finding the square error
    return 1/(2*m)*np.sum(square_err)#Returning the values
def gradientDescent (X,y,theta,alpha,num_iters):#Another function is defined
    n=len(y)
    J_history=[]
    for i in  range(num_iters):# 'for' is used for iteration over the keywords
        h=X.dot(theta)
        error=np.dot(X.transpose(),(h-y))#for finding out the error we use the formula
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(ComputeCost(X,y,theta))# append is used for adding elements 
    return theta,J_history # return is used for values of the functions given above


# In[3]:


data=pd.read_csv('ex1data1.txt',header=None) # to entering the Data sheet to the program 
data_n=data.values
m=len(data_n[:,0])
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)#elements that asto includfe in the graph
y=data_n[:,1].reshape(m,1) # for reshaping the data given
theta=np.zeros((2,1))
ComputeCost(X,y,theta)
theta,J_history=gradientDescent(X,y,theta,0.01,1500)
theta[0,0]
plt.scatter(data[0],data[1])
plt.plot(X,theta[1,0]*X+theta[0,0])#which are the data are showing in the graph
plt.show # for showing the graph we this given function


# In[ ]:




