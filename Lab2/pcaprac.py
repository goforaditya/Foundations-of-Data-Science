# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:51:08 2018

@author: Subhajeet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("student-mat.csv",sep=';')
print(df.head())

#correlation_matrix(df)

d=df.select_dtypes(include=[np.number])

mean=np.mean(d)

std=np.std(d)

plt.plot(d)

dnorm=(d/std)-mean

print(dnorm.head())

print(np.cov(dnorm))

plt.plot(dnorm)
#%%
#correlation_matrix(dnorm)

i=dnorm.corr()

j=np.transpose(i)

k=np.matmul(i,j)
print(k)
eig=np.linalg.eig

e_vals,e_vecs=eig(k)
print(e_vals,'/n',e_vecs)

plt.plot(e_vecs)
#%%
l=np.matrix(dnorm.T)
q=[]
for i in range(15,0,-1):
    k=np.matmul(l[i],l[i].T)
    p=e_vecs[i]
    r=np.square(p-k*p)
    q.append(np.sum(r))
plt.plot(q)
plt.show()