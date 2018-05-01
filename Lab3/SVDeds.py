# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:27:29 2018

@author: Frndzzz
"""

#%%

import numpy as np
import pandas as pd
dataset=pd.read_csv('iris.data.txt')
#df1=pd.read_csv('cancer.data.txt')
#dataset=pd.read_csv('glass.data.txt')
A=np.matrix(dataset._get_numeric_data())
ATA = np.matmul(A.T,A) 
AAT = np.matmul(A,A.T)
eigvalsATA, eigvecsATA = np.linalg.eig(ATA)
eigvalsAAT, eigvecsAAT = np.linalg.eig(AAT)
eigvalsATA[::-1].sort()
eigvalsAAT[::-1].sort()

np.real(eigvalsAAT)
n=len(A)
k=len(A.T)
SIG=np.zeros((n,n))

for i in range(n):
    SIG[i,i]=np.sqrt(eigvalsAAT[i])
SIGA = SIG[:,0:len(eigvecsATA)]
SIGM = np.asmatrix(SIGA)
SIGM
VT=np.asmatrix(eigvecsATA)
U=np.asmatrix(eigvecsAAT)
APPROX = np.matmul(U,np.matmul(SIGA,VT))
print("\n\nDIMENSIONS: \nU : ",U.shape,"\n VT : ",VT.shape,"\n SIG : ",SIGA.shape)
Norm = np.linalg.norm(A-APPROX)
print("\n ERROR : ",Norm)
#%%

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

eig=np.linalg.eig
norm=np.linalg.norm

df=pd.read_csv("iris.data.txt",sep=',',header=None)
print(df.head())
df=df._get_numeric_data()
print(df.head())
print(df.shape)
#%%
dmat=np.matrix(df)
dmatt=dmat.T
u1=np.matmul(dmat,dmatt)
v1=np.matmul(dmatt,dmat)
ueva,uevec=eig(u1)
veva,vevec=eig(v1)
#%%
m=min(u1.shape)
n=min(v1.shape)
j=ueva.argsort()[::-1]
ue_v=ueva[j]
uev=uevec[:,j]
i=veva.argsort()[::-1]
ve_v=veva[i]
vev=vevec[:,i]
d=np.diag(ve_v)
if m>n:
    l=m-n
    k=n
else:
    l=n-m
    k=m
z=np.zeros((l,k))
eta=np.concatenate((d,z))
u=np.matrix(uev)
v=np.matrix(vev)
dmatf=np.matmul(np.matmul(u,eta),v)
err=mae(dmat,dmatf)
print(err)
#%%
#direct command do not use
svd=np.linalg.svd
U,E,V=svd(df)
print(U)
print(E)
print(V)

#%%