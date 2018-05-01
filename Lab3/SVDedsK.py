# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:44:04 2018

@author: Frndzzz
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
dataset=pd.read_csv('cancer.data.txt')
#dataset=pd.read_csv('glass.data.txt')
A=np.matrix(dataset._get_numeric_data())
A=preprocessing.normalize(A)
ATA = np.matmul(A.T,A) 
AAT = np.matmul(A,A.T)
eigvalsATA, eigvecsATA = np.linalg.eig(ATA)
eigvalsAAT, eigvecsAAT = np.linalg.eig(AAT)
eigvalsATA[::-1].sort()
eigvalsAAT[::-1].sort()


np.real(eigvalsAAT)
n=len(A)
#u,s,vh = np.linalg.svd(A)
k=len(A.T)
VT=np.asmatrix(eigvecsATA[:,0:k])
U=np.asmatrix(eigvecsAAT)
Norms = []
#Norms1 = []

for i in range(k):
    #SIG=np.diagflat(eigvalsAAT[0:i])
    SIG=np.zeros((n,k))
    for j in range(i-1):
        SIG[j,j] =  np.sqrt(eigvalsAAT[j]) 
    SIGM = np.asmatrix(SIG)
    SIGM
    APPROX = np.matmul(U,np.matmul(SIG,VT))
    Norms.append(np.linalg.norm(A-APPROX))
print("\n\nDIMENSIONS: \nU : ",U.shape,"\n VT : ",VT.shape,"\n SIG : ",SIG.shape)
x = [i for i in range(1,k+1)]
plt.plot(x,Norms)
plt.ylabel('Errors')
print("\n ERROR : ",Norms)
#%%