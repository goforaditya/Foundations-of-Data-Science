# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:48:07 2018

@author: Aditya
"""

import numpy as np

def PCAErrors(dataset):
    original=np.matrix(dataset._get_numeric_data())
    transposed=original.T
    m=len(transposed)
    n=len(original)
    normalized=np.zeros((len(original),len(transposed)))
    
    
    for i in range (m):
        for j in range (n):
            normalized[j,i]=(original[j,i] - np.mean(original[:,i]))/np.std(original[:,i])
    
    normalized
    normT=normalized.T
    covariance_matrix = np.cov(normT)
    print(covariance_matrix)
    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    
    
    #error calculation
    #PCA
    toUse=eig_vecs
    toUse
    error = []
    for i in range(len(toUse)):
        GT=toUse[:,0:i+1]
        GT=np.asmatrix(GT)
        G=GT.T
        PC=np.matmul(G,normT)
        approx=np.matmul(GT,PC)
        
        N1 = np.linalg.norm(normT-approx)
        error.append(N1)
        
    import matplotlib.pyplot as plt
    x=range(1,len(transposed)+1)
    plt.plot(x,error,color='m')
    print("Eigenvectors \n",eig_vecs)
    print('\nEigenvalues \n', eig_vals)
    print('\nErrors \n', error)
    print('\nData Dimensions \n')
    print("Features: ",m,"\nEntries: ",n,"\nEigenvectors :",len(eig_vecs))
    
    
import pandas as pd
df1=pd.read_csv('iris.data.txt')
#df1=pd.read_csv('cancer.data.txt')
#df1=pd.read_csv('glass.data.txt')
PCAErrors(df1)
#%%