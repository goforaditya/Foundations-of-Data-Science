# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:48:07 2018

@author: Aditya
"""

import numpy  #its allow us to generate random number
import pandas as pd ##pandas is use for makes it easy to manipulate data in python
#read data set 


df=pd.read_csv('iris.data.txt')
a=[0,1,2,3]

mat=df[[0,1,2,3]]
mat
mat=numpy.matrix(mat)
mat

Y = df[[4]]
Y=numpy.matrix(Y)
Y



mat2=mat.T
mat2
matmean=numpy.zeros((149,4))
matstd=numpy.zeros((149,4))

#now start pca
#step 1 Normalize the initial variables(center the data and scale the data)

for i in range (len(mat2)):
    for j in range(len(mat)):
        matmean[j,i]=mat[j,i]-numpy.mean(mat[:,i])
           
matmean 

matmeanT=matmean.T

for i in range (len(matmeanT)):
    for j in range (len(matmean)):
        matstd[j,i]=matmean[j,i]/numpy.std(matmean[:,i])



matstd
matstdT=matstd.T


#Strp  2 Compute the covariance matrix of the normalized data


covariance_matrix = numpy.cov(matstdT)
print(covariance_matrix)



#step 3 Eigen Vectors and Eigen Values from Covariance Matrix

eig_vals, eig_vecs = numpy.linalg.eig(covariance_matrix)


#step 4 Choose the number of new dimensions and select the first eigenvectors
# We reduce dimension to k dimension by taking k Eigen value
#so we choose the most variance eigen value 
eig_vals[0] / sum(eig_vals)



#step 5 Project the normalized data points onto the first k eigenvectors.

projected_X = matstd.dot(eig_vecs.T[0])
projected_X


result = pd.DataFrame(projected_X, columns=['PC1'])
result
result['y-axis'] = 0.0
result
result['label'] = Y
result.head(10)


import seaborn as sns
import matplotlib.pyplot as plt
plt.grid(True)

sns.lmplot("PC1","y-axis",data=result,fit_reg=False,hue="label")

#%%
#error calculation
#PCA1

evalu=eig_vals
evalu

EV1=eig_vecs
EV1
error = []
for i in range(len(EV1)):
    G1=EV1[:,0:i+1]
   
    G1=numpy.asmatrix(G1)
    GT1=G1.T

    #gGt
    tmp=numpy.matmul(GT1,matstdT)
    PC1=numpy.matmul(G1,tmp)
    
    N1 = numpy.linalg.norm(matstdT-PC1)
    #N1=numpy.sum(numpy.square(numpy.absolute(matstdT-PC1)))
    error.append(N1)
    
    
print("Eigenvectors \n",eig_vecs)
print('\nEigenvalues \n', eig_vals)

print('\nErrors \n', error)

import matplotlib.pyplot as plt
x=range(1,5)
plt.scatter(x,error,color='k')
