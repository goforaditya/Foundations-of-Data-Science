# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:46:22 2018

@author: Samiksha Agarwal
"""

#%%
import pandas as pd
df= pd.read_csv('data.csv')
#%%
import numpy as np
from sklearn import preprocessing

matrix=np.matlib.rand(20,2)
#matrix

#data normalize
# mean of each column
matrix2=matrix.T
mean=np.mean(matrix2, axis=1)
# standred deviation of each column
sd=np.std(matrix2,axis=1)


for i in range(len(matrix2)):
    
    for j in range(20):
        #print(matrix2[i,j])
        matrix2[i,j]=(matrix2[i,j]-mean[i,0])/sd[i,0]
        j +=1
    i +=1
print(matrix2)    
data_scaled=preprocessing.scale(matrix)
 #%%
import matplotlib.pyplot as plt
matrix3=matrix2.T 
matarix3_2=np.asarray(matrix3) # we ahave to chage it into array otherwise ploting will not happen ..we can plot a matrix 

covariance=np.cov(matarix3_2,rowvar=False)
eigen_value_vector=np.linalg.eig(covariance)
w,v=np.linalg.eig(np.cov(matarix3_2, rowvar =False))
pca_main_axis=v[:,0]
projected_data=np.dot(matarix3_2,pca_main_axis)

#plot original data and pca line
x=matarix3_2[0]

c0=0
c1=pca_main_axis[1]/pca_main_axis[0]
yf=lambda x: c0+c1*x
pca_y=list(map(yf,matarix3_2[:,0]))
#%%
my_dpi=96
plt.figure(figsize=(800/my_dpi,800/my_dpi),dpi=my_dpi)

plt.scatter(matarix3_2[:,0],matarix3_2[:,1],color='b',s=20)
plt.plot(matarix3_2[:,0],pca_y,color='r',linewidth=3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('original variable x and y')
plt.savefig('pca_1.png')

#%%
#PLOT NEW VARIABLE
xx=projected_data
yy=np.zeros((len(projected_data)))
plt.figure(figsize=(800/my_dpi,800/my_dpi),dpi=my_dpi)
plt.scatter(xx,yy,color='b',s=20)
plt.xlabel('X')
#plt.ylabel('Y')
plt.title('projected_data')
plt.savefig('pca_2.png')


#%%
x=matrix2[0]
c0=0
c1=pca_main_axis[1]/pca_main_axis[0]
yf=lambda x: c0+c1*x
pca_y=list(map(yf,data_scaled[:,0]))

my_dpi=96
plt.figure(figsize=(800/my_dpi,800/my_dpi),dpi=my_dpi)

plt.scatter(data_scaled[:,0],data_scaled[:,1],color='b',s=20)

#%%



