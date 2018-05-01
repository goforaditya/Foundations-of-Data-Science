# -*- coding: utf-8 -*-
#%%
import pandas as pd
c=[0,1,2,3]
df= pd.read_csv('iris.data.txt',usecols=c,header=None)
#%%
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

matrix=df.values
#matrix


#data normalize
# mean of each column
matrix2=matrix.T
mean=np.mean(matrix2, axis=1)
#from mpl_toolkits.mplot3d import Axes3D as ax
#ax.scatter(df[[0,]], df[[1,]], df[[2,]], c=df[[3,]], cmap=plt.hot()) 
plt.show() 
# standred deviation of each column
sd=np.std(matrix2,axis=1)


for i in range(len(matrix2)):
    
    for j in range(150):
        #print(matrix2[i,j])
        matrix2[i,j]=(matrix2[i,j]-mean[i])/sd[i]
        j +=1
    i +=1
print(matrix2)    
#data_scaled=preprocessing.scale(matrix)
 #%%
import matplotlib.pyplot as plt
matrix3=matrix2.T
matarix3_2=np.asarray(matrix3) # we ahave to change it into array otherwise ploting will not happen ..we can plot a matrix 

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
#PLOTTING ERROR
for i in range(len(v)):
    pca_axis=v[:,i]
    c0=0
    c1=pca_axis[1]/pca_axis[0]
    yf=lambda x: c0+c1*x
    pca_y=list(map(yf,matarix3_2[:,0]))
    print(pca_y)
    #projected_data=np.dot(matarix3_2,pca_axis)
    a = sum((matrix2-pca_y)**2)



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


