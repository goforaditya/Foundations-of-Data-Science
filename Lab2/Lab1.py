# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:38:39 2018

@author: Frndzzz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:50:40 2018

@author: Frndzzz
"""

#%%
import pandas as pd
c=[0,1,2,3]
df= pd.read_csv('iris.data.txt',usecols=c)
#%%
import numpy as np
from sklearn import preprocessing

matrix=df.values
#matrix

#data normalize
# mean of each column
matrix2=matrix.T
mean=np.mean(matrix2, axis=1)
# standred deviation of each column
sd=np.std(matrix2,axis=1)


for i in range(len(matrix2)):
    
    for j in range(149):
        #print(matrix2[i,j])
        matrix2[i,j]=(matrix2[i,j]-mean[i])/sd[i]
        j +=1
    i +=1
print(matrix2)    
#data_scaled=preprocessing.scale(matrix)
 #%%
import matplotlib.pyplot as plt
matrix3=matrix2.T 
matarix3_2=np.asarray(matrix3) # we ahave to chage it into array otherwise ploting will not happen ..we can plot a matrix 

#covariance=np.cov(matarix3_2,rowvar=False)
#eigen_value_vector=np.linalg.eig(covariance)
w,v=np.linalg.eig(np.cov(matarix3_2, rowvar =False))
#G = np.array()
#for i in range(len(v)):
#    pca_m = v[:,i]
#    np.append(np.dot(matarix3_2,pca_m)
       
pca_main_axis=v[:,0]
pca_main_axis1=v[:,1]
pca_main_axis2=v[:,2]
pca_main_axis3=v[:,3]
PC1=np.dot(matarix3_2,pca_main_axis)
PC2=np.dot(matarix3_2,pca_main_axis1)
PC3=np.dot(matarix3_2,pca_main_axis2)
PC4=np.dot(matarix3_2,pca_main_axis3)

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
xx=PC1
yy=np.zeros((len(PC1)))
plt.figure(figsize=(800/my_dpi,800/my_dpi),dpi=my_dpi)
plt.scatter(xx,yy,color='b',s=20)
plt.xlabel('X')
#plt.ylabel('Y')
plt.title('projected_data')
plt.savefig('pca_2.png')

#PLOTTING ERROR


#%%
print(PC1)
print(matrix2)
#%% for pc1
G1=np.asmatrix(PC1)
Gtrans1=G1.T
a1=np.matmul(G1,matrix3)
X1=np.dot(Gtrans1,a1)
#%% for pc2
G2=np.asmatrix(PC2)
Gtrans2=G2.T
a2=np.matmul(G2,matrix3)
X2=np.dot(Gtrans2,a2)
#%%for pc3
G3=np.asmatrix(PC3)
Gtrans3=G3.T
a3=np.matmul(G3,matrix3)
X3=np.dot(Gtrans3,a3)

#%%for pc4
G4=np.asmatrix(PC4)
Gtrans4=G4.T
a4=np.matmul(G4,matrix3)
X4=np.dot(Gtrans4,a4)

#%%
from sklearn.metrics import mean_absolute_error
#norm=np.absolute(matrix33-matrix3)
error1=mean_absolute_error(matrix3,X1)
print(error1)
error2=mean_absolute_error(X2,matrix3)
error3=mean_absolute_error(X3,matrix3)
error4=mean_absolute_error(X4,matrix3)

Y=np.array([error1,error2,error3,error4])
print(w)
print(Y)
plt.plot(w,Y)
 #%%


from numpy import linalg as LA
error1_1=LA.norm(X1-matrix3)
error1_2=LA.norm(X2-matrix3)
#%%
G=np.asmatrix(matrix.T[0])
Gtrans=G.T
aa=np.matmul(G,PC1)
Xbar=np.matmul(Gtrans,aa)

err1=np.sum(np.square(Xbar-PC1))
#%%
Gg=np.asmatrix(matrix.T[1])
Gtransg=G.T
aa1=np.matmul(Gg,PC2)
Xbar1=np.matmul(Gtransg,aa1)

err2=np.sum(np.square(Xbar1-PC2))
#%%

#step 4 Choose the number of new dimensions and select the first eigenvectors
# We reduce dimension to k dimension by taking k Eigen value
#so we choose the most variance eigen value 
eig_vals[0] / sum(eig_vals)



#step 5 Project the normalized data points onto the first k eigenvectors.

projected_X = mats.dot(eig_vecs.T[0])
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
