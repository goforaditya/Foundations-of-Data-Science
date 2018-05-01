# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:05:42 2018

@author: Frndzzz
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

dataset=pd.read_csv('cancer.data.txt')
#dataset=pd.read_csv('glass.data.txt')
A=np.matrix(dataset._get_numeric_data())
A=preprocessing.normalize(A)
n = len(A)
ATA = np.matmul(A.T,A) 
AAT = np.matmul(A,A.T)
eigvalsATA, eigvecsATA = np.linalg.eig(ATA)
eigvalsAAT, eigvecsAAT = np.linalg.eig(AAT)

V = np.asmatrix(eigvecsATA)
SIG = np.diag(eigvalsATA)
U = np.array()
Norms = []
for i in range(len(SIG)):
    U[i]=(np.matmul(A,V[i])/np.sqrt(SIG[i,i]))
    APPROX = np.matmul(U,np.matmul(SIG,V.T))
    Norms.append(np.linalg.norm(A-APPROX))
    
plt.plot(Norms)    

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
SIG=np.zeros((n,k))
for j in range(k):
    SIG[j,j] =  np.sqrt(eigvalsAAT[j])

APPROX = U-np.matmul(A,VT.T)/SIG
np.linalg.norm(A-APPROX)    

#Norms1 = []
#%%
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

# NETWORK-X Explorarion

import networkx as nx
import matplotlib.pyplot as plt

g = nx.read_edgelist('facebook_combined.txt',create_using=nx.Graph(),nodetype=int)
print(nx.info(g))
sp = nx.spring_layout(g)
plt.axis('off')
nx.draw_networkx(g,pos=sp,with_labels=False,node_size=35)
#%%

import networkx as nx
import matplotlib.pyplot as plt
i=10
i=0.1
#i=3
g = nx.fast_gnp_random_graph(25,i/10)
nx.draw_circular(g)
#nx.draw_networkx(g)
#nx.draw(g)
#nx.draw_random(g)
#nx.draw_spectral(g)