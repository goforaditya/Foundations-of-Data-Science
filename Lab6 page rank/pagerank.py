# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:22:06 2018

@author: Aditya Singh Rathore
"""
#%%
def random_bnp_pagerank(n=5,p=0.06):
    """
        n: Number of nodes, p: Probability value
    """
    import numpy as np
    import networkx as nx
    #import math
    import random

    #pc = 1/(n-1) 
    #pf = math.log(n)/float(n)
    g = nx.fast_gnp_random_graph(n,p)
    #g1 = nx.DiGraph(g)
    
    #Random Weights ()
    #for (u,v) in g.edges():
    #    g[u][v]['weight'] = random.randint(0,10)
    
    A = nx.adjacency_matrix(g)
    Adjacency = A.todense()
    print(Adjacency)
    
    Transition=np.zeros((n,n))
    for i in range(n):    
        for j in range(n):
            if(np.sum(Adjacency[i])<=0): # to handle isolated nodes and give rank 0 to them
                Transition[i,j]=0
            else:
                Transition[i,j]=(Adjacency[i,j])/np.sum(Adjacency[i])
    
    # Initial proabilities assuming Uniformity
    P0 = np.repeat(1/float(n),n)
    
    Pi = np.matmul(P0,Transition)
    eps = 0.001
    
    #for i in range(50):
    while np.sum(abs(Pi - P0))>eps:
        P0 = Pi
        print(P0)
        Pi = np.matmul(P0,Transition)
        
    print("Page Ranks Of All %d Pages"%(n))
        
    for i in range(n):
        print("\nPage %d Rank: %6.3f"%(i+1,Pi[i]))
    # For Outputting
    nx.draw(g)

def main():

    n = 15 # Take 100,150,10000000
    p = 0.55 # Take 0.005,0.33,0.007
    random_bnp_pagerank(n,p)


if __name__=='__main__':
    main()
#%%
import numpy as np
#Number of pages
n=3
# Yahoo, Amazon, Microsoft
P0=np.repeat(1/float(n),3)
#P0 = np.array([1/float(n),1/3.0,1/3.0])
Transition = np.array([[0.5,0.5,0.0],
                       [0.5,0.0,0.5],
                       [0.0,0.0,1.0]])
Pi = np.matmul(P0,Transition)
eps = 0.001

#for i in range(50):
while np.sum(abs(Pi - P0))>eps:
    P0 = Pi
    print(P0)
    Pi = np.matmul(P0,Transition)
print("Page Ranks\nYahoo : %6.3f"%(Pi[0]))
print("\nAmazon : %6.3f"%(Pi[1]))
print("\nMicrosoft : %6.3f"%(Pi[2]))

#%%
import numpy as np
# Yahoo, Amazon, Microsoft
P0 = np.array([1/3.0,1/3.0,1/3.0])
Transition = np.array([[0.5,0.5,0.0],
                       [0.5,0.0,0.5],
                       [0.0,1.0,0.0]])
Pi = np.matmul(P0,Transition)
eps = 0.00001

while np.sum(abs(Pi - P0))>eps:
    P0 = Pi
    Pi = np.matmul(P0,Transition)
print("Page Ranks\nYahoo :",Pi[0])
print("\nAmazon :",Pi[1])
print("\nMicrosoft :",Pi[2])
#%%

#%%

#IGNORE
a=[0.3333,0.3333,0.33333]
#T=np.matrix([[0.0,1.0,0.0],[0.5,0.0,0.5],[0.0,1.0,0.0]])
T = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,1.0,0.0]])

c=np.matmul(a,T)
print(c)
c1=np.matmul(c,T)
print(c1)
c2=np.matmul(c1,T)
print(c2)
c3=np.matmul(c2,T)
print(c3)
#%%
while np.sum(abs(c-a))>0.001:
    a=c
    c=np.matmul(a,T)
    print(c)

print(c)