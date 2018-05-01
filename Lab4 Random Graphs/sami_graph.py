# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 02:09:38 2018

@author: Frndzzz
"""

#%%
#showing graph transitions
import networkx as nx
import math
import matplotlib.pyplot as plt

n=150

pc = 1/n
pmax = math.log(n)/float(n)

pvals = [d*pc for d in range(8)]
region = 440
for pv in pvals:
    
    g=nx.fast_gnp_random_graph(n,pv)
    #h=max(nx.connected_component_subgraphs(g),key=len)

    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.1,hspace=0.4)

    region+=1
    plt.subplot(region)
    plt.title("P = %6.3f"%(pv))
    nx.draw(g,node_size = 15,node_color='r')
    
    #region+=1
    #plt.subplot(region)
    #nx.draw(h,node_size = 100,node_color='b')

    #print("Total Nodes: ",nx.number_of_nodes(g))
    #print("Giant Nodes: ",nx.number_of_nodes(h))
#%%
#%%
def  transisting_graph(n):
     import matplotlib.pyplot as plt
     import networkx as nx
     
     d=1
     step_size=d/10
     pvals=[]
     for i in range(5):
          p=d/n
          pvals.append(p)
          d=d-step_size
     region=330
     for p in reversed(pvals):
         G=nx.fast_gnp_random_graph(n,p)
         region+=1
         plt.subplot(region)
         plt.title("p = %6.3f"%(p)) 
         nx.draw(G,with_labels=False,node_size=10)
     e=1
     step_size=e/10
     qvals=[] 
     for i in range(4):  
         q=e/n
         e+=step_size
         qvals.append(q) 
     for p in qvals:
         G=nx.fast_gnp_random_graph(n,p)
         region+=1
         plt.subplot(region)
         plt.title("p = %6.3f"%(p))
         nx.draw(G,with_labels=False,node_size=10 ) 
           
                 
transisting_graph(50)                
#%%
def giant_component(n):
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    import math
    graph=nx.fast_gnp_random_graph(n,0.033)#math.log(n)/n
    plt.figure(1)
    plt.subplot(211)
    plt.title('Random Graph')
    nx.draw_random(graph,with_labels=True)
    subgraphs=list(nx.connected_component_subgraphs(graph))
    number_of_node=[]
    for i in subgraphs:
         number_of_node.append(nx.number_of_nodes(i))
    t=np.argsort(number_of_node)[::-1]
    giant_component=nx.number_of_nodes(subgraphs[t[0]])
    print('------ ----- ------ ----- -------','\n\n')
    print('Number of nodes in Giant Component =',giant_component,'\n')
    plt.figure(2)
    plt.subplot(212)
    plt.title('Graph of Giant Component')
    nx.draw(subgraphs[t[0]],node_color='blue',node_size=500,alpha=0.8,edge_color='r',with_labels=True)
    isolates=[]
    for i in subgraphs:
        if nx.number_of_nodes(i)==1:
            isolates.append(i)
    print('Total number of Isolates =',len(isolates),'\n') 
    print('------ ----- ------ ----- -------','\n\n')        



giant_component(30)    
          
            
            
          
        
             