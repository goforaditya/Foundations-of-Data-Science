# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import matplotlib.pyplot as plt
import networkx as nx
import math

layout=nx.spring_layout


n=200  # 150 nodes
# p value at which giant component (of size log(n) nodes) is expected
p_giant=1.0/(n-1)
# p value at which graph is expected to become completely connected
p_conn=math.log(n)/float(n)

# the following range of p values should be close to the threshold
pvals=[0.003, 0.006, 0.008, 0.03]

region=220 # for pylab 2x2 subplot layout
plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.1,hspace=0.1)
for p in pvals:
    G=nx.binomial_graph(n,p)
    pos=layout(G)
    region+=1
    plt.subplot(region)
    plt.title("p = %6.3f"%(p))
    nx.draw(G,pos,with_labels=False,node_size=10)
    # identify largest connected component
    Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    G0=Gcc[0]
    nx.draw_networkx_edges(G0,pos,with_labels=False,edge_color='r',width=0.5)
    # show other connected components
    for Gi in Gcc[1:]:
       if len(Gi)>1:
          nx.draw_networkx_edges(Gi,pos,with_labels=False,edge_color='r',alpha=0.3,width=5.0)
plt.show() # display
#%%

import numpy as np
import networkx as nx
layout=nx.spring_layout

n=50
p_critical = 1/(n-1)
p_max = math.log(n)/float(n)
l = [0.0,0.003,p_critical,p_max]
region=220
plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.1,hspace=0.1)
for p in l:
    g=nx.fast_gnp_random_graph(n,p)
    pos=layout(G)
    region+=1
    plt.subplot(region)
    plt.title(p)
    nx.draw(g,pos,node_size=15)
    all_components = sorted(nx.connected_component_subgraphs(g),key=len,reverse=True)
    giant = all_components[0]
    total_nodes = nx.number_of_nodes(g)
    giant_nodes = nx.number_of_nodes(giant)
    
    print("\nP =",p,"\nNumber of nodes in original graph :",total_nodes)
    print("Nodes in giant component =",giant_nodes)
    print('Number of isolates are :',nx.number_of_isolates(g))
#%%
#giant component
n=150

pc = 1/n
pmax = math.log(n)/float(n)

pvals = [pc,pmax]
region = 220
for pv in pvals:
    
    g=nx.fast_gnp_random_graph(n,pv)
    h=max(nx.connected_component_subgraphs(g),key=len)

    #plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.1,hspace=0.1)

    region+=1
    plt.subplot(region)
    plt.title("P = %6.3f"%(pv))
    nx.draw(g,node_size = 15,node_color='r')
    
    region+=1
    plt.subplot(region)
    nx.draw(h,node_size = 100,node_color='b')

    print("Total Nodes: ",nx.number_of_nodes(g))
    print("Giant Nodes: ",nx.number_of_nodes(h))
#%%
t=list(nx.connected_component_subgraphs(g))
k=[]
for i in t:
    k.append(nx.number_of_nodes(i))
i=np.argsort(k)[::-1]
nx.draw(t[i[0]])
g_n_e=nx.number_of_nodes(g)
h_n_e=nx.number_of_nodes(t[i[0]])
print("Number of nodes in original graph :",g_n_e,"\n")
print("number of nodes in giant component :",h_n_e)
print('number of isolates are :',nx.number_of_isolates(g))
#%%
import networkx as nx

g=nx.Graph()
g.add_nodes_from([1,2])
g.add_edge(1,2)
nx.draw(g)

g1=nx.fast_gnp_random_graph(10,0.003)
#%%
#showing graph transitions
import networkx as nx
import math
import matplotlib.pyplot as plt

n=150

pc = 1/n
pmax = math.log(n)/float(n)

pvals = [d*pc for d in range(8)]
region = 240
for pv in pvals:
    
    g=nx.fast_gnp_random_graph(n,pv)
    #h=max(nx.connected_component_subgraphs(g),key=len)

    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0.1,hspace=0.4)

    region+=1
    plt.subplot(region)
    plt.title("P = %6.3f"%(pv))
    nx.draw(g,node_size = 10,node_color='r')
    
    #region+=1
    #plt.subplot(region)
    #nx.draw(h,node_size = 100,node_color='b')

    #print("Total Nodes: ",nx.number_of_nodes(g))
    #print("Giant Nodes: ",nx.number_of_nodes(h))