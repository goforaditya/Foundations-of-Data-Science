# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:24:35 2018

@author: Frndzzz
"""

#%%
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
g = nx.fast_gnp_random_graph(10,i/10)
#nx.draw_circular(g)
#nx.draw_networkx(g)
#nx.draw(g)
nx.draw_random(g)
#nx.draw_spectral(g)
#%%