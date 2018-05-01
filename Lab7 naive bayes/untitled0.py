# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 23:46:20 2018

@author: Frndzzz
"""

#%%
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob

df = pd.read_csv('iris.data.txt')
splitratio = np.random.rand(len(df)) < 0.8

training = df[splitratio]
outcome = training.iloc[:,-1]
training = training.iloc[:,:-1]

new_sample = df[~splitratio].iloc[:,:-1]

#training   = np.asarray(((1.0,0,1,1),(1,1,0,0),(1,0,2,1),(0,1,1,1),(0,0,0,0),(0,1,2,1),(0,1,2,0),(1,1,1,1)));
#outcome    = np.asarray((0,1,1,1,0,1,0,1))
#new_sample = np.asarray((1,0,1,0))

outcome = df.iloc[:,-1]

classes     = np.unique(outcome)
rows, cols  = np.shape(training)
likelihoods = {}
for cls in classes:
    likelihoods[cls] = defaultdict(list)
 
class_probabilities = occurrences(outcome)
 
for cls in classes:
    row_indices = np.where(outcome == cls)[0]
    subset      = training[row_indices, :]
    r, c        = np.shape(subset)
    for j in range(0,c):
        likelihoods[cls][j] += list(subset[:,j])
 
for cls in classes:
    for j in range(0,cols):
         likelihoods[cls][j] = occurrences(likelihoods[cls][j])
 
 
results = {}
for cls in classes:
     class_probability = class_probabilities[cls]
     for i in range(0,len(new_sample)):
         for j in range(0,len(new_sample[i])):    
             relative_values = likelihoods[cls][i]
             if new_sample[i][j] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             else:
                 class_probability *= 0
             results[i][cls] = class_probability
print(results)
#%%