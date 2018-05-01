# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:49:17 2018

@author: Frndzzz
"""

#%%
#%%
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import sys

def occurrences(list1):
    '''
    ================================================
    
    Given all the different unique classs in a list
    
    Find out probability of each unique class given the list
    
    Input: List
    Output: Dictionary {'<Class_Name>':<Probability_Value>...(for all n distinct classes)}
    ================================================
    '''
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / (float(no_of_examples) + 2)
    return prob

def naive_bayes_classifier_on_file(filename,split_ratio):
    '''
    ================================================
    
    Given a table split the 
    
    Find out probability of each unique class given the list
    
    Input: List
    Output: Dictionary {'<Class_Name>':<Probability_Value>...(for all n distinct classes)}
    ================================================
    '''
    #df = pd.read_csv('iris.data.txt')
    df = pd.read_csv(filename) # With Laplace Correction Works Well on Large Datasets
    
    n=len(df)
    splitat = np.random.rand(len(df)) < split_ratio
    
    #sp = int(n*0.7)
    training = df[splitat]
    #training = df.iloc[:sp]
    outcome = np.asarray(training.iloc[:,-1])
    training = np.asarray(training.iloc[:,:-1])
    
    
    
    samples = df[~splitat]
    #samples = df.iloc[sp:]
    original = np.asarray(samples.iloc[:,-1])
    new_samples = np.asarray(samples.iloc[:,:-1])
    
    classes = np.unique(outcome)
    rows, cols = np.shape(training)
    likelihoods = {}
     
    class_probabilities = occurrences(outcome)
        
    for cls in classes:
        # Initiallize the likelihood for current class 
        likelihoods[cls] = defaultdict(list)
        # indices of all the rows which have the current class value 
        row_indices = np.where(outcome == cls)[0]
        # Take only the rows with current class values
        subset = training[row_indices, :]
        #r, c = np.shape(subset)
        
        # Calculate Likelihoods for each column value
        for j in range(0,cols):
            likelihoods[cls][j] += list(subset[:,j])
     
    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])
     
     
    results = []
    
    for new_sample in new_samples:
        result = {}
        for cls in classes:
             class_probability = class_probabilities[cls]
             for i in range(0,len(new_sample)):
                 relative_values = likelihoods[cls][i]
    
                 if new_sample[i] in relative_values.keys():
                     class_probability *= relative_values[new_sample[i]]
                 else:
                     # Correcting for unseen before value in new_sample[< Attribute i >]
                     class_probability *= 1/float(len(relative_values) + 2)
                 result[cls] = class_probability
        results.append(result)                 
    #print(results)
    
    predictions = []
    for r in results:
        inverse = [(value, key) for key, value in r.items()]
        classi = max(inverse)[1]
        predictions.append(classi)
    
    predictions = np.asarray(predictions)
    
    ss = (predictions==original)
    
    accuracy = (len(ss[ss==True])/len(ss))*100.0
    
    d = pd.DataFrame()
    d['Original'] = original
    d['Predicted'] = predictions
    d['Correct'] = ss
    print("\n\n === CLASSIFICATION ===\n",d)
    print("\n === ACCURACY ===\n",accuracy)
    
    return d

if __name__=='__main__':
    
    split_ratio = 0.90 # Deafult Splitratio 90%
    name_of_file = 'car.data.txt' # Hardwire File name
    
    if len(sys.argv)>1:
        name_of_file = sys.argv[1] # Pass filename as command line argument
        print("Applying Naive Bayes on",name_of_file)    
        if len(sys.argv)>2:
            split_ratio = sys.argv[2] # Pass splitratio as command line argument default 90
    
    naive_bayes_classifier_on_file(name_of_file,float(split_ratio))

