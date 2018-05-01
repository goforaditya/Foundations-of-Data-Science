# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:24:58 2018

@author: me
"""
#%%
import numpy as np
import pandas as pd
f=pd.read_csv("tennis.csv")
#a=np.uni(f)
#b=list(f)
d=f.iloc[:,-1]
e=np.unique(d)

'''
data=[]
for line in e:
    line=[element for element in line.rstrip('\n').split(',')]
    data.append(np.asarray(line))
'''
split=np.random.rand(len(f)) < 0.7
train = f[split].iloc[:,:-1]
output=f[split].iloc[:,-1]
n = len(train)
outs = np.unique(output)

probs = {}
for o in outs:
    probs[o] = len(output[output==o])/len(output)

posprobs = {}

features = list(train) #Gives all column names or features
for feature in features:
    fprobs = {}
    for val in np.unique(train[feature]):
        prob = {}
        for o in outs:
            #print(val,output)
            #prob[o]=1
            indx = (train[feature]==val) & (output==o)
            prob[o]=(len(train[indx]))/(len(output[output==o]))
        fprobs[val]=prob
    posprobs[feature]=fprobs


# Calculating proabilities for test data
test = f[~split]
tdata = test.iloc[:,:-1]
tdata = tdata.reset_index(drop = True)
m = len(test)
prob = {}

for i in range(len(tdata)):
    a = tdata.iloc[i,:]
    probout = {}
    for o in outs:
        p = 1
        for feature in features:
            aval = tdata.loc[i,feature]
            if aval in posprobs[feature].keys():
                p*=posprobs[feature][aval][o]
            else:
                # Handling attribute value not in test data
                p=1/(n + 2)
        probout[o] = p
    prob[i] = probout

# Predictions and errors
predictions = [] 
given = test.iloc[:,-1] 

for a in prob:
    inverse = [(value, key) for key, value in prob[a].items()]
    predictions.append(max(inverse)[1])


equality = (predictions == given)
s=pd.DataFrame()
s['Actual'] = given
s['Predicted'] = predictions
s['Equality'] = equality
print(s)
print("Correctness: ",((len(equality[equality == True]))/float(len(equality)))*100)
#%%
    