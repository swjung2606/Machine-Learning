# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 09:24:23 2022

@author: swjun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import SoftmaxRegression





'''
Multinomial Logistic Regression 개념설명 사이트.
http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/
'''    

##### 1. Data preparation
# 3 featurs
h_1, h_2, h_3 = np.random.normal(149, 6, 1500) , np.random.normal(162, 5, 1500) , np.random.normal(170, 5, 1500)
w_1, w_2, w_3 = np.random.normal(48, 4, 1500) , np.random.normal(56, 5, 1500), np.random.normal(63, 5, 1500)
iq_1, iq_2, iq_3 = np.random.normal(90, 6, 1500), np.random.normal(107, 5, 1500), np.random.normal(98, 6, 1500)

h_col, w_col, iq_col = np.hstack((h_1,h_2,h_3)) , np.hstack((w_1,w_2,w_3)) , np.hstack((iq_1,iq_2,iq_3))

# class
y0, y1, y2 = np.zeros(1500,) , np.ones(1500,) , np.ones(1500,)*2
y = np.hstack((y0, y1, y2))

pd_dict = {'Height' : h_col , 'Weight' : w_col , 'IQ' : iq_col , 'Class' : y}

df = pd.DataFrame(pd_dict)



##### 2. Normalization 
def normalizer(a):
    n = len(a.columns)
    for i in range(0,n-1):
        a[a.columns[i]] = (a[a.columns[i]] - a[a.columns[i]].mean()) / a[a.columns[i]].std()
    

normalizer(df)
X = df[['Height', 'Weight', 'IQ']].values
y = df['Class'].values
# X : (n_samples x n_features)
W = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
bias = np.array([0.01, 0.01, 0.01])
# W : (n_features x n_classes)



##### 3. Find Z = bias + XW = bias + (w_1)(x_1) + (w_2)(x_2) + (w_3)(x_3)
Z = X@W + bias



##### 4. Softmax
def softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

# This smax matrix indicates the probability of each sample belongs to which class.
# For example, if the first row is [0.427, 0.32, 0.248], the first sample
# is likely to be part of class 0.
smax = softmax(Z)

# based on smax matrix, let's classify each sample's class
def to_classlabel(z):
    return z.argmax(axis=1)

classified_class = to_classlabel(smax)


##### 5. Target encoding 
# Define y_enc for computing cost function
def train_class_encoder(y):
    y_enc=[]
    for i in range(len(y)):
        if y[i]==0:
            y_enc.append([1,0,0])
        elif y[i]==1:
            y_enc.append([0,1,0])
        elif y[i]==2:
            y_enc.append([0,0,1])
            
    return y_enc

# For 0 class, [1,0,0], for 1 class, [0,1,0], for 2 class, [0,0,1]
y_enc = train_class_encoder(y)



##### 6.Compute cost function J
# Cost Function J = mean(H(T,O))
# H(T,O) = -sum(T * log(O))
# T : true class labels, O : smax output
# example. Let's say first row of smax is [0.5, 0.3, 0.2]
# H(1) = -{  [ln(0.5), ln(0.3), ln(0.2)]  @  [1,0,0]} = -ln(0.5)*1
# xent is the vector [H(1), H(2), ..., H(4500)]
def cross_entropy(output, y_target):
    return - np.sum(np.log(output) * (y_target), axis=1)

xent = cross_entropy(smax, y_enc)
J_cost = np.mean(xent)




### Use SoftmaxRegression in order to find the minimum cost function & training our softmax model.

y = y.astype('int32')

lr = SoftmaxRegression(eta=0.01, epochs=20)
lr.fit(X,y)
'''
plot_decision_regions(X, y, clf=lr, )
plt.title('Softmax Regression - Gradient Descent')
plt.show()
'''

plt.plot(range(len(lr.cost_)), lr.cost_)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()