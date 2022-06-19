# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:53:08 2022

@author: swjun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import *


iris = sns.load_dataset('iris')
iris.insert(0,'ones',np.ones(150))
X = iris[0:100][['ones','petal_length','petal_width']].values  # X is (100 x 3) matrix
y = np.where(iris[0:100]['species']=='setosa', 0, 1) # y is (100 x 1) vector



class LogisticRegressionGD():
    
    '''
    eta : float
        learning rate [0.0 , 1.0]
    n_iter : int
        # of iteration
    random_state : int
        random number generator for w vector
    
    w_ : 1d-array
        trained weight
    cost_ : list
        collection of the cost function values for each epoch
    '''
    
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        
        '''
        X : {array-like}, shape = [n_samples, n_features]
            [n_samples x n_features] matrix
        y : array-like, shape = [n_samples]
            [n_samples] vector
        
        return value is self : object
        '''
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.cost_ = []
        
        # Let us say X is ( 100 x 3 ) matrix. 
        # w is ( (3 x 1) ) vector. (w_0 , w_1 , w_2).T
        
        for i in range(self.n_iter):
            net_input = np.dot(X, self.w_) # (100 x 1) vector
            output = 1.0 / (1.0 + np.exp(-np.clip(net_input, -250, 250))) # (100 x 1) vector
            errors = (y - output) # (100 x 1) vector
            
            # w = w + delta(w) = w + eta * (X.T @ errors)
            self.w_ = self.w_ + self.eta * (X.T @ errors) # w_ is (3x1) , X.T is (3x100 , errors is (100x1))
            
            # J = -y@log(output)
            cost =  -y.dot(np.log(output))  - (1-y).dot(np.log(1-output))
            self.cost_.append(cost)
        return self
            
    
    def predict(self,X):
        return np.where(np.dot(X,self.w_) >= 0.0, 1, 0)
            





            
            
        


