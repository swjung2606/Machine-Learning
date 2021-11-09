# -*- coding: utf-8 -*-
"""
@author: swjun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg



def normalization(data, r=1):
    
    ''' Z-score Normalization'''
    
    name_collections = list(data.name)
    group_collections = list(data.group)
    
    result = (data - data.mean()) / (r*data.std())
    
    result.name = name_collections
    result.group = group_collections
    
    result = result[['name', 'a1', 'a2', 'a3', 'a4', 'a5', 'group']]
    
    return result



def softmax_scaling(data,r: int):
    
    new_data = normalization(data,r)  # First, it needs to be (Z-score) normalized
    a1 = np.array(new_data.a1)
    a2 = np.array(new_data.a2)
    a3 = np.array(new_data.a3)
    a4 = np.array(new_data.a4)
    a5 = np.array(new_data.a5)
    
    a1 = 1 / (1 + np.exp(-a1))
    a2 = 1 / (1 + np.exp(-a2))
    a3 = 1 / (1 + np.exp(-a3))
    a4 = 1 / (1 + np.exp(-a4))
    a5 = 1 / (1 + np.exp(-a5))
    
    new_data.a1 = a1
    new_data.a2 = a2
    new_data.a3 = a3
    new_data.a4 = a4
    new_data.a5 = a5
    
    return new_data



def distance_calculator(x: str , y: str):
    
    ''' x, y indicate data's name. For example x = "x1", y="x3" '''
    
    for i in range(len(data)):
        if x == data.name[i]:
            temp1 = np.array( [ data.a1[i] , data.a2[i] , data.a3[i] , 
                     data.a4[i] , data.a5[i] ] )
        if y == data.name[i]:
            temp2 = np.array( [ data.a1[i] , data.a2[i] , data.a3[i] , 
                     data.a4[i] , data.a5[i] ] )
            
    dist = linalg.norm(temp1-temp2, 2)
    
    return dist
    
    


data = pd.read_csv("dataset.csv") # Load data file from .csv
data = normalization(data)        # Z-score Normalization

print("(Z-Score) Normalized data frame")
print(data)
print("")
print("Distance between {x1} and {x2} = ", distance_calculator("x1","x2"))   # distance between x1 and x2
print("Distance between {x6} and {x9} = ", distance_calculator("x6","x9"))   # distance between x6 and x9
print("")

data = softmax_scaling(data, r=1)
print("")
print("Softmax Scaling data frame based on r = 1")
print(data)



