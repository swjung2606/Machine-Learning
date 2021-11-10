# -*- coding: utf-8 -*-
"""
@author: swjun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from collections import Counter



def normalization(data, r=1):
    
    ''' Z-score Normalization'''
    
    rad_avg = data.mean().Radius
    rad_std = data.std().Radius
    rad_res = (data.Radius - rad_avg) / rad_std
    
    wei_avg = data.mean().Weight
    wei_std = data.std().Weight
    wei_res = (data.Weight - wei_avg) / wei_std
    
    
    
    data.Radius = rad_res
    data.Weight = wei_res
    
    
    return data





def kNN(data, k, num_of_unknown):
    
    
    ''' num_of_unknown means the number of unclassified samples'''
    
    group_name = list(data.Fruit)
    group_name = group_name[0:len(group_name)-num_of_unknown]
    data_list = []
    unclassified_list = []
    
    
    # Distinguish known samples and unknown samples.
    for i in range(len(data)):
        if data.loc[i].Fruit in group_name:
            data_list.append(list(data.loc[i]))
        
        else:
            unclassified_list.append(list(data.loc[i]))
    
    
    # Compute the distance between known samples and unknown samples.
    result_list = []
    for i in range(len(unclassified_list)):
        result_list.append([])
        for j in range(len(data_list)):
            result_list[i].append( [ ( (data_list[j][1] - unclassified_list[i][1])**2 + (data_list[j][2] - unclassified_list[i][2])**2 )**0.5 , data_list[j][0] , data_list[j][3] ] )
            
    
    # result_list structure ->   [  [ [], [], ... ] , [ [], [], ...]  ] 
    # result_list[0] : Distance between unknown sample1 and known samples. ex> [1.6, 2, "Apple"] , [2.1, 2, "Lemon"], . . . 
    # result_list[0][0] : Distance between unknown sample1 and known sample1. ex> [1.6, 1, "Apple"]
    # result_list[0][0][0] : Distance between unknown sample1 and known sample1. ex> 1.6        
    
    
    
    # This will show the nearest neighbor samples of unknown samples.
    # If more classes are added such as Pear, Kiwi etc, it should be added. 
    for i in range(len(result_list)):   # len(result_list) = number of unknown samples
        Lemon = 0
        Apple = 0
        Pear = 0
        
        result_list[i].sort()
        result_list[i] = result_list[i][0:k]
        print(f"Nearest Neighbor samples for unknown sample " , unclassified_list[i])
        print("")
        print(result_list[i])
        print("")
        
        for j in range(len(result_list[i])):
            if result_list[i][j][2] == "Lemon":
                Lemon += 1
            elif result_list[i][j][2] == "Apple":
                Apple += 1
            elif result_list[i][j][2] == "Pear":
                Pear += 1
        
        if max(Lemon,Apple,Pear) == Lemon:
            print("Therefore, ", unclassified_list[i], "belongs to Lemon group.")
            print("")
            print("")
            print("")
            data.at[unclassified_list[i][0]-1 , "Fruit"] =  "Lemon"
            unclassified_list[i][3] = "Lemon"
            
        elif max(Lemon,Apple,Pear) == Apple:
            print("Therefore, ", unclassified_list[i], "belongs to Apple group.")
            print("")
            print("")
            print("")
            data.at[unclassified_list[i][0]-1 , "Fruit"] =  "Apple"
            unclassified_list[i][3] = "Apple"
            
        elif max(Lemon,Apple,Pear) == Pear:
            print("Therefore, ", unclassified_list[i], "belongs to Pear group.")
            print("")
            print("")
            print("")
            data.at[unclassified_list[i][0]-1 , "Fruit"] =  "Pear"
            unclassified_list[i][3] = "Pear"
        
        else:
            print("Undefined.")
        
        
    
    return unclassified_list , data
    
    
    
        
    


# Input the unclassified samples in kNN_assignment3_1 file.
# Important!! unknown sample should be placed under the known samples. 


data = pd.read_csv("trainData1.csv")


data = normalization(data)


# Before use the function kNN, the data frame should be (Z-Score) Normalized. 
# It will tell us where two unclassified samples should belong to, based on k = 3.

a = kNN(data,3,5) 



