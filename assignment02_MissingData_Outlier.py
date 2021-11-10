# -*- coding: utf-8 -*-
"""
@author: swjun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from typing import List


def read_csv(filename):
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row)
    return np.array(data)

data = read_csv("assignment02.csv")  # Read data from assignment02.csv file





### Plotting Part###


Lemon_radius = []
Lemon_weight = []
Apple_radius = []
Apple_weight = []
Pear_radius = []
Pear_weight = []

for i in data:
    if i[3] == "Lemon":
        Lemon_radius.append( float(i[1]) )
        Lemon_weight.append( float(i[2]) )
    elif i[3] == "Apple":
        Apple_radius.append( float(i[1]) )
        Apple_weight.append( float(i[2]) )
    elif i[3] == "Pear":
        Pear_radius.append( float(i[1]) )
        Pear_weight.append( float(i[2]) )


plt.scatter( Lemon_radius, Lemon_weight, marker="o")
plt.scatter( Apple_radius, Apple_weight, marker="+")
plt.scatter( Pear_radius, Pear_weight, marker="x")
plt.legend(["Lemon", "Apple", "Pear"])
plt.grid()
plt.show()





### Bar Chart Part ###


def color_collections(data,fruitname: str):
    Green = 0
    Red = 0
    Yellow = 0
    for i in data:
        if i[3] == fruitname:
            if i[0] == "Green":
                Green += 1
            elif i[0] == "Red":
                Red += 1
            elif i[0] == "Yellow":
                Yellow += 1
    
    return [Green, Red, Yellow]


# Chart size and shape decision
fig, ax = plt.subplots()
index = np.array([0,1,2])
bar_width = 0.2
opacity = 1.0


# Data visualization in terms of Bar
bar1 = plt.bar(index , color_collections(data , "Apple") , bar_width , color="white" , edgecolor="black" , label="Apples" , hatch="/")
bar2 = plt.bar(index + bar_width , color_collections(data , "Pear") , bar_width , color="white" , edgecolor="black" , label="Pears" , hatch="o")
bar3 = plt.bar(index + bar_width*2 , color_collections(data , "Lemon") , bar_width , color="white" , edgecolor="black" , label="Lemons" , hatch="\\")

plt.ylabel("Frequency")
plt.xticks(index + bar_width,("Green","Red","Yellow"))
plt.legend()
plt.tight_layout()
plt.show()

    



### Outlier Removing Part ###


def same_sample_remover(data):
    
    ''' This function can remove overlapped samples '''
    
    overlapped_index = []
    
    for i in range(len(data)-1):
        for j in range(i+1, len(data)):
            data_bool = data[i] == data[j]
            if False not in data_bool:
                if i not in overlapped_index:
                    overlapped_index.append(i)
                
    print("Overlapped index : ", overlapped_index)
    print("")            
    data = list(data)
    for k in overlapped_index:
        del data[k]
    data = np.array(data)
    
    return data



def missing_attribute_filler(data): 
    
    Lemon_group = []
    Apple_group = []
    Pear_group = []
    Missed_Radius_sample_index = []
    Missed_Weight_sample_index = []
    
    # Change the data type from np.array to list.
    data = list(data)
    for i in range(len(data)):
        data[i] = list(data[i])
        data[i][1] = float(data[i][1])
        data[i][2] = float(data[i][2])
        
        
        # Samples that do not include 0 would go into their assigned group (Lemon_group, Apple_group, or Pear_group) for computing average of radius or weight.
        # Samples that include 0 would be reported their sample index number in either Missed_Radius_sample_index or Missed_Weight_sample_index.
        
        if data[i][3] == "Lemon":
            if 0 not in data[i]:
                Lemon_group.append(data[i])
            else:
                if data[i][1] == 0:
                    Missed_Radius_sample_index.append(i)
                elif data[i][2] == 0:
                    Missed_Weight_sample_index.append(i)
                    
        if data[i][3] == "Apple":
            if 0 not in data[i]:
                Apple_group.append(data[i])
            else:
                if data[i][1] == 0:
                    Missed_Radius_sample_index.append(i)
                elif data[i][2] == 0:
                    Missed_Weight_sample_index.append(i)
            
        if data[i][3] == "Pear":
            if 0 not in data[i]:
                Pear_group.append(data[i])
            else:
                if data[i][1] == 0:
                    Missed_Radius_sample_index.append(i)
                elif data[i][2] == 0:
                    Missed_Weight_sample_index.append(i)
    
                    
    
    # It lets users know which samples are missing their data.
    print("Samples that miss their radius value : ")
    for l in Missed_Radius_sample_index:
        print(data[l])
    print("")
    print("Samples that miss their weight value : ")
    for l in Missed_Weight_sample_index:
        print(data[l])
    
        
    
    # Radius missed sample will be filled with the average value of their group's radius.
    for i in Missed_Radius_sample_index:
        if data[i][3] == "Lemon":
            radius_collection = []
            for j in range(len(Lemon_group)):
                radius_collection.append(Lemon_group[j][1])
            avg = sum(radius_collection) / len(radius_collection)
            
            data[i][1] = avg
            
        if data[i][3] == "Apple":
            radius_collection = []
            for j in range(len(Apple_group)):
                radius_collection.append(Apple_group[j][1])
            avg = sum(radius_collection) / len(radius_collection)
            
            data[i][1] = avg
            
        if data[i][3] == "Pear":
            radius_collection = []
            for j in range(len(Pear_group)):
                radius_collection.append(Pear_group[j][1])
            avg = sum(radius_collection) / len(radius_collection)
            
            data[i][1] = avg
            
            
            
    # Weight missed sample will be filled with the average value of their group's weight.        
    for i in Missed_Weight_sample_index:
        if data[i][3] == "Lemon":
            weight_collection = []
            for j in range(len(Lemon_group)):
                weight_collection.append(Lemon_group[j][2])
            avg = sum(weight_collection) / len(weight_collection)
            
            data[i][2] = avg
            
        if data[i][3] == "Apple":
            weight_collection = []
            for j in range(len(Apple_group)):
                weight_collection.append(Apple_group[j][1])
            avg = sum(weight_collection) / len(weight_collection)
            
            data[i][2] = avg
            
        if data[i][3] == "Pear":
            weight_collection = []
            for j in range(len(Pear_group)):
                weight_collection.append(Pear_group[j][1])
            avg = sum(weight_collection) / len(weight_collection)
            
            data[i][2] = avg
    
    
    # Switch data type from list to np.array again.
    data = np.array(data)
    return data
    


#data_1 = same_sample_remover(data)   # This data doesn't include any overlapped sample.
#data_2 = missing_attribute_filler(data)  # This data fills all missing attribute values.


print("1. Original data frame")
print("2. Overlap samples removed data frame")
print("3. Missing values filled data frame")
ans = input("Enter the number you want to see (1/2/3) : ")

if ans == "1":
    print(data)
elif ans == "2":
    data_1 = same_sample_remover(data)
    print(data_1)
    
elif ans == "3":
    data_2 = missing_attribute_filler(data)
    print(data_2)
    
else:
    print("You should enter among three numbers 1,2, or 3.")
    



