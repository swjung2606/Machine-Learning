# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:17:41 2022

@author: swjun
"""

import pandas as pd
import numpy as np



''' 
4.1 Missing Data Processing



df = pd.DataFrame([[1.0,2.0,3.0,4.0],[5.0,6.0,np.nan,8.0]
                   ,[10.0,11.0,12.0,np.nan]], columns=['A','B','C','D'])


# Delete rows including NaN
df.dropna(axis=0)
# Delete columns including NaN
df.dropna(axis=1)
# Delete column 'C' if it includes NaN at least one.
df.dropna(subset=['C'])


'''


# Define dataframe
df = pd.DataFrame([['green','M',10.1,'class1'], 
                  ['red', 'L', 13.5, 'class2'],
                  ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

# column 'size' mapping
# Let us assume that XL = L + 1 = M + 2 
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)

# column 'classlabel' mapping
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)

# Or, use LabelEncoder in sklearn
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)


'''
# column 'color' mapping
# color doesn't have order
# It assigns 'blue' as 0, 'green' as 1, and 'red' as 2
# But, it's not good method because the number 0,1,2 haave order, while the color blue, green, and red doesn't have.
color_le = LabelEncoder()
df['color'] = color_le.fit_transform(df['color'].values)
'''

# It is much better to use this method
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
X = df[['color' , 'size' , 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]), 
                              ('nothing','passthrough', [1,2])])
c_transf.fit_transform(X)
print(c_transf.fit_transform(X))








