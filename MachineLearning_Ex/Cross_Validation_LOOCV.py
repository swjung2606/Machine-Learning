# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:04:53 2022

@author: sunjung
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut




############# 1D Example (y = ax+b) #############


X = np.array([[1,2,3,4,5,6,7]])
X = X.reshape((7,1))
y = np.array([2.9, 4.9, 7.1, 8.85, 11, 13.1, 14.9])

poly = PolynomialFeatures(degree=1)
X_poly = pd.DataFrame(poly.fit_transform(X))


loo = LeaveOneOut()
model = LinearRegression()

# [ -(y1 - yy1)**2 , -(y2 - yy2)**2 , ... , -(y7 - yy7)**2 ]
scores = cross_val_score(model, X_poly , y , scoring='neg_mean_squared_error', cv=loo, n_jobs=-1)

# Root Mean Square Error, RMSE
result = np.sqrt(abs(scores).mean())






############# 2D Example (z = ax + by + c) #############


dict_data = {'x' : [1,1,1,2,2,2,3,3,3] , 'y' : [1,2,3,1,2,3,1,2,3] , 'z' : [1.59,0.47, -0.53, 3.07, 2.1, 0.87, 4.39, 3.51, 2.57]}
df = pd.DataFrame(dict_data)

XX = df[['x','y']].values
yy = df['z'].values

poly1 = PolynomialFeatures(degree=1)
X_poly1 = pd.DataFrame(poly1.fit_transform(XX))

loo = LeaveOneOut()
model = LinearRegression()

# [ -(observation1 - predicted1)**2 , -(obervation2 - predicted2)**2 , ... , -(observation9 - predicted9)**2 ]
scores1 = cross_val_score(model, X_poly1 , yy , scoring='neg_mean_squared_error', cv=loo, n_jobs=-1)

# Root Mean Square Error, RMSE
result1 = np.sqrt(abs(scores1).mean())






############# 2D Example 2nd order (z = a + bX + cY + dX^2 + eXY + fY^2) #############


dict_data2 = {'x' : [1,1,1,2,2,2,3,3,3] , 'y' : [1,2,3,1,2,3,1,2,3] , 'z' : [1.02,2.95, 9.1, 0.9, -0.05, 3.06, 1.98, -2.05, -2.01]}
df2 = pd.DataFrame(dict_data2)

XX2 = df2[['x','y']].values
yy2 = df2['z'].values

poly2 = PolynomialFeatures(degree=2)
X_poly2 = pd.DataFrame(poly2.fit_transform(XX2))

loo = LeaveOneOut()
model = LinearRegression()

# [ -(observation1 - predicted1)**2 , -(obervation2 - predicted2)**2 , ... , -(observation9 - predicted9)**2 ]
scores2 = cross_val_score(model, X_poly2 , yy2 , scoring='neg_mean_squared_error', cv=loo, n_jobs=-1)

# Root Mean Square Error, RMSE
result2 = np.sqrt(abs(scores2).mean())
