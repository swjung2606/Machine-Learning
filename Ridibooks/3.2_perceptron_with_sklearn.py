# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:23:18 2022

@author: swjun
"""

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

from sklearn.model_selection import train_test_split

# Randomly assign 30% as test data, 70% as train data
X_train , X_test , y_train , y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


from sklearn.preprocessing import StandardScaler

# load StandardScaler object as sc
sc = StandardScaler()
# it computes the mean and the std of X_train
sc.fit(X_train)
# Standardization of X_train and X_test
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# Let us train the perceptron model
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)




