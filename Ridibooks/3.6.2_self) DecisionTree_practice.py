# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:22:48 2022

@author: swjun
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz




def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # Set marker and colormap
    markers = ('s','x','o','^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Define the boundaries
    x1_min, x1_max = X[:,0].min() - 1 , X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1 , X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl,0], y = X[y==cl,1], 
                    alpha = 0.8, c=colors[idx], 
                    marker=markers[idx], label=cl, 
                    edgecolor='black')
        
    # Highlight test samples
    
    if test_idx:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], 
                    facecolors='none', edgecolor='black', alpha=1.0, 
                    linewidth=1, marker='o', 
                    s=100, label='test set')




KOR_h = np.random.normal(161.2 , 3 , (200,1))
VIE_h = np.random.normal(152.7 , 3 , (200,1))
HOL_h = np.random.normal(169.8 , 3 , (200,1))

KOR_w = np.random.normal(57.6, 3, (200,1))
VIE_w = np.random.normal(51.1, 3, (200,1))
HOL_w = np.random.normal(64.8, 3, (200,1))

K = np.concatenate([KOR_h, KOR_w], axis=1)
V = np.concatenate([VIE_h, VIE_w], axis=1)
H = np.concatenate([HOL_h, HOL_w], axis=1)


y_KOR = np.zeros(200,int)
y_VIE = np.ones(200,int)
y_HOL = np.ones(200,int)*2


# Train data setting
X_train = np.vstack((K,V,H))
y_train = np.hstack((y_KOR,y_VIE,y_HOL))

nation_tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
nation_tree.fit(X_train, y_train)



dot_data = export_graphviz(nation_tree, filled=True, rounded=True, 
                           class_names=['Korea','Vietnam', 'Holland'], feature_names=['Height','Weight'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('Decision_Tree_Practice.png')

