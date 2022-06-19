# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:17:56 2022

@author: swjun
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np



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



# Data Extraction
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)


# create object sc for standardization
sc = StandardScaler()
# save the information of X_train's mean and standard-deviation.
sc.fit(X_train)
# standardize X_train and X_test based on X_train's mean and standard-deviation by using transform.
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# merge two matrices vertically
X_combined_std = np.vstack((X_train_std, X_test_std))
# merge two vectors
y_combined = np.hstack((y_train, y_test))




# Create the SVM object
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm,
                      test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()





