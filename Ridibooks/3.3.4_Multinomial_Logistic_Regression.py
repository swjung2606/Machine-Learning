# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:28:56 2022

@author: swjun
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
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




# Define the object of logistics regression
lr = LogisticRegression(C=100.0, random_state=1)
# lr.fit(X_train_std, y_train) 이것이 의미하는 것은 X_train_std 와 y_train 두 데이터를 이용하여 각 클래스에 적합한 weight 계수를 계산하는 것을 의미 한다. 
# 즉, lr.fit 이 명령어 자체가 머신러닝을 했다는 의미임.
lr.fit(X_train_std, y_train)

# Draw graph
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



# Let us show the probability of which new samples belong to which class
# It shows the probability of in which classes first three new samples belong
lr.predict_proba(X_test_std[0:3, :])

# It shows which sample is predicted to belong to which classes
lr.predict_proba(X_test_std[0:3, :]).argmax(axis=1)


# It shows that the test data X_test_std is almost correctly predicted by Logistic Regression lr except one component.  
print(np.argmax(lr.predict_proba(X_test_std), axis=1) == y_test)




