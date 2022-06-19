# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:26:47 2022

@author: swjun
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


'''

Wine sample data set

wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data', header=None)

'''

# Data setting
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data', header=None)

df_wine=df_wine.set_axis(['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'], axis=1)

# Divide train data and test data
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0, stratify=y)






### 4.4 Normalization and Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# Various ways of scaling 
ex = np.array([0,1,2,3,4,5])
from sklearn.preprocessing import scale, minmax_scale, robust_scale, maxabs_scale
print('StandardScaler: ', scale(ex))
print('MinMaxScaler: ', minmax_scale(ex))
print('RobustScaler: ', robust_scale(ex))
print('MaxAbsScaler: ', maxabs_scale(ex))





### Penalty for logistic regression
# C=1.0 is base.
# We can control the effect of penalty by changeing C

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', penalty='l1', C=1.0, random_state=1)
lr.fit(X_train_std, y_train)
print('Train Data Accuracy : ', lr.score(X_train_std, y_train))
print('Test Data Accuracy : ', lr.score(X_test_std, y_test))


# Check 절편값 of the model
# Each of three classes has their own 절편값(w_0)
lr.intercept_

# It shows w_1 , w_2 , . . . , w_n
lr.coef_

# 규제 강도 변화 (C값)에 대한 가중치 변화 graph
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
weights, params = [], []

for c in np.arange(-4.0 , 6.0):
    lr = LogisticRegression(solver='liblinear', penalty='l1', C=10.0**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    
weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:,column], label=df_wine.columns[column+1], color=color)
    
plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()







### Feature selection
# Sequential Backward Selection Algorithm
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score


class SBS():
    
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        
        self.k_score_ = self.scores_[-1]
        
        return self
    
    
    def transform(self, X):
        return X[:,self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:,indices], y_train)
        y_pred = self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test, y_pred)
        return score
    

# Let us check if SBS method works well by using kNN algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)


# Let us see how accuracy changes depending on the number of features
# According to the result of the graph, the accuracy shows 100% when the number of features are 3,7,8,9,10,11,12
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# Now, we are wondering which features lead us to get 100% accuracy. 
k3 = list(sbs.subsets_[10])  
print('Three Features that can attain 100% accuracy :',df_wine.columns[1:][k3])  


# Check accuracy score of original model (with 13 features)
knn.fit(X_train_std, y_train)
print('Accuracy of Train data of Original Model (13 features) :', knn.score(X_train_std, y_train))
print('Accuracy of Test data of Original Model (13 features) :', knn.score(X_test_std, y_test))

print()

# Now, let us see the accuracy of the model with 3 features
knn.fit(X_train_std[:,k3], y_train)
print('Accuracy of Train data of SBS Model (3 features) :', knn.score(X_train_std[:,k3], y_train))
print('Accuracy of Test data of SBS Model (3 features) :', knn.score(X_test_std[:,k3], y_test))


    
        






