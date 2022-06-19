# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 08:41:44 2022

@author: swjun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Data setting
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data', header=None)

df_wine=df_wine.set_axis(['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'], axis=1)
feat_labels = df_wine.columns[1:]

# Divide train data and test data
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0, stratify=y)



forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)

# It shows the importance score of each feature. Sum of importance score is 1
importances = forest.feature_importances_
# It shows each feature's importance ranking
indices = np.argsort(importances)[::-1]



# Visualize this.
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()



    
    