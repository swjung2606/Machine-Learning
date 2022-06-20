# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:56:29 2022

@author: sunjung
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression




# Data setting
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data', header=None)
df_wine=df_wine.set_axis(['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'], axis=1)


# Define train & test data set
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, stratify=y, random_state=0)


# 5.2.2
# preprocessing (각 feature 들의 Normalization)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)





'''

########## Without sklearn part ##########

# 5.2.3
# 각 클래스에 대한 평균 백터 mean_vecs 만들기.
# 평균백터는 feature 들의 평균값. 즉, mean_vecs[0], mean_vecs[1], mean_vecs[2] 는 모두 1x13 백터이다. 13은 feature 의 갯수.
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))

# Define S_W 
d = 13 # number of feature
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter

# Define S_B
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d,1) # 열 벡터로 전환
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i+1,:].shape[0]
    mean_vec = mean_vec.reshape(d,1) # 열 벡터로 전환
    S_B += n * (mean_vec - mean_overall).dot((mean_vec-mean_overall).T)
    


# 5.2.4
# (S_W)**-1 @ S_B 에 대한 고윳값을 계산
# eigen_pairs 는 13개의 고유값과 그에 해당하는 고유벡터를 보여준다. ex) eigen_pairs[0] -> Eigenvalue, eigen_pairs[0][1] -> Eigenvector for eigen_pairs[0]
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)


# 처음 두 개의 선형 판별벡터가 Wine 데이터셋에 있는 정보 중 거의 100%를 잡아낸다.
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, alpha=0.5, align='center', label='Individual "discriminability"')
plt.step(range(1,14), cum_discr, where='mid', label='Cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear discriminants')
plt.ylim([-0.1,1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# 가장 값이 큰 두 개의 고윳값의 고유백터 두개를 연결하여 하나의 matrix W 로 만들기.
W = np.hstack( (eigen_pairs[0][1][:,np.newaxis].real, eigen_pairs[1][1][:,np.newaxis].real))



# 5.2.5
# X' = XW
X_train_lda = X_train_std.dot(W)
colors=['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1] * (-1), c=c, label=l, marker=m)
    
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
    
'''


########## sklearn part ##########


# Default visualization function
from matplotlib.colors import ListedColormap

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






# 5.2.6
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)


# 로지스틱스 회귀 분류가 잘 동작하는지 확인하기
lr = LogisticRegression(random_state=1)
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


 