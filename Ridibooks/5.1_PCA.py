# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:00:32 2022

@author: swjun
"""




########## Without sklearn part ##########

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Data setting
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data', header=None)
df_wine=df_wine.set_axis(['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'], axis=1)


# Define train & test data set
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, stratify=y, random_state=0)


# 5.1.2
# preprocessing (각 feature 들의 Normalization)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



# 공분산행렬 정의는 책을 참고하자. 참고로 이 데이터셋의 경우 feature 갯수가 13개 이므로 
# numpy 를 이용하여 공분산 매트릭스 cov_mat (13x13) 을 얻고
cov_mat = np.cov(X_train_std.T)
# 이 행렬의 고유값과 각 고윳값에 대응하는 고유벡터가 열에 저장된 13 x 13 차원의 행렬 eigen_vec 을 얻는다.
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)



# 5.1.3
# 이제 어떤 고윳값이 가장 많은 정보를 가지고 있는지 파악해보자.
# 그러기 위해서 각 고윳값의 explained variance ratio 를 파악하자.

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# Visualization to check which eigen value is the most valuable


# 그래프에서 보다시피, 첫 번째 주 성분이 거의 40%를 커버하고, 처음 두 개의 주성분이 데이터셋에 있는 분산의 약 60%를 커버하는걸 알 수 있다.
plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label='Cumulativve explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



# 5.1.4
# (고윳값, 고유 벡터) 튜플의 리스트 만들기.
# 고유값이 높은 순서대로 고유값과 그에 해당하는 고유벡터를 나타낸다.
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# 가장 큰 두 개의 고윳값에 해당하는 고유벡터를 선택해서 투영행렬(Projection Matrix) w (13x2) 를 만든다
# [:,np.newaxis] 행백터를 열백터로 바꿔준다.
w = np.hstack((eigen_pairs[0][1][:,np.newaxis] , eigen_pairs[1][1][:,np.newaxis]))

# (124x13) X_train_std 와 (13x2) 의 w 를 dot product ㄱㄱ
X_train_PCA = X_train_std.dot(w)


# Do visualization
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_PCA[y_train==l,0], 
                X_train_PCA[y_train==l,1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()





########## 5.1.5 using sklearn ##########


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


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# PCA 변환기와 로지스틱 회귀 추정기를 초기화
pca = PCA(n_components=2)
lr = LogisticRegression(random_state=1)

# 차원 축소
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)

# 축소된 데이터셋으로 로지스틱 회귀 모델 훈련
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()



