# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:33:12 2022

@author: sunjung
"""

from scipy.spatial.distance import pdist, squareform
from numpy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# 5.3.2
def rbf_kernel_pca(X, gamma, n_components):
    
    ''' 
    RBF 커널 PCA 구현
    
    매개변수
    ------------
    X : {넘파이 ndarray} , shape = [n_samples, n_features]
    gamma : float , RBF 커널 튜닝 매개변수
    n_components : int , 반환할 주성분 개수
    
    
    반환값
    ------------
    X_pc : {넘파이 ndarray}, shape = [n_samples, k_features] , 투영된 데이터셋
    
    '''
    
    # M x N 차원의 데이터셋에서 샘플 간의 유클리디안 거리의 제곱을 계산
    # 만약 샘플이 1000개 라면 999 + 998 + ... + 1 = 499500 개의 항을 가진 벡터로 계산된다. 
    sq_dists = pdist(X, 'sqeuclidean')
    
    # 샘플 간의 거리를 정방 대칭 행렬로 변환
    # 샘플간 거리의 행렬형태. 샘플이 1000개 라면 1000 x 1000 행렬이 된다.
    # 각 행은 특정한 한개의 샘플과 다른 모든 샘플간의 거리가 됨. 그래서 1000 x 1000
    mat_sq_dists = squareform(sq_dists)
    
    # 커널 행렬을 계산
    K = exp(-gamma * mat_sq_dists)
    
    # 커널 행렬을 중앙에 맞춘다
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    
    # 중앙에 맞춰진 커널 행렬의 고윳값과 고유벡터를 구합니다.
    # scipy.linalg.eigh 함수는 오름차순으로 반환합니다.
    
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    
    # 최상위 k개의 고유 벡터를 선택합니다.(투영 결과)
    X_pc = np.column_stack([eigvecs[:,i] for i in range(n_components)])
    
    
    return X_pc



'''
# example 1. 반달 모양 구분하기

# Define Dataset
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5)
plt.show()


# Application
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

# Visualization
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0,0], X_kpca[y==0,1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1,0], X_kpca[y==1,1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y==0,0], np.zeros((50,1))-0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1,0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()
'''


'''
# example 2. 

from sklearn.datasets import make_circles
X,y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()

# Application
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0,0], X_kpca[y==0,1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1,0], X_kpca[y==1,1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0,0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1,0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()
'''





# 5.3.3

from scipy.spatial.distance import pdist, squareform
from numpy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
    
    '''
    
    RBF 커널 PCA 구현
    
    
    매개변수
    ----------
    X: {넘파이 ndarray}, shape = [n_samples, n_features]
    gamma: float, RBF 커널 튜닝 매개변수
    n_components: int, 반환할 주성분 개수
    
    
    반환값
    ----------
    alphas: {넘파이 ndarray}, shape = [n_samples, k_features], 투영된 데이터셋
    lambdas: list, 고윳값
    
    '''
    
    # M x N 차원의 데이터셋에서 샘플 간의 유클리디안 거리의 제곱을 계산
    # 만약 샘플이 1000개 라면 999 + 998 + ... + 1 = 499500 개의 항을 가진 벡터로 계산된다. 
    sq_dists = pdist(X, 'sqeuclidean')
    
    # 샘플 간의 거리를 정방 대칭 행렬로 변환
    # 샘플간 거리의 행렬형태. 샘플이 1000개 라면 1000 x 1000 행렬이 된다.
    # 각 행은 특정한 한개의 샘플과 다른 모든 샘플간의 거리가 됨. 그래서 1000 x 1000
    mat_sq_dists = squareform(sq_dists)
    
    # 커널 행렬을 계산
    K = exp(-gamma * mat_sq_dists)
    
    # 커널 행렬을 중앙에 맞춘다
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    
    # 중앙에 맞춰진 커널 행렬의 고윳값과 고유벡터를 구합니다.
    # scipy.linalg.eigh 함수는 오름차순으로 반환합니다.
    
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    
    # 최상위 k개의 고유 벡터를 선택합니다.(투영 결과)
    alphas = np.column_stack([eigvecs[:,i] for i in range(n_components)])
    
    # 고유 벡터에 상응하는 고윳값을 선택합니다.
    lambdas = [eigvals[i] for i in range(n_components)]
    
    return alphas, lambdas


