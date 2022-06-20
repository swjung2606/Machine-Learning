# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:18:02 2022

@author: swjun
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/rickiepark/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt',
                 header=None, sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']


# 10.2.2
# 산점도 행렬 그리기. feature 간의 상관관계를 시각화 해준다.
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
scatterplotmatrix(df[cols].values, figsize=(10,8), names=cols, alpha=0.5)
plt.tight_layout()
plt.show()


# 10.2.3 Correlation matrix
from mlxtend.plotting import heatmap
# cm = cols 에 들어있는 다섯개의 feature 들 간의 correlation 값을 보여주는 매트릭스.
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()






# Example
# https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9


from sklearn import linear_model
X = df[df.columns[0:13]]
y = df['MEDV']

# object lm 을 정의하고
lm = linear_model.LinearRegression()
# Linear regression 을 실행한다. 
lm.fit(X,y)


# 이제 결과를 볼 차례...
# lm.predict(X) 는 예상되는 y 값을 모두 보여준다. 즉 len(lm.predict(X)) 는 샘플갯수 506개와 같음.
lm.predict(X)

# 예측한 regression 이 얼마나 잘 매칭되는지 R^2 값을 구하면
lm.score(X,y)

# 각 feature 에 해당하는 coefficient 들을 살펴보자. 즉,
# 구한 식 (a_0)  +  (a_1)* 'CRIM'  +  (a_2)* 'ZN'  + ... + (a_13)* 'LSTAT'
# 이 식에서의 (a_0) , (a_1) , ... , (a_13) 의 값을 보여준다.
lm.coef_


# 마지막으로 상수항 (a_0) 을 보자.
lm.intercept_
