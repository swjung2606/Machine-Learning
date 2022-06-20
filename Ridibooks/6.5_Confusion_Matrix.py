# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


########## Without sklearn part ##########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Data setting
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)


from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
# 이 명령어는 y가 악성이면(M) 클래스 1로, 양성이면(B) 0으로 표현되게 하는 기능이다.
y = le.fit_transform(y)


# Divide dataset for two group.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

