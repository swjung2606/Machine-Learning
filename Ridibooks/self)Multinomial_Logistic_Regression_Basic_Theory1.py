import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Multinomial Logistic Regression 개념설명 사이트.
http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/
'''    

# Step 1. 데이터 로딩
data = pd.read_csv('t1t1.csv')

    
# Step 2. Normalization & Set X(features) and y(target)
def normalizer(a):
    n = len(a.columns)
    for i in range(0,n-1):
        a[a.columns[i]] = (a[a.columns[i]] - a[a.columns[i]].mean()) / a[a.columns[i]].std()
    

normalizer(data)
X = data[['Height','Weight']].values
y = data['Nationality'].values
# X -> (n_features x n_classes)
W = np.array([[0.1, 0.2, 0.3] , [0.1, 0.2, 0.3]])
bias = np.array([0.01, 0.05, 0.1])


# Step 3. Set net_input Z = w_0 + (w_1)(x_1) + (w_2)(x_2)
Z = X@W + bias


# Step 4. Softmax
exp_Z = np.exp(Z)

def softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

# smax 의 행 숫자는 샘플 숫자이고, 각 행이 합은 1.
# 즉, 각 행은 그 샘플이 해당 클래스에 속할 확률을 보여줌.
smax = softmax(exp_Z)


# Define y_enc for computing cost function
def train_class_encoder(y):
    y_enc=[]
    for i in range(len(y)):
        if y[i]==0:
            y_enc.append([1,0,0])
        elif y[i]==1:
            y_enc.append([0,1,0])
        elif y[i]==2:
            y_enc.append([0,0,1])
            
    return y_enc

y_enc = train_class_encoder(y)


# Cost Function can be defined as  J = (1/n) * sum(-T * log(O))
# T is y_enc value, O is softmax(smax) value. 
# sum(-T * log(O))  <- this part is called cross_entropy

def cross_entropy(smax, y_enc):
    return -np.sum(np.log(smax) * y_enc , axis=1)



# Simply, cross entropy is multiplication of smax and y_enc
# ex) smax[0] * y_enc[0] = xent[0]

xent = cross_entropy(smax, y_enc)
print('Cross Entropy:', xent)

def cost(smax, y_enc):
    return np.mean(cross_entropy(smax, y_enc))

# J_cost is the mean of xent vector
J_cost = cost(smax, y_enc)
print('Cost: ', J_cost)





