# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:00:41 2020
@author: Zhenzhang Li
"""

# import general python package/ read in compounds list
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
#数据的归一化处理
def featureNormalize(X):
    mu = X.mean(0)#平均值
    sigma= X.std(0)#标准差
    X = (X-mu)/sigma
    index = np.ones((len(X),1))
    X = np.hstack((X,index))
    return X
#从csv文件中读取数据
def process_data():
    data = pd.read_excel('to_predict_relative_permittivity.xls',hedaer=None)
    data = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
    #print(data.isnull().any())
    #print(np.isnan(data).any())
    X = data.iloc[:,1:98]#特征集
    Y = data.iloc[:,99]#目标集
    Y = Y.values.reshape(-1, 1)
    #数据的归一化处理（特征缩放）
    X = featureNormalize(X)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8)
    return train_x, test_x, train_y, test_y
def svr(train_x, test_x, train_y):
    lft = SVR(kernel="linear",C=0.5)
    lft.fit(train_x,train_y)
    pred_y=lft.predict(test_x)
    return pred_y
def score(Y,pred_y):
    #4-1.评价
    print("mae:",mean_absolute_error(Y,pred_y))
    print("mse:",mean_squared_error(Y,pred_y))
    print("median-ae:",median_absolute_error(Y,pred_y))
    print("r2:",r2_score(Y,pred_y))
if __name__=="__main__":
    train_x, test_x, train_y, test_y=process_data()
    pred_y=svr(train_x, test_x, train_y)
    score(test_y, pred_y)