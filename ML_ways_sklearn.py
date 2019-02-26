# Author:       coneypo
# Blog:         http://www.cnblogs.com/AdaminXie/
# Github:       https://github.com/coneypo/Smile_Detector

# Created on:   2018-01-27
# Updated on:   2018-10-09

# pandas 读取 CSV
import pandas as pd

# 分割数据
from sklearn.model_selection import train_test_split

# 用于数据预加工标准化
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression     # 线性模型中的 Logistic 回归模型
from sklearn.neural_network import MLPClassifier        # 神经网络模型中的多层网络模型
from sklearn.svm import LinearSVC                       # SVM 模型中的线性 SVC 模型
from sklearn.linear_model import SGDClassifier          # 线性模型中的随机梯度下降模型

from sklearn.externals import joblib


# 从 csv 读取数据
def pre_data():
    # 41 维表头
    column_names = []
    for i in range(0, 40):
        column_names.append("feature_" + str(i + 1))
    column_names.append("output")

    # read csv
    rd_csv = pd.read_csv("data/data_csvs/data.csv", names=column_names)

    # 输出 csv 文件的维度
    # print("shape:", rd_csv.shape)

    X_train, X_test, y_train, y_test = train_test_split(

        # input 0-40
        # output 41
        rd_csv[column_names[0:40]],
        rd_csv[column_names[40]],

        # 25% for testing, 75% for training
        test_size=0.25,
        random_state=33)

    return X_train, X_test, y_train, y_test


path_models = "data/data_models/"


# LR, logistic regression, 逻辑斯特回归分类（线性模型）
def model_LR():
    # get data
    X_train_LR, X_test_LR, y_train_LR, y_test_LR = pre_data()

    # 数据预加工
    # 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
    ss_LR = StandardScaler()
    X_train_LR = ss_LR.fit_transform(X_train_LR)
    X_test_LR = ss_LR.transform(X_test_LR)

    # 初始化 LogisticRegression
    LR = LogisticRegression()

    # 调用 LogisticRegression 中的 fit() 来训练模型参数
    LR.fit(X_train_LR, y_train_LR)

    # save LR model
    joblib.dump(LR, path_models + "model_LR.m")

    # 评分函数
    score_LR = LR.score(X_test_LR, y_test_LR)
    # print("The accurary of LR:", score_LR)

    # print(type(ss_LR))
    return (ss_LR)


# model_LR()


# MLPC, Multi-layer Perceptron Classifier, 多层感知机分类（神经网络）
def model_MLPC():
    # get data
    X_train_MLPC, X_test_MLPC, y_train_MLPC, y_test_MLPC = pre_data()

    # 数据预加工
    ss_MLPC = StandardScaler()
    X_train_MLPC = ss_MLPC.fit_transform(X_train_MLPC)
    X_test_MLPC = ss_MLPC.transform(X_test_MLPC)

    # 初始化 MLPC
    MLPC = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)

    # 调用 MLPC 中的 fit() 来训练模型参数
    MLPC.fit(X_train_MLPC, y_train_MLPC)

    # save MLPC model
    joblib.dump(MLPC, path_models + "model_MLPC.m")

    # 评分函数
    score_MLPC = MLPC.score(X_test_MLPC, y_test_MLPC)
    # print("The accurary of MLPC:", score_MLPC)

    return (ss_MLPC)


# model_MLPC()


# Linear SVC， Linear Supported Vector Classifier, 线性支持向量分类(SVM支持向量机)
def model_LSVC():
    # get data
    X_train_LSVC, X_test_LSVC, y_train_LSVC, y_test_LSVC = pre_data()

    # 数据预加工
    ss_LSVC = StandardScaler()
    X_train_LSVC = ss_LSVC.fit_transform(X_train_LSVC)
    X_test_LSVC = ss_LSVC.transform(X_test_LSVC)

    # 初始化 LSVC
    LSVC = LinearSVC()

    # 调用 LSVC 中的 fit() 来训练模型参数
    LSVC.fit(X_train_LSVC, y_train_LSVC)

    # save LSVC model
    joblib.dump(LSVC, path_models + "model_LSVC.m")

    # 评分函数
    score_LSVC = LSVC.score(X_test_LSVC, y_test_LSVC)
    # print("The accurary of LSVC:", score_LSVC)

    return ss_LSVC


# model_LSVC()


# SGDC, Stochastic Gradient Decent Classifier, 随机梯度下降法求解(线性模型)
def model_SGDC():
    # get data
    X_train_SGDC, X_test_SGDC, y_train_SGDC, y_test_SGDC = pre_data()

    # 数据预加工
    ss_SGDC = StandardScaler()
    X_train_SGDC = ss_SGDC.fit_transform(X_train_SGDC)
    X_test_SGDC = ss_SGDC.transform(X_test_SGDC)

    # 初始化 SGDC
    SGDC = SGDClassifier(max_iter=5)

    # 调用 SGDC 中的 fit() 来训练模型参数
    SGDC.fit(X_train_SGDC, y_train_SGDC)

    # save SGDC model
    joblib.dump(SGDC, path_models + "model_SGDC.m")

    # 评分函数
    score_SGDC = SGDC.score(X_test_SGDC, y_test_SGDC)
    # print("The accurary of SGDC:", score_SGDC)

    return ss_SGDC

# model_SGDC()
