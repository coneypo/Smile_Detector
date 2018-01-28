# ML_smiles
# 2018-1-27
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie/
# ML_ways.py

# pd读取CSV
import pandas as pd

# 分割数据
from sklearn.model_selection import train_test_split

# 用于数据预加工标准化
from sklearn.preprocessing import StandardScaler

# 使用的四种ML模型
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


# 从CSV读取数据
def pre_data():

    # 41维表头
    column_names = []
    for i in range(0, 40):
        column_names.append("feature_" + str(i+1))
    column_names.append("output")

    path_csv = "F:/code/python/P_ML_smile/data_csv/"

    rd_csv = pd.read_csv(path_csv+"data.csv", names=column_names)

    # 输出CSV文件的维度
    print("shape:", rd_csv.shape)

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        rd_csv[column_names[0:40]],
        rd_csv[column_names[40]],
        test_size=0.25,
        random_state=33)

# ML_smiles
# 2018-1-27
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie/
# way_LR() in ML_ways.py

# 罗吉斯特回归LR
def way_LR():

    X_train_LR = X_train
    y_train_LR = y_train

    X_test_LR = X_test
    y_test_LR = y_test

    # 标准化数据预加工
    ss_LR = StandardScaler()

    X_train_LR = ss_LR.fit_transform(X_train_LR)
    X_test_LR = ss_LR.transform(X_test_LR)

    # 初始化LogisticRegression
    LR = LogisticRegression()

    # 调用LogisticRegression中的fit()来训练模型参数
    LR.fit(X_train_LR, y_train_LR)

    # 使用训练好的模型lr对X_test进行预测，结果储存在lr_y_predict中
    global y_predict_LR
    y_predict_LR = LR.predict(X_test_LR)

    global lr_score
    lr_score=LR.score(X_test_LR, y_test_LR)
    print("The accurary of LR:", LR.score(X_test_LR, y_test_LR))

    return ss_LR, LR

# 随机梯度下降SGD
def way_SGD():

    X_train_SGD = X_train
    y_train_SGD = y_train

    X_test_SGD = X_test
    y_test_SGD = y_test

    # 标准化数据
    ss_SGD = StandardScaler()
    X_train_SGD = ss_SGD.fit_transform(X_train_SGD)
    X_test_SGD = ss_SGD.transform(X_test_SGD)

    # 初始化SGDClassifier
    SGD = SGDClassifier(max_iter=5)

    # 调用SGDClassifier中的fit函数用来训练模型参数
    SGD.fit(X_train_SGD, y_train_SGD)

    # 使用训练好的模型sgd对X_test进行预测，结果储存在sgd_y_predict中

    global y_predict_SGD
    y_predict_SGD = SGD.predict(X_test_SGD)

    global sgd_score
    sgd_score = SGD.score(X_test_SGD, y_test_SGD)
    print ("The accurary of SGD:", SGD.score(X_test_SGD, y_test_SGD))

    return ss_SGD, SGD

# 多层神经网络MLP
def way_MLP():

    X_train_MLP = X_train
    y_train_MLP = y_train

    X_test_MLP = X_test
    y_test_MLP = y_test

    ss_MLP = StandardScaler()
    X_train_MLP = ss_MLP.fit_transform(X_train_MLP)
    X_test_MLP = ss_MLP.transform(X_test_MLP)

    #调用MLP实例化
    MLP = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=500)
    MLP.fit(X_train_MLP, y_train_MLP)

    global y_predict_MLP
    y_predict_MLP = MLP.predict(X_test_MLP)
    #print(X_test_MLP)

    #结果
    global mlp_score
    mlp_score = MLP.score(X_test_MLP, y_test_MLP)
    print("The accurary of MLP:", mlp_score)

    return ss_MLP, MLP

# 支持向量机svm
def way_SVM():

    X_train_SVM = X_train
    y_train_SVM = y_train

    X_test_SVM = X_test
    y_test_SVM = y_test

    ss_SVM = StandardScaler()
    X_train_SVM = ss_SVM.fit_transform(X_train_SVM)
    X_test_SVM = ss_SVM.transform(X_test_SVM)

    #调用线性SVC实例化
    LSVC = LinearSVC()
    LSVC.fit(X_train_SVM, y_train_SVM)

    global y_predict_SVM
    y_predict_SVM = LSVC.predict(X_test_SVM)

    global svm_score
    svm_score = LSVC.score(X_test_SVM, y_test_SVM)
    print("The accurary of SVM:", LSVC.score(X_test_SVM, y_test_SVM))

    return ss_SVM, LSVC


