# ML_smiles
# 2018-1-27
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie/
# test_single_pic.py

import cv2

from ML_ways import pre_data
from ML_ways import way_LR
from ML_ways import way_MLP
from ML_ways import way_SGD
from ML_ways import way_SVM

# 获得单张人脸的特征点
path_test_pic = "F:/code/python/P_ML_smile/pic/"
#path_test_pic = "F:/code/pic/faces/the_muct_face_database/jpg/"

XXXpic = "i064re-mn.jpg"

# 训练LR模型
pre_data()

# 使用标准化参数和ML模型
ss_LR, LR = way_LR()
ss_SGD, SGD = way_SGD()
ss_SVM, SVM = way_SVM()
ss_MLP, MLP = way_MLP()

# 提取单张40维度特征
from Get_features import returnfeatures
single_features = []
returnfeatures(path_test_pic, XXXpic, single_features)
#print("single_40_features: ", single_features)

############## LR模型预测 ##############

# 特征数据预加工
X_single_LR = ss_LR.transform([single_features])
# 利用训练好的LR模型预测
y_predict_LR_single = LR.predict(X_single_LR)

con_LR = str(y_predict_LR_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("LR:", con_LR)

############## SGD模型预测 ##############

# 特征数据预加工
X_single_SGD = ss_SGD.transform([single_features])
# 利用训练好的SGD模型预测
y_predict_SGD_single = SGD.predict(X_single_SGD)

con_SGD = str(y_predict_SGD_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("SGD:", con_SGD)

############## SVM模型预测 ##############

# 特征数据预加工
X_single_SVM = ss_SVM.transform([single_features])
# 利用训练好的SVM模型预测
y_predict_SVM_single = SVM.predict(X_single_SVM)

con_SVM = str(y_predict_SVM_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("SVM:", con_SVM)

############## MLP模型预测 ##############

# 特征数据预加工
X_single_MLP = ss_MLP.transform([single_features])
# 利用训练好的MLP模型预测
y_predict_MLP_single = MLP.predict(X_single_MLP)

con_MLP = str(y_predict_MLP_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("MLP:", con_MLP)

img = cv2.imread(path_test_pic+XXXpic)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, "LR: "+con_LR, (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(img, "SGD: "+con_SGD, (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(img, "SVM: "+con_SVM, (20, 150), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(img, "MLP: "+con_MLP, (20, 200), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

cv2.namedWindow("img", 2)
cv2.imshow("img", img)
cv2.waitKey(0)

