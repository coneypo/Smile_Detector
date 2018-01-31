# ML_smiles
# created: 2018-1-27
# updated: 2018-1-31
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie
# test_single_pic.py

import cv2

from ML_ways import pre_data
from ML_ways import way_LR
from ML_ways import way_MLPC
from ML_ways import way_LSVC
from ML_ways import way_SGDC

# 获得单张人脸的特征点
path_test_pic = "F:/code/python/P_ML_smile/pic/"

XXXpic = "test4.jpg"

# 训练LR模型
pre_data()

# 使用标准化参数和ML模型
ss_LR, LR = way_LR()
ss_SGDC, SGDC = way_SGDC()
ss_LSVC, LSVC = way_LSVC()
ss_MLPC, MLPC = way_MLPC()

# 提取单张40维度特征
from Get_features import returnfeatures
single_features = []
returnfeatures(path_test_pic, XXXpic, single_features)

print("single_136_features: ", len(single_features), " ", single_features)

# opencv读取
img = cv2.imread(path_test_pic+XXXpic)

# 标68个特征点
pos_tmp = []
for i in range(0, len(single_features), 2):
    # 利用cv2.circle标注嘴部特征点，共20个
    pos = tuple([single_features[i], single_features[i+1]])
    cv2.circle(img, pos, 3, color=(0, 255, 0))

############## LR模型预测 ##############

# 特征数据预加工
X_single_LR = ss_LR.transform([single_features])
# 利用训练好的LR模型预测
y_predict_LR_single = LR.predict(X_single_LR)

con_LR = str(y_predict_LR_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("LR:", con_LR)

############## MLPC模型预测 ##############

# 特征数据预加工
X_single_MLPC = ss_MLPC.transform([single_features])
# 利用训练好的MLPC模型预测
y_predict_MLPC_single = MLPC.predict(X_single_MLPC)

con_MLPC = str(y_predict_MLPC_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("MLPC:", con_MLPC)

############## LSVC模型预测 ##############

# 特征数据预加工
X_single_LSVC = ss_LSVC.transform([single_features])
# 利用训练好的LSVC模型预测
y_predict_LSVC_single = LSVC.predict(X_single_LSVC)

con_LSVC = str(y_predict_LSVC_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("LSVC:", con_LSVC)

############## SGDC模型预测 ##############

# 特征数据预加工
X_single_SGDC = ss_SGDC.transform([single_features])
# 利用训练好的SGDC模型预测
y_predict_SGDC_single = SGDC.predict(X_single_SGDC)

con_SGDC = str(y_predict_SGDC_single[0]).replace("1", "smiles").replace("0", "no_smiles")
print("SGDC:", con_SGDC)

# cv2.putText在图像上标注文字
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, "LR: "+con_LR, (20, 30), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(img, "MLPC: "+con_MLPC, (20, 120), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(img, "LSVC: "+con_LSVC, (20, 90), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(img, "SGDC: "+con_SGDC, (20, 60), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

cv2.namedWindow("img", 2)
cv2.imshow("img", img)
cv2.waitKey(0)

