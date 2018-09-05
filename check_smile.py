# Created on:   2018-01-27
# Updated on:   2018-09-03
# Author:       coneypo
# Blog:         http://www.cnblogs.com/AdaminXie/
# Github:       https://github.com/coneypo/ML_smiles

# use the saved model
from sklearn.externals import joblib

from get_features import get_features
import ML_ways_sklearn

# path of test img
path_test_img = "data_imgs/test1.jpg"

# 提取单张40维度特征
pos_49to68_test = []
get_features(path_test_img, pos_49to68_test)

# path of models
path_models = "data_models/"

print("The result of"+path_test_img+":")
print('\n')

# #########  LR  ###########
LR = joblib.load(path_models+"model_LR.m")
ss_LR = ML_ways_sklearn.model_LR()
X_test_LR = ss_LR.transform([pos_49to68_test])
print("LR:", str(LR.predict(X_test_LR)[0]).replace('0', "no smile").replace('1', "with smile"))

# #########  LSVC  ###########
LSVC = joblib.load(path_models+"model_LSVC.m")
ss_LSVC = ML_ways_sklearn.model_LSVC()
X_test_LSVC = ss_LSVC.transform([pos_49to68_test])
print("LSVC:", str(LSVC.predict(X_test_LSVC)[0]).replace('0', "no smile").replace('1', "with smile"))

# #########  MLPC  ###########
MLPC = joblib.load(path_models+"model_MLPC.m")
ss_MLPC = ML_ways_sklearn.model_MLPC()
X_test_MLPC = ss_MLPC.transform([pos_49to68_test])
print("MLPC", str(MLPC.predict(X_test_MLPC)[0]).replace('0', "no smile").replace('1', "with smile"))

# #########  SGDC  ###########
SGDC = joblib.load(path_models+"model_SGDC.m")
ss_SGDC = ML_ways_sklearn.model_SGDC()
X_test_SGDC = ss_SGDC.transform([pos_49to68_test])
print("SGDC", str(SGDC.predict(X_test_SGDC)[0]).replace('0', "no smile").replace('1', "with smile"))

