# ML_smiles
# 2018-1-27
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie/
# Get_features.py


import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv
import os           # 读取文件
import csv          # csv操作

# ML_smiles
# 2018-1-27
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie/


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 输入图像文件所在路径，返回一个41维数组（包含提取到的40维特征和1维输出标记）
def returnfeatures(path_pic, XXXpic, features_csv):

    # 输入:  path_pic:    图像文件所在目录
    #       XXXpic:      图像文件名

    # 输出:  features_csv 41维度的数组，前40维为(提取的20个特征点坐标的40个值)，第41维为标记output

    # 比如 path_pic + XXXpic = "F:/code/test.jpg" 精确到jpg
    img = cv2.imread(path_pic + XXXpic)
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算68点坐标
    pos_68 = []
    rects = detector(img_gray, 0)
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        pos_68.append(pos)

    # 将点49-68写入csv
    # 即pos_68[48]-pos_68[67]
    for i in range(48, 68):
        features_csv.append(pos_68[i][0])
        features_csv.append(pos_68[i][1])

    #print(features_csv)
    return features_csv

# ML_smiles
# 2018-1-27
# By TimeStamp
# cnblogs: http://www.cnblogs.com/AdaminXie/
# Get_features.py


# 读取图像所在的路径
path_pic_smile = "F:/code/python/P_ML_smile/pic/database/smile/"
path_pic_nosmile = "F:/code/python/P_ML_smile/pic/database/no/"

# 获取路径下的图像文件
namedir_smile = os.listdir(path_pic_smile)
namedir_nosmile = os.listdir(path_pic_nosmile)

# 存储提取特征数据的CSV的路径
path_csv = "F:/code/python/P_ML_smile/data_csv/"

def writeintoCSV():
    with open(path_csv+"data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 处理带笑脸的图像
        print("######## with smiles #########")
        for i in range(len(namedir_smile)):
            print("pic:", path_pic_smile, namedir_smile[i])

            # 用来存放41维特征
            features_csv_smiles = []

            # 利用 returnfeatures 函数提取特征
            returnfeatures(path_pic_smile, namedir_smile[i], features_csv_smiles)
            features_csv_smiles.append(1)
            print("features:", features_csv_smiles, "\n")

            # 写入CSV
            writer.writerow(features_csv_smiles)

        # 处理不带笑脸的图像
        print("######## no smiles #########")
        for i in range(len(namedir_nosmile)):
            print("pic:", path_pic_nosmile, namedir_nosmile[i])

            # 用来存放41维特征
            features_csv_nosmiles = []

            # 利用 returnfeatures 函数提取特征
            returnfeatures(path_pic_nosmile, namedir_nosmile[i], features_csv_nosmiles)
            features_csv_nosmiles.append(0)
            print("features:", features_csv_nosmiles, "\n")

            # 写入CSV
            writer.writerow(features_csv_nosmiles)

# 写入CSV
#writeintoCSV()