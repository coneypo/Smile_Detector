# Created on:   2018-01-27
# Updated on:   2018-09-03
# Author:       coneypo
# Blog:         http://www.cnblogs.com/AdaminXie/
# Github:       https://github.com/coneypo/ML_smiles


import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv
import os           # 读取文件
import csv          # csv操作


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# 输入图像文件所在路径，返回一个41维数组（包含提取到的40维特征和1维输出标记）
def get_features(img_rd, pos_49to68):

    # 输入:  img_rd:      图像文件
    # 输出:  pos_49to68:  feature 49 to feature 68, 20 feature points in all, 40 points

    # read img file
    img = cv2.imread(img_rd)
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算 68 点坐标
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
        pos_49to68.append(pos_68[i][0])
        pos_49to68.append(pos_68[i][1])

    return pos_49to68


# 读取图像所在的路径
path_pic_smiles = "data_imgs/database/smiles/"
path_pic_no_smiles = "data_imgs/database/no_smiles/"

# 获取路径下的图像文件
imgs_smiles = os.listdir(path_pic_smiles)
imgs_no_smiles = os.listdir(path_pic_no_smiles)

# 存储提取特征数据的CSV的路径
path_csv = "data_csv/"


def write_into_CSV():
    with open(path_csv+"data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 处理带笑脸的图像
        print("######## with smiles #########")
        for i in range(len(imgs_smiles)):
            print("img:", path_pic_smiles, imgs_smiles[i])

            # 用来存放41维特征
            features_csv_smiles = []

            # append "1" means "with smiles"
            get_features(path_pic_smiles+imgs_smiles[i], features_csv_smiles)
            features_csv_smiles.append(1)
            print("features:", features_csv_smiles, "\n")

            # 写入CSV
            writer.writerow(features_csv_smiles)

        # 处理不带笑脸的图像
        print("######## no smiles #########")
        for i in range(len(imgs_no_smiles)):
            print("img", path_pic_no_smiles, imgs_no_smiles[i])

            # 用来存放41维特征
            features_csv_no_smiles = []

            # append "0" means "no smiles"
            get_features(path_pic_no_smiles+imgs_no_smiles[i], features_csv_no_smiles)
            features_csv_no_smiles.append(0)
            print("features:", features_csv_no_smiles, "\n")

            # 写入CSV
            writer.writerow(features_csv_no_smiles)


# 写入CSV
# write_into_CSV()