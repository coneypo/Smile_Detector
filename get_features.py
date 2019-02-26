# Author:       coneypo
# Blog:         http://www.cnblogs.com/AdaminXie/
# Github:       https://github.com/coneypo/Smile_Detector

# Created on:   2018-01-27
# Updated on:   2018-10-09

import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import os           # 读取文件
import csv          # CSV 操作


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib_model/shape_predictor_68_face_landmarks.dat')


# 输入图像文件所在路径，返回一个41维数组（包含提取到的40维特征和1维输出标记）
def get_features(img_rd):

    # 输入:  img_rd:      图像文件
    # 输出:  positions_lip_arr:  feature point 49 to feature point 68, 20 feature points / 40D in all

    # read img file
    img = cv2.imread(img_rd)
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算68点坐标
    positions_68_arr = []
    faces = detector(img_gray, 0)
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[0]).parts()])

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        positions_68_arr.append(pos)

    positions_lip_arr = []
    # 将点 49-68 写入 CSV
    # 即 positions_68_arr[48]-positions_68_arr[67]
    for i in range(48, 68):
        positions_lip_arr.append(positions_68_arr[i][0])
        positions_lip_arr.append(positions_68_arr[i][1])

    return positions_lip_arr


# 读取图像所在的路径
path_images_with_smiles = "data/data_imgs/database/smiles/"
path_images_no_smiles = "data/data_imgs/database/no_smiles/"

# 获取路径下的图像文件
imgs_smiles = os.listdir(path_images_with_smiles)
imgs_no_smiles = os.listdir(path_images_no_smiles)

# 存储提取特征数据的 CSV 的路径
path_csv = "data/data_csvs/"


# write the features into CSV
def write_into_CSV():
    with open(path_csv+"data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 处理带笑脸的图像
        print("######## with smiles #########")
        for i in range(len(imgs_smiles)):
            print(path_images_with_smiles+imgs_smiles[i])

            # append "1" means "with smiles"
            features_csv_smiles = get_features(path_images_with_smiles+imgs_smiles[i])
            features_csv_smiles.append(1)
            print("positions of lips:", features_csv_smiles, "\n")

            # 写入CSV
            writer.writerow(features_csv_smiles)

        # 处理不带笑脸的图像
        print("######## no smiles #########")
        for i in range(len(imgs_no_smiles)):
            print(path_images_no_smiles+imgs_no_smiles[i])

            # append "0" means "no smiles"
            features_csv_no_smiles = get_features(path_images_no_smiles + imgs_no_smiles[i])
            features_csv_no_smiles.append(0)
            print("positions of lips:", features_csv_no_smiles, "\n")

            # 写入CSV
            writer.writerow(features_csv_no_smiles)


# 写入CSV
# write_into_CSV()