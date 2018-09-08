# Created on:   2018-01-27
# Updated on:   2018-09-06

# Author:       coneypo
# Blog:         http://www.cnblogs.com/AdaminXie/
# Github:       https://github.com/coneypo/Smile_Detector

# draw the positions of someone's lip

import dlib         # 人脸识别的库 Dlib
import cv2          # 图像处理的库 OpenCv
from get_features import get_features   # return the positions of feature points

path_test_img = "data_imgs/test_imgs/i064rc-mn.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

pos_49to68 = get_features(path_test_img)

img_rd = cv2.imread(path_test_img)

# draw on the lip points
for i in range(0, len(pos_49to68), 2):
    print(pos_49to68[i],pos_49to68[i+1])
    cv2.circle(img_rd, tuple([pos_49to68[i],pos_49to68[i+1]]), radius=1, color=(0,255,0))

cv2.namedWindow("img_read", 2)
cv2.imshow("img_read", img_rd)
cv2.waitKey(0)