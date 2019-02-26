Smile Detector
##############

Introduction
************

利用机器学习模型训练和检测笑脸 /
Training and testing smile-detector with machine learning models

#. get_features.py
  
   * 输入人脸图像路径;

   * 利用 Dlib 的 “shape_predictor_68_face_landmarks.dat” 提取嘴部20个特征点坐标的40个特征值;

   * write_into_CSV() 将 40 维特征输入和1维的输出标记写入 data.csv;


#. ML_ways_sklearn.py

   * 读取 data.csv 中的数据, 然后提取出训练集 X_train 和测试集 X_test;
   * train and test by sklearn

#. check_smile.py

   * 利用模型测试图像文件中人脸是否微笑

#. show_lip.py

   * 显示某人嘴唇的位置　

More
****


For more details, please refer to my blog (in chinese) or mail to me /

可以访问我的博客获取本项目的更详细介绍，如有问题可以邮件联系我:

* Blog: https://www.cnblogs.com/AdaminXie/p/8367348.html

* Mail: coneypo@foxmail.com


仅限于交流学习, 商业合作勿扰;

Thanks for your support.
