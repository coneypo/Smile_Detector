# Smile Detector

Author:       coneypo
Last Updated: Sep 6

Training and testing smile-detector Machine Learning models / 利用机器学习模型训练和检测笑脸 

1. get_features.py : 
  
  func get_features(img_rd, pos_49to68):　　　　
    # 输入人脸图像路径，利用 Dlib 的 “shape_predictor_68_face_landmarks.dat” 提取嘴部20个特征点坐标的40个特征值；
  func write_into_CSV(): 　　　　　　　　　　　　  
    # 将40维特征输入和1维的输出标记（1代表有微笑/0代表没微笑）写入 CSV 文件中；


2. ML_ways_sklearn.py :
  
  func pre_data():　　　　　　　　　
    # 读取 CSV 中的数据，然后提取出训练集 X_train 和测试集 X_test　


3. show_lip.py :
    #显示某人嘴唇的位置　
    
Tks for your support.
