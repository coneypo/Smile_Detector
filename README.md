# Smile Detector
<br>

Training and testing smile-detector Machine Learning models / 
利用机器学习模型训练和检测笑脸 
<br>

 get_features.py 
  
 > get_features(img_rd, pos_49to68);<br>　　
> 输入人脸图像路径;<br>
> 利用 Dlib 的 “shape_predictor_68_face_landmarks.dat” 提取嘴部20个特征点坐标的40个特征值；
   
> write_into_CSV();<br>
> 将 40 维特征输入和1维的输出标记写入 CSV 文件中；

<br>


 ML_ways_sklearn.py 
  
 >  pre_data();

>  读取 CSV 中的数据，然后提取出训练集 X_train 和测试集 X_test;

<br>

 show_lip.py 
> 显示某人嘴唇的位置　
   
<br>

Author :       &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; coneypo <br>
Mail : &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp;coneypo@foxmail.com <br>
Last Updated:  &nbsp;&nbsp;&nbsp; &nbsp;Oct 9
