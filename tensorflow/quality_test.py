#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf
import cv2

#测试use_TILDE函数
sift = cv2.SIFT(1500)
img_path_list = ['/home/javen/javenlib/images/bikes/img1.ppm',
                 '/home/javen/javenlib/images/bikes/img2.ppm']
tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/javenlib/images/bikes/H1to2p')
imga = plt.imread(img_path_list[0])
imgb = plt.imread(img_path_list[1])
kp_set_afternms_list = javenlib_tf.use_TILDE_optimized(img_path_list)
print 'a yes!'
imga_kp_cnn = kp_set_afternms_list[0]
imga_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn)
imga_kp_cnn_obj,imga_kp_cnn_des = sift.compute(imga,imga_kp_cnn_obj)

imgb_kp_cnn = kp_set_afternms_list[1]
imgb_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn)
imgb_kp_cnn_obj,imgb_kp_cnn_des = sift.compute(imgb,imgb_kp_cnn_obj)

imga_kp_sift_obj = sift.detect(imga)
imga_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj)
# imga_kp_sift = javenlib_tf.NMS_4_points_set(imga_kp_sift)
imga_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_sift)
imga_kp_sift_obj,imga_kp_sift_des = sift.compute(imga,imga_kp_sift_obj)

imgb_kp_sift_obj = sift.detect(imgb)
imgb_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj)
# imgb_kp_sift = javenlib_tf.NMS_4_points_set(imgb_kp_sift)
imgb_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_sift)
imgb_kp_sift_obj,imgb_kp_sift_des = sift.compute(imgb,imgb_kp_sift_obj)

# print imga_kp_cnn_des.shape,imgb_kp_cnn_des.shape,imga_kp_sift_des.shape,imgb_kp_sift_des.shape
print 'sift shape:',imga_kp_sift.shape,imgb_kp_sift.shape
javenlib_tf.match_accuracy(imga_kp_sift,imga_kp_sift_des,imgb_kp_sift,imgb_kp_sift_des,tranform_matrix)
print 'cnn shape:',imga_kp_cnn.shape,imgb_kp_cnn.shape
javenlib_tf.match_accuracy(imga_kp_cnn,imga_kp_cnn_des,imgb_kp_cnn,imgb_kp_cnn_des,tranform_matrix)

print 'kp cnn:','\n',imga_kp_cnn[0:10]

# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_sift)
javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_cnn)


