#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf
import cv2
import cmath

#测试use_TILDE函数
sift = cv2.xfeatures2d.SIFT_create()

sift_50 = cv2.KAZE_create(extended=True,threshold=0.024)
sift_100 = cv2.KAZE_create(extended=True,threshold=0.020)
sift_250 = cv2.KAZE_create(extended=True,threshold=0.012)
sift_500 = cv2.KAZE_create(extended=True,threshold=0.008)
sift_1000 = cv2.KAZE_create(extended=True,threshold=0.004)
# for count in range(159):
#     if(count < 10):
#         s1 = '000000000' + str(count)
#     elif(count < 100):
#         s1 = '00000000' + str(count)
#     else:
#         s1 = '0000000' + str(count)
#     if (count+1 < 10):
#         s2 = '000000000' + str(count+1)
#     elif (count+1 < 100):
#         s2 = '00000000' + str(count+1)
#     else:
#         s2 = '0000000' + str(count+1)
#     print 's1:',s1
#     print 's2:',s2
img_path_list = ['/home/javen/javenlib/images/kitti_city_gray_0005/0000000010.png',
                 '/home/javen/javenlib/images/kitti_city_gray_0005/0000000011.png']
imga = cv2.imread(img_path_list[0])
imgb = cv2.imread(img_path_list[1])

# if imga.mean() < 1:
#     imgc = np.copy(np.dot(imga*255.0,[0.2989,0.5870,0.1140]))
#     imga = np.copy(imgc.round().astype(np.uint8))
#     imgc = np.copy(np.dot(imgb*255.0,[0.2989,0.5870,0.1140]))
#     imgb = np.copy(imgc.round().astype(np.uint8))


#变化矩阵载入,两种方法,已知的和自己计算的
# #第一种,已知的
# tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/javenlib/images/bikes/H1to3p')

# #第二种,使用计算得到的变换矩阵
# degree = -30
# radian = 1.0*degree/180.0*cmath.pi
# print imga.shape
# row_num = imga.shape[0]
# column_num = imga.shape[1]
# tranform_matrix = np.array([[cmath.cos(radian),-cmath.sin(radian),-0.5*(column_num-1)*cmath.cos(radian)+0.5*(row_num-1)*cmath.sin(radian)+0.5*(column_num-1)],
# 							  [cmath.sin(radian),cmath.cos(radian),-0.5*(column_num-1)*cmath.sin(radian)-0.5*(row_num-1)*cmath.cos(radian)+0.5*(row_num-1)],
# 							  [0,0,1]]).real

# #第三种,amos数据集的homography
# tranform_matrix = np.array([[1,0,0],
#                             [0,1,0],
#                             [0,0,1]])

#第四种,kitti数据集的homography
tranform_matrix = javenlib_tf.get_homography_from2picture(img_path_list)
print tranform_matrix

# # print imga.shape,imga.mean()
# imga_laplacian = cv2.Laplacian(imga,ddepth=0,ksize=1)
# imgb_laplacian = cv2.Laplacian(imgb,ddepth=0,ksize=1)

#(1) 理论式NMS + 无scale处理 + SIFT求descriptor(128维)
img_kp_set_afternms_list = javenlib_tf.use_TILDE_scale8(img_path_list)
print 'cnn done!'

############# 50
imga_kp_cnn_50 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[0],quantity_to_choose=50)
imga_kp_cnn_50 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imga_kp_cnn_50)
imga_kp_cnn_obj_50 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn_50)
# print 'imga_kp_cnn_obj:',len(imga_kp_cnn_obj_50),imga_kp_cnn_obj_50[0].pt,imga_kp_cnn_obj_50[0].size
imga_kp_cnn_obj_50,imga_kp_cnn_des_50 = sift.compute(imga,imga_kp_cnn_obj_50)

imgb_kp_cnn_50 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[1],quantity_to_choose=50)
imgb_kp_cnn_50 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imgb_kp_cnn_50)
imgb_kp_cnn_obj_50 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn_50)
imgb_kp_cnn_obj_50,imgb_kp_cnn_des_50 = sift.compute(imgb,imgb_kp_cnn_obj_50)

imga_kp_sift_obj_50 = sift_50.detect(imga)
print 'length:',len(imga_kp_sift_obj_50)
imga_kp_sift_50 = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj_50[0:50])
# imga_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_sift)
imga_kp_sift_obj_50,imga_kp_sift_des_50 = sift_50.compute(imga,imga_kp_sift_obj_50[0:50])
print imga_kp_sift_50.shape,imga_kp_sift_des_50.shape
imgb_kp_sift_obj_50 = sift_50.detect(imgb)
imgb_kp_sift_50 = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj_50[0:50])
# imgb_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_sift)
imgb_kp_sift_obj_50,imgb_kp_sift_des_50 = sift_50.compute(imgb,imgb_kp_sift_obj_50[0:50])

############ 100
imga_kp_cnn_100 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[0],quantity_to_choose=100)
imga_kp_cnn_100 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imga_kp_cnn_100)
imga_kp_cnn_obj_100 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn_100)
imga_kp_cnn_obj_100,imga_kp_cnn_des_100 = sift.compute(imga,imga_kp_cnn_obj_100)

imgb_kp_cnn_100 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[1],quantity_to_choose=100)
imgb_kp_cnn_100 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imgb_kp_cnn_100)
imgb_kp_cnn_obj_100 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn_100)
imgb_kp_cnn_obj_100,imgb_kp_cnn_des_100 = sift.compute(imgb,imgb_kp_cnn_obj_100)

imga_kp_sift_obj_100 = sift_100.detect(imga)
imga_kp_sift_100 = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj_100[0:100])
imga_kp_sift_obj_100,imga_kp_sift_des_100 = sift_100.compute(imga,imga_kp_sift_obj_100[0:100])

imgb_kp_sift_obj_100 = sift_100.detect(imgb)
imgb_kp_sift_100 = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj_100[0:100])
imgb_kp_sift_obj_100,imgb_kp_sift_des_100 = sift_100.compute(imgb,imgb_kp_sift_obj_100[0:100])

############### 250
imga_kp_cnn_250 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[0],quantity_to_choose=250)
imga_kp_cnn_250 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imga_kp_cnn_250)
imga_kp_cnn_obj_250 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn_250)
imga_kp_cnn_obj_250,imga_kp_cnn_des_250 = sift.compute(imga,imga_kp_cnn_obj_250)

imgb_kp_cnn_250 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[1],quantity_to_choose=250)
imgb_kp_cnn_250 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imgb_kp_cnn_250)
imgb_kp_cnn_obj_250 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn_250)
imgb_kp_cnn_obj_250,imgb_kp_cnn_des_250 = sift.compute(imgb,imgb_kp_cnn_obj_250)

imga_kp_sift_obj_250 = sift_250.detect(imga)
imga_kp_sift_250 = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj_250[0:250])
imga_kp_sift_obj_250,imga_kp_sift_des_250 = sift_250.compute(imga,imga_kp_sift_obj_250[0:250])

imgb_kp_sift_obj_250 = sift_250.detect(imgb)
imgb_kp_sift_250 = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj_250[0:250])
imgb_kp_sift_obj_250,imgb_kp_sift_des_250 = sift_250.compute(imgb,imgb_kp_sift_obj_250[0:250])

############# 500
imga_kp_cnn_500 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[0],quantity_to_choose=500)
imga_kp_cnn_500 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imga_kp_cnn_500)
imga_kp_cnn_obj_500 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn_500)
imga_kp_cnn_obj_500,imga_kp_cnn_des_500 = sift.compute(imga,imga_kp_cnn_obj_500)

imgb_kp_cnn_500 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[1],quantity_to_choose=500)
imgb_kp_cnn_500 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imgb_kp_cnn_500)
imgb_kp_cnn_obj_500 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn_500)
imgb_kp_cnn_obj_500,imgb_kp_cnn_des_500 = sift.compute(imgb,imgb_kp_cnn_obj_500)

imga_kp_sift_obj_500 = sift_500.detect(imga)
imga_kp_sift_500 = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj_500[0:500])
imga_kp_sift_obj_500,imga_kp_sift_des_500 = sift_500.compute(imga,imga_kp_sift_obj_500[0:500])

imgb_kp_sift_obj_500 = sift_500.detect(imgb)
imgb_kp_sift_500 = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj_500[0:500])
imgb_kp_sift_obj_500,imgb_kp_sift_des_500 = sift_500.compute(imgb,imgb_kp_sift_obj_500[0:500])

############## 1000
imga_kp_cnn_1000 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[0],quantity_to_choose=1000)
imga_kp_cnn_1000 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imga_kp_cnn_1000)
imga_kp_cnn_obj_1000 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn_1000)
imga_kp_cnn_obj_1000,imga_kp_cnn_des_1000 = sift.compute(imga,imga_kp_cnn_obj_1000)

imgb_kp_cnn_1000 = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[1],quantity_to_choose=1000)
imgb_kp_cnn_1000 = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imgb_kp_cnn_1000)
imgb_kp_cnn_obj_1000 = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn_1000)
imgb_kp_cnn_obj_1000,imgb_kp_cnn_des_1000 = sift.compute(imgb,imgb_kp_cnn_obj_1000)

imga_kp_sift_obj_1000 = sift_1000.detect(imga)
imga_kp_sift_1000 = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj_1000[0:1000])
imga_kp_sift_obj_1000,imga_kp_sift_des_1000 = sift_1000.compute(imga,imga_kp_sift_obj_1000[0:1000])

imgb_kp_sift_obj_1000 = sift_1000.detect(imgb)
imgb_kp_sift_1000 = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj_1000[0:1000])
imgb_kp_sift_obj_1000,imgb_kp_sift_des_1000 = sift_1000.compute(imgb,imgb_kp_sift_obj_1000[0:1000])



############################################################
#显示quality测试的结果
# print imga_kp_cnn_des.shape,imgb_kp_cnn_des.shape,imga_kp_sift_des.shape,imgb_kp_sift_des.shape
ac_sift_50 = javenlib_tf.match_accuracy(imga_kp_sift_50,imga_kp_sift_des_50,imgb_kp_sift_50,imgb_kp_sift_des_50,tranform_matrix)
ac_cnn_50 = javenlib_tf.match_accuracy(imga_kp_cnn_50,imga_kp_cnn_des_50,imgb_kp_cnn_50,imgb_kp_cnn_des_50,tranform_matrix)

ac_sift_100 = javenlib_tf.match_accuracy(imga_kp_sift_100,imga_kp_sift_des_100,imgb_kp_sift_100,imgb_kp_sift_des_100,tranform_matrix)
ac_cnn_100 = javenlib_tf.match_accuracy(imga_kp_cnn_100,imga_kp_cnn_des_100,imgb_kp_cnn_100,imgb_kp_cnn_des_100,tranform_matrix)

ac_sift_250 = javenlib_tf.match_accuracy(imga_kp_sift_250,imga_kp_sift_des_250,imgb_kp_sift_250,imgb_kp_sift_des_250,tranform_matrix)
ac_cnn_250 = javenlib_tf.match_accuracy(imga_kp_cnn_250,imga_kp_cnn_des_250,imgb_kp_cnn_250,imgb_kp_cnn_des_250,tranform_matrix)

ac_sift_500 = javenlib_tf.match_accuracy(imga_kp_sift_500,imga_kp_sift_des_500,imgb_kp_sift_500,imgb_kp_sift_des_500,tranform_matrix)
ac_cnn_500 = javenlib_tf.match_accuracy(imga_kp_cnn_500,imga_kp_cnn_des_500,imgb_kp_cnn_500,imgb_kp_cnn_des_500,tranform_matrix)

ac_sift_1000 = javenlib_tf.match_accuracy(imga_kp_sift_1000,imga_kp_sift_des_1000,imgb_kp_sift_1000,imgb_kp_sift_des_1000,tranform_matrix)
ac_cnn_1000 = javenlib_tf.match_accuracy(imga_kp_cnn_1000,imga_kp_cnn_des_1000,imgb_kp_cnn_1000,imgb_kp_cnn_des_1000,tranform_matrix)
print "\n"
print 'ac:',ac_sift_50,ac_cnn_50,ac_sift_100,ac_cnn_100,ac_sift_250,ac_cnn_250,ac_sift_500,ac_cnn_500,ac_sift_1000,ac_cnn_1000
print img_path_list[0]
# print '100:',ac_sift_100,ac_cnn_100
# print '250:',ac_sift_250,ac_cnn_250
# print '500:',ac_sift_500,ac_cnn_500
# print '1000:',ac_sift_1000,ac_cnn_1000

# #显示检测到的kp_set
# javenlib_tf.show_kp_set(img_path_list[0],imga_kp_sift_250)
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_sift_250)
# javenlib_tf.show_kp_set(img_path_list[0],imga_kp_cnn_250)
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_cnn_250)



