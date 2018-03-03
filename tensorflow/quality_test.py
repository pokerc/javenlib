#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf
import cv2

#测试use_TILDE函数
sift = cv2.SIFT(250)
img_path_list = ['/home/javen/javenlib/images/wall/img1.ppm',
                 '/home/javen/javenlib/images/wall/img6.ppm']
tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/javenlib/images/wall/H1to6p')
imga = plt.imread(img_path_list[0])
imgb = plt.imread(img_path_list[1])
# img_kp_set_afternms_list = javenlib_tf.use_TILDE_scale8_withpyramid(img_path_list)
# print 'cnn done!'
# imga_kp_cnn = javenlib_tf.choose_kp_from_list(img_kp_set_afternms_list[0],quantity_to_choose=250)
# imga_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn)
# imga_kp_cnn_obj,imga_kp_cnn_des = sift.compute(imga,imga_kp_cnn_obj)
#
# imgb_kp_cnn = javenlib_tf.choose_kp_from_list(img_kp_set_afternms_list[1],quantity_to_choose=250)
# imgb_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn)
# imgb_kp_cnn_obj,imgb_kp_cnn_des = sift.compute(imgb,imgb_kp_cnn_obj)


kp_set_list = javenlib_tf.use_TILDE_scale8_withpyramid(img_path_list) #包含两幅图的kp_set
print len(kp_set_list[0]),len(kp_set_list[1])
kp_list_chosenIMG1 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[0],quantity_to_choose=250,boundary_pixel=21)
print '考虑octave边界后选出的kp list:',len(kp_list_chosenIMG1),kp_list_chosenIMG1[0]
print kp_list_chosenIMG1
kp_list_withRotatedPatchIMG1 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[0],kp_list_chosenIMG1)
print '取出28*28patch后的结构:',kp_list_withRotatedPatchIMG1[0]
kp_list_withDescriptorIMG1 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG1)
print 'kp_list_withDescriptorIMG1:',len(kp_list_withDescriptorIMG1),kp_list_withDescriptorIMG1[0][6].shape,kp_list_withRotatedPatchIMG1[0][5].shape
imga_kp_cnn,imga_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG1)

kp_list_chosenIMG2 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[1],quantity_to_choose=250,boundary_pixel=21)
kp_list_withRotatedPatchIMG2 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[1],kp_list_chosenIMG2)
kp_list_withDescriptorIMG2 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG2)
imgb_kp_cnn,imgb_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG2)

imga_kp_sift_obj = sift.detect(imga)
imga_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj)
# imga_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_sift)
imga_kp_sift_obj,imga_kp_sift_des = sift.compute(imga,imga_kp_sift_obj)

imgb_kp_sift_obj = sift.detect(imgb)
imgb_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj)
# imgb_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_sift)
imgb_kp_sift_obj,imgb_kp_sift_des = sift.compute(imgb,imgb_kp_sift_obj)

# print imga_kp_cnn_des.shape,imgb_kp_cnn_des.shape,imga_kp_sift_des.shape,imgb_kp_sift_des.shape
print 'sift shape:',imga_kp_sift.shape,imgb_kp_sift.shape
javenlib_tf.match_accuracy(imga_kp_sift,imga_kp_sift_des,imgb_kp_sift,imgb_kp_sift_des,tranform_matrix)
print 'cnn shape:',imga_kp_cnn.shape,imgb_kp_cnn.shape
javenlib_tf.match_accuracy(imga_kp_cnn,imga_kp_cnn_des,imgb_kp_cnn,imgb_kp_cnn_des,tranform_matrix)

#显示检测到的kp_set
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_sift)
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_cnn)
# javenlib_tf.show_kp_set(img_path_list[0],imga_kp_cnn)


