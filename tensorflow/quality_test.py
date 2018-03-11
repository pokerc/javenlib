#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf
import cv2
import cmath

#测试use_TILDE函数
sift = cv2.SIFT(250)
img_path_list = ['/home/javen/javenlib/images/bikes/img1.ppm',
                 '/home/javen/javenlib/images/bikes/img2.ppm']
imga = plt.imread(img_path_list[0])
imgb = plt.imread(img_path_list[1])

if imga.mean() < 1:
    imgc = np.copy(np.dot(imga*255.0,[0.2989,0.5870,0.1140]))
    imga = np.copy(imgc.round().astype(np.uint8))
    imgc = np.copy(np.dot(imgb*255.0,[0.2989,0.5870,0.1140]))
    imgb = np.copy(imgc.round().astype(np.uint8))


#变化矩阵载入,两种方法,已知的和自己计算的
#第一种,已知的
tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/javenlib/images/bikes/H1to2p')
# #第二种,使用计算得到的变换矩阵
# degree = -30
# radian = 1.0*degree/180.0*cmath.pi
# print imga.shape
# row_num = imga.shape[0]
# column_num = imga.shape[1]
# tranform_matrix = np.array([[cmath.cos(radian),-cmath.sin(radian),-0.5*(column_num-1)*cmath.cos(radian)+0.5*(row_num-1)*cmath.sin(radian)+0.5*(column_num-1)],
# 							  [cmath.sin(radian),cmath.cos(radian),-0.5*(column_num-1)*cmath.sin(radian)-0.5*(row_num-1)*cmath.cos(radian)+0.5*(row_num-1)],
# 							  [0,0,1]]).real



#(1) 理论式NMS + 无scale处理 + SIFT求descriptor(128维)
img_kp_set_afternms_list = javenlib_tf.use_TILDE_scale8(img_path_list)
print 'cnn done!'
imga_kp_cnn = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[0],quantity_to_choose=250)
imga_kp_cnn = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imga_kp_cnn)
imga_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn)
print 'imga_kp_cnn_obj:',len(imga_kp_cnn_obj),imga_kp_cnn_obj[0].pt,imga_kp_cnn_obj[0].size
imga_kp_cnn_obj,imga_kp_cnn_des = sift.compute(imga,imga_kp_cnn_obj)

imgb_kp_cnn = javenlib_tf.choose_kp_from_list_careboundary(img_kp_set_afternms_list[1],quantity_to_choose=250)
imgb_kp_cnn = javenlib_tf.extract_kp_pos_array_from_kp_set_list(imgb_kp_cnn)
imgb_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn)
imgb_kp_cnn_obj,imgb_kp_cnn_des = sift.compute(imgb,imgb_kp_cnn_obj)

# #(2) 理论式NMS + 无scale处理 + CNN求descriptor(pool2,3136维)
# kp_set_list = javenlib_tf.use_TILDE_scale8(img_path_list) #包含两幅图的kp_set
# print len(kp_set_list[0]),len(kp_set_list[1])
# kp_list_chosenIMG1 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[0],quantity_to_choose=250,boundary_pixel=21)
# print '考虑octave边界后选出的kp list:',len(kp_list_chosenIMG1),kp_list_chosenIMG1[0]
# print kp_list_chosenIMG1
# kp_list_withRotatedPatchIMG1 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[0],kp_list_chosenIMG1)
# print '取出28*28patch后的结构:',kp_list_withRotatedPatchIMG1[0]
# kp_list_withDescriptorIMG1 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG1)
# print 'kp_list_withDescriptorIMG1:',len(kp_list_withDescriptorIMG1),kp_list_withDescriptorIMG1[0][6].shape,kp_list_withRotatedPatchIMG1[0][5].shape
# imga_kp_cnn,imga_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG1)
#
# kp_list_chosenIMG2 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[1],quantity_to_choose=250,boundary_pixel=21)
# kp_list_withRotatedPatchIMG2 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[1],kp_list_chosenIMG2)
# kp_list_withDescriptorIMG2 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG2)
# imgb_kp_cnn,imgb_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG2)

# #(3) 理论式NMS + 3scale处理 + SIFT求descriptor(128维)
# kp_set_list = javenlib_tf.use_TILDE_scale8_withpyramid(img_path_list) #包含两幅图的kp_set
# print len(kp_set_list[0]),len(kp_set_list[1])
# kp_list_chosenIMG1 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[0],quantity_to_choose=250,boundary_pixel=21)
# print '考虑octave边界后选出的kp list:',len(kp_list_chosenIMG1),kp_list_chosenIMG1[0]
# print kp_list_chosenIMG1
# kp_list_withRotatedPatchIMG1 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[0],kp_list_chosenIMG1)
# print '取出28*28patch后的结构:',kp_list_withRotatedPatchIMG1[0]
# kp_list_withDescriptorIMG1 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG1)
# print 'kp_list_withDescriptorIMG1:',len(kp_list_withDescriptorIMG1),kp_list_withDescriptorIMG1[0][6].shape,kp_list_withRotatedPatchIMG1[0][5].shape
# imga_kp_cnn,imga_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG1)
# imga_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn)
# imga_kp_cnn_obj,imga_kp_cnn_des = sift.compute(imga,imga_kp_cnn_obj)
#
# kp_list_chosenIMG2 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[1],quantity_to_choose=250,boundary_pixel=21)
# kp_list_withRotatedPatchIMG2 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[1],kp_list_chosenIMG2)
# kp_list_withDescriptorIMG2 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG2)
# imgb_kp_cnn,imgb_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG2)
# imgb_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn)
# imgb_kp_cnn_obj,imgb_kp_cnn_des = sift.compute(imgb,imgb_kp_cnn_obj)

# #(4) 理论式NMS + 3scale处理 + CNN求descriptor(pool2,3136维)
# kp_set_list = javenlib_tf.use_TILDE_scale8_withpyramid(img_path_list) #包含两幅图的kp_set
# print len(kp_set_list[0]),len(kp_set_list[1])
# kp_list_chosenIMG1 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[0],quantity_to_choose=250,boundary_pixel=21)
# print '考虑octave边界后选出的kp list:',len(kp_list_chosenIMG1),kp_list_chosenIMG1[0]
# print kp_list_chosenIMG1
# kp_list_withRotatedPatchIMG1 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[0],kp_list_chosenIMG1)
# print '取出28*28patch后的结构:',kp_list_withRotatedPatchIMG1[0]
# kp_list_withDescriptorIMG1 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG1)
# print 'kp_list_withDescriptorIMG1:',len(kp_list_withDescriptorIMG1),kp_list_withDescriptorIMG1[0][6].shape,kp_list_withRotatedPatchIMG1[0][5].shape
# imga_kp_cnn,imga_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG1)
#
# kp_list_chosenIMG2 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[1],quantity_to_choose=250,boundary_pixel=21)
# kp_list_withRotatedPatchIMG2 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[1],kp_list_chosenIMG2)
# kp_list_withDescriptorIMG2 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG2)
# imgb_kp_cnn,imgb_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG2)

# #(5) SIFT求pos + CNN求descriptor(pool2,3136维)
# imga_kp_cnn_obj = sift.detect(imga)
# imga_kp_cnn = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_cnn_obj)
# print 'img kp cnn:',imga_kp_cnn.shape,imga_kp_cnn[0:5]
# row_num = imga.shape[0]
# column_num = imga.shape[1]
# print row_num,column_num
# kp_list_chosenIMG1 = javenlib_tf.transform_from_array_2_list(imga_kp_cnn,row_num,column_num)
# # print kp_list_chosenIMG1[0:5]
# kp_list_withRotatedPatchIMG1 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[0],kp_list_chosenIMG1)
# kp_list_withDescriptorIMG1 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG1)
# imga_kp_cnn,imga_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG1)
#
# imgb_kp_cnn_obj = sift.detect(imgb)
# imgb_kp_cnn = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_cnn_obj)
# kp_list_chosenIMG2 = javenlib_tf.transform_from_array_2_list(imgb_kp_cnn,row_num,column_num)
# kp_list_withRotatedPatchIMG2 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[1],kp_list_chosenIMG2)
# kp_list_withDescriptorIMG2 = javenlib_tf.use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG2)
# imgb_kp_cnn,imgb_kp_cnn_des = javenlib_tf.kp_list_2_pos_des_array(kp_list_withDescriptorIMG2)

# #(6) 理论式NMS + 无scale处理 + ORB中rBRIEF求descriptor(32维)
# orb = cv2.ORB(250)
# kp_set_list = javenlib_tf.use_TILDE_scale8(img_path_list) #包含两幅图的kp_set
# print len(kp_set_list[0]),len(kp_set_list[1])
# kp_list_chosenIMG1 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[0],quantity_to_choose=250,boundary_pixel=21)
# print '考虑octave边界后选出的kp list:',len(kp_list_chosenIMG1),kp_list_chosenIMG1[0]
# print kp_list_chosenIMG1
# kp_list_withRotatedPatchIMG1 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[0],kp_list_chosenIMG1)
# print '取出28*28patch后的结构:',kp_list_withRotatedPatchIMG1[0]
# kp_list_objIMG1 = javenlib_tf.KeyPoint_reverse_convert_forORB(kp_list_withRotatedPatchIMG1)
# # for i in range(10):
# #     print kp_list_obj[i].pt,kp_list_obj[i].size,kp_list_obj[i].angle,kp_list_obj[i].response,kp_list_obj[i].octave,kp_list_obj[i].class_id
# kp_list_objIMG1,imga_kp_cnn_des = orb.compute(imga,kp_list_objIMG1)
# imga_kp_cnn = javenlib_tf.KeyPoint_convert_forOpencv2(kp_list_objIMG1)
#
# kp_list_chosenIMG2 = javenlib_tf.choose_kp_from_list_careboundary(kp_set_list[1],quantity_to_choose=250,boundary_pixel=21)
# kp_list_withRotatedPatchIMG2 = javenlib_tf.get_kp_list_withRotatedPatch(img_path_list[1],kp_list_chosenIMG2)
# kp_list_objIMG2 = javenlib_tf.KeyPoint_reverse_convert_forORB(kp_list_withRotatedPatchIMG2)
# kp_list_objIMG2,imgb_kp_cnn_des = orb.compute(imgb,kp_list_objIMG2)
# imgb_kp_cnn = javenlib_tf.KeyPoint_convert_forOpencv2(kp_list_objIMG2)



##########################################################
#SIFT方法求特征点pos和des
imga_kp_sift_obj = sift.detect(imga)
imga_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj)
# imga_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_sift)
imga_kp_sift_obj,imga_kp_sift_des = sift.compute(imga,imga_kp_sift_obj)

imgb_kp_sift_obj = sift.detect(imgb)
imgb_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj)
# imgb_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_sift)
imgb_kp_sift_obj,imgb_kp_sift_des = sift.compute(imgb,imgb_kp_sift_obj)

############################################################
#显示quality测试的结果
# print imga_kp_cnn_des.shape,imgb_kp_cnn_des.shape,imga_kp_sift_des.shape,imgb_kp_sift_des.shape
print 'sift shape:',imga_kp_sift.shape,imgb_kp_sift.shape
javenlib_tf.match_accuracy(imga_kp_sift,imga_kp_sift_des,imgb_kp_sift,imgb_kp_sift_des,tranform_matrix)
print 'cnn shape:',imga_kp_cnn.shape,imgb_kp_cnn.shape,imga_kp_cnn_des.shape
javenlib_tf.match_accuracy(imga_kp_cnn,imga_kp_cnn_des,imgb_kp_cnn,imgb_kp_cnn_des,tranform_matrix)

# #显示检测到的kp_set
# javenlib_tf.show_kp_set(img_path_list[0],imga_kp_sift)
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_sift)
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_cnn)
# javenlib_tf.show_kp_set(img_path_list[0],imga_kp_cnn)


