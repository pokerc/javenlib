#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf
import cmath



# imga = plt.imread('/home/javen/javenlib/images/bark/img1.ppm')
# imgb = plt.imread('/home/javen/javenlib/images/bark/img2.ppm')
# sift = cv2.SIFT(600)
# imga_kp_sift_obj = sift.detect(imga)
# print len(imga_kp_sift_obj)
# # print type(imga_kp_sift_obj),imga_kp_sift_obj[0].pt,imga_kp_sift_obj[0].pt[1],round(imga_kp_sift_obj[0].pt[0]),int(round(imga_kp_sift_obj[0].pt[1]))
# kp_list=javenlib_tf.KeyPoint_from_siftObjList_to_4dlist(imga_kp_sift_obj)
# print len(kp_list)
# kp_list_afterNMS = javenlib_tf.NonMaxSuppresion_4_kp_set(kp_list,threshold=100)
# print kp_list_afterNMS
# print len(kp_list_afterNMS)
# javenlib_tf.show_kp_set_listformat_FromDifOctave(imga,kp_list_afterNMS[0:250])
# # kp_list_afterNMS_chosen = javenlib_tf.choose_kp_from_list(kp_list_afterNMS,quantity_to_choose=250)
# # print len(kp_list_afterNMS_chosen)
# # print kp_list_afterNMS_chosen
# # for k in range(len(kp_list_afterNMS)):
# #     print kp_list_afterNMS[k]
# # javenlib_tf.show_kp_set_listformat_FromDifOctave(imga,kp_list_afterNMS)
# # imga_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj)
# # # imga_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_sift)
# # imga_kp_sift_obj,imga_kp_sift_des = sift.compute(imga,imga_kp_sift_obj)

degree = 30
row_num=43
column_num=43
radian = 1.0*degree/180.0*cmath.pi
#	print 'radian:',radian
rotate_matrix = np.array([[cmath.cos(radian),-cmath.sin(radian),-0.5*(column_num-1)*cmath.cos(radian)+0.5*(row_num-1)*cmath.sin(radian)+0.5*(column_num-1)],
							  [cmath.sin(radian),cmath.cos(radian),-0.5*(column_num-1)*cmath.sin(radian)-0.5*(row_num-1)*cmath.cos(radian)+0.5*(row_num-1)],
							  [0,0,1]]).real
old_position=np.array([[10],[10],[1]])
new_position = np.dot(rotate_matrix,old_position)
rotate_matrix_reverse = np.linalg.inv(rotate_matrix)
print rotate_matrix
print new_position
print rotate_matrix_reverse
x = np.dot(rotate_matrix,rotate_matrix_reverse)
print x
