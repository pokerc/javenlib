#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann
import javenlib_tf

image_classname = 'amosSchool'
image_type = 'jpg'
#提取图片集合中的positive patches和negative patches
img_path_list = ['/home/javen/javenlib/images/'+image_classname+'/img1.'+image_type,
		 '/home/javen/javenlib/images/'+image_classname+'/img2.'+image_type,
		 '/home/javen/javenlib/images/'+image_classname+'/img3.'+image_type,
		 '/home/javen/javenlib/images/'+image_classname+'/img4.'+image_type,
		 '/home/javen/javenlib/images/'+image_classname+'/img5.'+image_type]

kp_set_raw = javenlib_tf.get_kp_set_raw(img_path_list)

# kp_set_positive = javenlib_tf.get_kp_set_positive(kp_set_raw)
# print 'kp_set_positive:',kp_set_positive.shape
# kp_patch_set_positive = javenlib_tf.get_kp_patch_set_positive(img_path_list,kp_set_positive)
# print 'kp_patch_set_positive:',kp_patch_set_positive.shape
# javenlib_tf.show_patch_set(kp_patch_set_positive)
# javenlib_tf.show_kp_set(img_path_list[4],kp_set_positive,pixel_size=10)
#保存positive patch集合
# np.save('/home/javen/javenlib/tensorflow/TILDE_data/'+image_classname+'_positive_patches.npy',kp_patch_set_positive)

kp_set_negative = javenlib_tf.get_kp_set_negative(kp_set_raw,dist_threshold=2000)
print 'kp_set_nagative:',kp_set_negative.shape
kp_patch_set_negative = javenlib_tf.get_kp_patch_set_negative(img_path_list,kp_set_negative)
print 'kp_patch_set_negative:',kp_patch_set_negative.shape
javenlib_tf.show_patch_set(kp_patch_set_negative)
# javenlib_tf.show_kp_set(img_path_list[4],kp_set_negative,pixel_size=10)
#保存negative patch集合
# np.save('/home/javen/javenlib/tensorflow/TILDE_data/'+image_classname+'_negative_patches.npy',kp_patch_set_negative)

# new_test_data = np.copy(kp_set_negative)
# flann = pyflann.FLANN()
# for i in range(len(kp_set_negative)):
# 	origin_data = np.copy(new_test_data)
# 	test_data = np.copy(new_test_data)
# 	matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 2,
# 												  algorithm="kmeans", branching=32, iterations=7, checks=16)
# 	for j in range(len(test_data)-1,-1,-1):
# 		if matched_distances[j,1] < 1000:
# 			new_test_data = np.delete(test_data,j,axis=0)
# 			break
# print 'matched_indices:',matched_indices
# print 'matched_distances:',matched_distances
# print new_test_data
# print new_test_data.shape
# javenlib_tf.show_kp_set(img_path_list[4],new_test_data,pixel_size=10)