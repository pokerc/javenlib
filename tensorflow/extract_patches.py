#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann
import javenlib_tf

image_classname = 'leuven'
#提取图片集合中的positive patches和negative patches
img_path_list = ['/home/javen/javenlib/images/'+image_classname+'/img1.ppm',
		 '/home/javen/javenlib/images/'+image_classname+'/img2.ppm',
		 '/home/javen/javenlib/images/'+image_classname+'/img3.ppm',
		 '/home/javen/javenlib/images/'+image_classname+'/img4.ppm',
		 '/home/javen/javenlib/images/'+image_classname+'/img5.ppm']

kp_set_raw = javenlib_tf.get_kp_set_raw(img_path_list)
kp_set_positive = javenlib_tf.get_kp_set_positive(kp_set_raw)
print 'kp_set_positive:',kp_set_positive.shape
kp_patch_set_positive = javenlib_tf.get_kp_patch_set_positive(img_path_list,kp_set_positive)
print 'kp_patch_set_positive:',kp_patch_set_positive.shape
# javenlib_tf.show_patch_set(kp_patch_set_positive)
# javenlib_tf.show_kp_set(img_path_list[4],kp_set_positive)
#保存positive patch集合
np.save('/home/javen/javenlib/tensorflow/TILDE_data/'+image_classname+'_positive_patches.npy',kp_patch_set_positive)

kp_set_negative = javenlib_tf.get_kp_set_negative(kp_set_raw)
print 'kp_set_nagative:',kp_set_negative.shape
kp_patch_set_negative = javenlib_tf.get_kp_patch_set_negative(img_path_list,kp_set_negative)
print 'kp_patch_set_negative:',kp_patch_set_negative.shape
# javenlib_tf.show_patch_set(kp_patch_set_negative)
#保存negative patch集合
np.save('/home/javen/javenlib/tensorflow/TILDE_data/'+image_classname+'_negative_patches.npy',kp_patch_set_negative)
