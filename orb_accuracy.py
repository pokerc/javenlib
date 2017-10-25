#!/usr/bin/python
#encoding=utf-8 
import io
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib
import cmath

img1_cv = cv2.imread('/home/javen/javenlib/images/graf/img1.ppm')
img2_cv = cv2.imread('/home/javen/javenlib/images/graf/img1_rotate90.ppm')
degree = -90
row_num = img1_cv.shape[0]
column_num = img1_cv.shape[1]
radian = 1.0*degree/180.0*cmath.pi
rotation_matrix = np.array([[cmath.cos(radian),-cmath.sin(radian),-0.5*(column_num-1)*cmath.cos(radian)+0.5*(row_num-1)*cmath.sin(radian)+0.5*(column_num-1)],
							  [cmath.sin(radian),cmath.cos(radian),-0.5*(column_num-1)*cmath.sin(radian)-0.5*(row_num-1)*cmath.cos(radian)+0.5*(row_num-1)],
							  [0,0,1]]).real
#rotation_matrix = javenlib.get_matrix_from_file('/home/javen/javenlib/images/wall/H1to6p')
print rotation_matrix

orb = cv2.ORB(1200)
img1_kp_pos = orb.detect(img1_cv)
img1_kp_pos,img1_kp_des = orb.compute(img1_cv,img1_kp_pos)
img1_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img1_kp_pos)
img2_kp_pos = orb.detect(img2_cv)
img2_kp_pos,img2_kp_des = orb.compute(img2_cv,img2_kp_pos)
img2_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img2_kp_pos)
print img1_kp_des.shape,img2_kp_des.shape,img1_kp_pos.shape,img2_kp_pos.shape

a1 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
a2 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
a3 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
print a1,a2,a3,(a1+a2+a3)/3.










