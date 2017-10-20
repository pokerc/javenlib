#!/usr/bin/python
#encoding=utf-8 
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib
import cmath

filename1 = '/home/javen/javenlib/images/wall/img1.ppm'
filename2 = '/home/javen/javenlib/images/wall/img6.ppm'
degree = -90
rotation_filename = '/home/javen/javenlib/images/wall/H1to6p'
rotation_matrix = javenlib.get_matrix_from_file(rotation_filename)
detect_method = 'SIFTd'
layer = 'ip1'

img1_cv = cv2.imread(filename1)
img1_caffe = caffe.io.load_image(filename1)
img2_cv = cv2.imread(filename2)
img2_caffe = caffe.io.load_image(filename2)

row_num = img2_cv.shape[0]
column_num = img2_cv.shape[1]
#radian = 1.0*degree/180.0*cmath.pi
#rotation_matrix = np.array([[cmath.cos(radian),-cmath.sin(radian),-0.5*(column_num-1)*cmath.cos(radian)+0.5*(row_num-1)*cmath.sin(radian)+0.5*(column_num-1)],
#							  [cmath.sin(radian),cmath.cos(radian),-0.5*(column_num-1)*cmath.sin(radian)-0.5*(row_num-1)*cmath.cos(radian)+0.5*(row_num-1)],
#							  [0,0,1]]).real

orb = cv2.ORB(1000)
sift = cv2.SIFT(1000)

if detect_method == 'ORBd':
	img1_kp_pos = orb.detect(img1_cv)
	img1_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img1_kp_pos)
	img1_kp_des = javenlib.lenet5_compute(img1_caffe,img1_kp_pos,layer_name=layer)
	img2_kp_pos = orb.detect(img2_cv)
	img2_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img2_kp_pos)
	img2_kp_des = javenlib.lenet5_compute(img2_caffe,img2_kp_pos,layer_name=layer)
	print 'ORBd accuracy:'
	a1 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
	a2 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
	a3 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
	print a1,a2,a3,(a1+a2+a3)/3.
	print filename2
if detect_method == 'SIFTd':
	img1_kp_pos = sift.detect(img1_cv)
	img1_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img1_kp_pos)
	for i in range(len(img1_kp_pos)-1,-1,-1):
		if img1_kp_pos[i,1] < 21 or img1_kp_pos[i,1] > row_num-1-21 or img1_kp_pos[i,0] < 21 or img1_kp_pos[i,0] > column_num-1-21:
			img1_kp_pos = np.delete(img1_kp_pos,i,0)
#	print 'img1_kp_pos',img1_kp_pos.shape
	img1_kp_des = javenlib.lenet5_compute(img1_caffe,img1_kp_pos,layer_name=layer)
	img2_kp_pos = sift.detect(img2_cv)
	img2_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img2_kp_pos)
	for i in range(len(img2_kp_pos)-1,-1,-1):
			if img2_kp_pos[i,1] < 21 or img2_kp_pos[i,1] > row_num-1-21 or img2_kp_pos[i,0] < 21 or img2_kp_pos[i,0] > column_num-1-21:
				img2_kp_pos = np.delete(img2_kp_pos,i,0)
	img2_kp_des = javenlib.lenet5_compute(img2_caffe,img2_kp_pos,layer_name=layer)
	print 'SIFTd accuracy:'
	a1 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
	a2 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
	a3 = javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
	print a1,a2,a3,(a1+a2+a3)/3.
	print filename2
