#!/usr/bin/python
#encoding=utf-8 
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib
import cmath

filename1 = '/home/javen/javenlib/images/boat/img1.pgm'
filename2 = '/home/javen/javenlib/images/boat/img2.pgm'
rotation_filename = '/home/javen/javenlib/images/boat/H1to2p'

detect_method = 'ORBd'

img1_cv = cv2.imread(filename1)
img1_caffe = caffe.io.load_image(filename1)
img2_cv = cv2.imread(filename2)
img2_caffe = caffe.io.load_image(filename2)
rotation_matrix = javenlib.get_matrix_from_file(rotation_filename)
orb = cv2.ORB(1000)
sift = cv2.SIFT(1000)

if detect_method == 'ORBd':
	img1_kp_pos = orb.detect(img1_cv)
	img1_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img1_kp_pos)
	img1_kp_des = javenlib.lenet5_compute(img1_caffe,img1_kp_pos)
	img2_kp_pos = orb.detect(img2_cv)
	img2_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img2_kp_pos)
	img2_kp_des = javenlib.lenet5_compute(img2_caffe,img2_kp_pos)
	print 'ORBd accuracy:'
	javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
if detect_method == 'SIFTd':
	img1_kp_pos = sift.detect(img1_cv)
	img1_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img1_kp_pos)
	img1_kp_des = javenlib.lenet5_compute(img1_caffe,img1_kp_pos)
	img2_kp_pos = sift.detect(img2_cv)
	img2_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img2_kp_pos)
	img2_kp_des = javenlib.lenet5_compute(img2_caffe,img2_kp_pos)
	print 'SIFTd accuracy:'
	javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)
	
