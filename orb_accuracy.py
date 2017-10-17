#!/usr/bin/python
#encoding=utf-8 
import io
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib
import cmath

img1_cv = cv2.imread('/home/javen/javenlib/images/boat/img1.pgm')
img2_cv = cv2.imread('/home/javen/javenlib/images/boat/img6.pgm')
rotation_matrix = javenlib.get_matrix_from_file('/home/javen/javenlib/images/boat/H1to6p')

orb = cv2.ORB(1000)
img1_kp_pos = orb.detect(img1_cv)
img1_kp_pos,img1_kp_des = orb.compute(img1_cv,img1_kp_pos)
img1_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img1_kp_pos)
img2_kp_pos = orb.detect(img2_cv)
img2_kp_pos,img2_kp_des = orb.compute(img2_cv,img2_kp_pos)
img2_kp_pos = javenlib.KeyPoint_convert_forOpencv2(img2_kp_pos)
print img1_kp_des.shape,img2_kp_des.shape,img1_kp_pos.shape,img2_kp_pos.shape

javenlib.match_accuracy(img1_cv,img1_kp_pos,img1_kp_des,img2_cv,img2_kp_pos,img2_kp_des,rotation_matrix)











