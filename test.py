#!/usr/bin/python
#encoding=utf-8 
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib
import cmath

#img_cv2 = cv2.imread('/home/javen/javenlib/images/cat.jpg') #360*480*3
#img_caffe = caffe.io.load_image('/home/javen/javenlib/images/cat.jpg')
#row_num = img_cv2.shape[0]
#column_num = img_cv2.shape[1]
#print row_num,column_num
#orb = cv2.ORB(500)
#kp = orb.detect(img_cv2)
##sift = cv2.SIFT(500)
##kp = sift.detect(img_cv2)
#kp = javenlib.KeyPoint_convert_forOpencv2(kp)
##剔除掉取方块会越界的kp
#for i in range(len(kp)-1,-1,-1):
#	if kp[i,1]<21 or kp[i,1]>row_num-1-21 or kp[i,0]<21 or kp[i,0]>column_num-1-21:
#		kp = np.delete(kp,i,0)
#print kp.shape
#kp_des = javenlib.lenet5_compute(img_caffe,kp)
#print kp_des[499],kp_des.shape

img1_cv = cv2.imread('/home/javen/javenlib/images/boat/img1.pgm')
img1_caffe = caffe.io.load_image('/home/javen/javenlib/images/boat/img1.pgm')
img2_cv = cv2.imread('/home/javen/javenlib/images/boat/img2.pgm')
img2_caffe = caffe.io.load_image('/home/javen/javenlib/images/boat/img2.pgm')
print img1_cv.shape,'\n',img1_caffe.shape
print img1_cv[1,1,:],img1_caffe[1,1,:]
orb = cv2.ORB(500)
kp = orb.detect(img1_cv)
kp,des = orb.compute(img1_cv,kp)
print des.shape
#a = np.zeros((43,43,3),dtype='uint8')
#for i in range(0,43):
#	for j in range(0,43):
#		a[i,j,:] = 255
#a = javenlib.area_set_zero(a)
#cv2.imshow('img',a)
#cv2.waitKey(0)
#cv2.destroyAllwindows()
