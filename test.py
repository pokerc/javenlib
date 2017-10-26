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

#img = cv2.imread('/home/javen/javenlib/images/graf/img1.ppm')
#img_caffe = caffe.io.load_image('/home/javen/javenlib/images/graf/img1.ppm')
#print img.shape
#area_43 = np.copy(img_caffe[300-21:300+21+1,300-21:300+21+1,:])
##img[300-21:300+21+1,300-21:300+21+1,:] = 255
#degree = javenlib.get_center_direction(javenlib.area_set_zero(area_43))
#rotated_outter_square = javenlib.image_rotate(area_43,-1*degree)
#x = np.copy(rotated_outter_square)
#for i in range(1,42):
#	for j in range(1,42):
#		if x[i,j,0] == 0:
#			x[i,j,0] = 0.25*(x[i-1,j,0]+x[i+1,j,0]+x[i,j-1,0]+x[i,j+1,0])
#			x[i,j,1] = 0.25*(x[i-1,j,1]+x[i+1,j,1]+x[i,j-1,1]+x[i,j+1,1])
#			x[i,j,2] = 0.25*(x[i-1,j,2]+x[i+1,j,2]+x[i,j-1,2]+x[i,j+1,2])
#print rotated_outter_square
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.imshow('area_43',area_43)
#cv2.waitKey(0)
#cv2.imshow('rotated',rotated_outter_square)
#cv2.waitKey(0)
#cv2.imshow('x',x)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img1_cv = cv2.imread('/home/javen/javenlib/images/boat/img1.pgm')
#img1_caffe = caffe.io.load_image('/home/javen/javenlib/images/boat/img1.pgm')
#img2_cv = cv2.imread('/home/javen/javenlib/images/boat/img2.pgm')
#img2_caffe = caffe.io.load_image('/home/javen/javenlib/images/boat/img2.pgm')
#print img1_cv.shape,'\n',img1_caffe.shape
#print img1_cv[1,1,:],img1_caffe[1,1,:]
#orb = cv2.ORB(500)
#kp = orb.detect(img1_cv)
#kp,des = orb.compute(img1_cv,kp)
#print des.shape
#a = np.zeros((43,43,3),dtype='uint8')
#for i in range(0,43):
#	for j in range(0,43):
#		a[i,j,:] = 255
#a = javenlib.area_set_zero(a)
#cv2.imshow('img',a)
#cv2.waitKey(0)
#cv2.destroyAllwindows()



img1 = cv2.imread('/home/javen/javenlib/images/graf/img1.ppm')
img2 = cv2.imread('/home/javen/javenlib/images/graf/img1_rotate10.ppm')
row_num = img1.shape[0]
column_num = img1.shape[1]
stitched_img = np.zeros((row_num,column_num*2,3),dtype='uint8')
stitched_img[0:row_num,0:column_num,:] = np.copy(img1)
stitched_img[0:row_num,column_num:,:] = np.copy(img2)
cv2.imshow('img',stitched_img)
cv2.waitKey(0)
cv2.destroyAllwindows()






























