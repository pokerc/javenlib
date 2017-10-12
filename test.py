#!/usr/bin/python
#encoding=utf-8 

import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib
import cmath

a = np.zeros((43,43,3))
#for i in range(0,43):
#	a[:,i,:] = i
a[21,21,:] = 100
a[22,22,:] = 100
a[22,21,:] = 100

javenlib.ph()
javenlib.get_center_direction(a)
print cmath.pi,cmath.atan(-1).real/cmath.pi*180
#javenlib.lenet5_compute(1,1)
##javenlib.convert_meanvalue()
#img_caffe = caffe.io.load_image('/home/javen/javenlib/images/cat.jpg')
#img_cv2 = cv2.imread('/home/javen/javenlib/images/cat.jpg')
##b,g,r = cv2.split(img_cv2)
##plt.imshow(img_caffe[:,:,:])
##img_cv2[:,:,0] = 0
##img_cv2[:,:,1] = 0
#img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_BGR2GRAY)
#print 'img_cv2 shape:',img_cv2.shape
#img_caffe[:,:,0] = 0
#img_caffe[:,:,2] = 0
##plt.imshow(img_cv2)
##plt.show()


#cv2.imshow('image',img_cv2)
#cv2.waitKey(0)
#cv2.destroyAllwindows()
