#!/usr/bin/python
#encoding=utf-8 

import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib

javenlib.ph()
javenlib.lenet5_compute(1,1)
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
