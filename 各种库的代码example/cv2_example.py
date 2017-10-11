#!/usr/bin/python
#encoding=utf-8 
import numpy as np
import cv2
import caffe
import matplotlib.pyplot as plt
import sys
import os
import javenlib

print 'this cv2 example code!'
img = cv2.imread('/home/javen/javenlib/images/cat.jpg')
print img.shape
sift = cv2.SIFT(500)
pos = sift.detect(img)
pos,des = sift.compute(img,pos)
print des.shape 
print pos[0].pt 
print len(pos)
x = javenlib.KeyPoint_convert_forOpencv2(pos)
print x.shape,'\n'

surf = cv2.SURF()
pos = surf.detect(img)
pos,des = surf.compute(img,pos)
print des.shape 
print pos[0].pt 
print len(pos)
x = javenlib.KeyPoint_convert_forOpencv2(pos)
print x.shape,img.shape
print np.amax(x[:,1])


#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

