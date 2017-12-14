#!/usr/bin/python
#encoding=utf-8 
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import javenlib
import cmath
import pyflann


flann = pyflann.FLANN()
origin_data = np.array([5.,7,9,10,4,8,6,3,15,11]).reshape(10,1)
test_data = np.arange(10.).reshape(10,1)
test_data[0]=30
print origin_data
print test_data
print origin_data[0].dtype
result,dists = flann.nn(origin_data,test_data,1,algorithm="kmeans",branching=32, iterations=7, checks=16)
print 'result:',result
print 'dists:',dists
x = np.zeros((10,3))
x[:,0] = dists
x[:,1] = result
x[:,2] = np.arange(10)
print x
print np.argsort(x[:,0])
print flann
print 'hello world!'
