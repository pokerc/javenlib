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
origin_data = np.arange(12.).reshape(12,1)
test_data = np.arange(10.).reshape(10,1)
print origin_data[0].dtype
result,dists = flann.nn(origin_data,test_data,5,algorithm="kmeans",branching=32, iterations=7, checks=16)
print 'result:',result
print 'dists:',dists
print flann
print 'hello world!'
