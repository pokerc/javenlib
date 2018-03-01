#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
# import javenlib_tf
import cmath

a=np.zeros((11,11))
print a
a[5,:]=1
a[:,5]=1
print a
b = np.delete(a,5,axis=0)
b = np.delete(b,5,axis=1)
print b.shape
print b