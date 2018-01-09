#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf
import cmath



mask = javenlib_tf.get_guassian_mask(sigma=2)
img = plt.imread('/home/javen/javenlib/images/lena/lena.tif')/255.
print img
javenlib_tf.guassian_conv_2d(img,mask)