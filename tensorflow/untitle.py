#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf
import cmath



mask = javenlib_tf.get_guassian_mask_2d(sigma=2)
img = plt.imread('/home/javen/javenlib/images/lena/lena.tif')/255.
javenlib_tf.guassian_conv_2d(img,mask)


# mask_1d = javenlib_tf.get_guassian_mask_1d(sigma=2)
# img = plt.imread('/home/javen/javenlib/images/lena/lena.tif')/255.
# javenlib_tf.guassian_conv_1d(img,mask_1d)