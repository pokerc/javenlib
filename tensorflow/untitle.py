#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf
import cmath
from tensorflow.examples.tutorials.mnist import input_data



# imga = plt.imread('/home/javen/javenlib/images/graf_rotate/img1.ppm')[320-260:320+260,400-260:400+260]
# plt.imshow(imga)
# # plt.show()
# plt.imsave('/home/javen/javenlib/images/graf_rotate/img1_0.jpg',imga,format='jpg')



imga = plt.imread('/home/javen/javenlib/images/EF_Dataset/lab580/img3.jpg')
returned_list = javenlib_tf.get_pyramid_of_image(imga)
imgb = cv2.pyrDown(imga)
imgc = cv2.pyrDown(imgb)
print imgb.shape,imgc.shape
# plt.imshow(imgc)
# plt.show()
plt.imsave('/home/javen/javenlib/images/EF_Dataset/lab580/img3_.jpg',imgc,format='jpg')