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



imga = plt.imread('/home/javen/javenlib/images/bikes/img1.ppm')
returned_list = javenlib_tf.get_pyramid_of_image(imga)
imgb = cv2.pyrDown(imga)
imgc = cv2.pyrDown(imgb)
print imgb.shape,imgc.shape

orb = cv2.ORB(250)
kp = orb.detect(imga)
kp,des = orb.compute(imga,kp)
print len(kp),kp[0].pt,kp[0].size,kp[0].angle,kp[0].response,kp[0].octave,kp[0].class_id
for i in range(10):
    print kp[i].pt, kp[i].size, kp[i].angle, kp[i].response, kp[i].octave, kp[i].class_id
print des.shape,des[0]

patch1 = np.copy(imga[413-16:413+17,530-16:530+17])
print patch1.shape
angle = javenlib_tf.get_patch_angle(patch1)
print angle