#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf
import cmath



mask = javenlib_tf.get_guassian_mask_1d(sigma=8)
img = plt.imread('/home/javen/javenlib/images/lena/lena.tif')/255.
# img_afterconv = javenlib_tf.guassian_conv_1d(img,mask)
# print img_afterconv.shape
# img_downsample = np.copy(img_afterconv[::2,::2])
# print img_downsample.shape
# plt.figure(1)
# plt.imshow(img_afterconv)
# plt.figure(2)
# plt.imshow(img_downsample)
# plt.show()
img_down1 = cv2.pyrDown(img)
img_down2 = cv2.pyrDown(img_down1)
img_up1 = cv2.pyrUp(img)
img_up2 = cv2.pyrUp(img_up1)
print img_up2.shape,img_up1.shape,img.shape,img_down1.shape,img_down2.shape
plt.subplot(1,3,1)
plt.imshow(img_up2)
plt.subplot(1,3,2)
plt.imshow(img_up1)
plt.subplot(1,3,3)
plt.imshow(img)
plt.show()


# mask_1d = javenlib_tf.get_guassian_mask_1d(sigma=2)
# img = plt.imread('/home/javen/javenlib/images/lena/lena.tif')/255.
# javenlib_tf.guassian_conv_1d(img,mask_1d)