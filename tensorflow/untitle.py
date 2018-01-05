import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf

a = plt.imread('/home/javen/datasets/EFDataset/rushmore/test/image_color/img1.png')
img_path = '/home/javen/datasets/EFDataset/rushmore/test/image_color/img1.png'
# plt.imshow(a)
# plt.show()
print a.shape,a
a = np.copy(a*255.)
b = a.astype(np.uint8)
print b.shape,b
# plt.imshow(b)
# plt.show()
sift = cv2.SIFT(100)
kp = sift.detect(b)
kp = javenlib_tf.KeyPoint_convert_forOpencv2(kp)
javenlib_tf.show_kp_set('/home/javen/datasets/EFDataset/rushmore/test/image_color/img1.png',kp,pixel_size=5)
# b = plt.imread('/home/javen/javenlib/images/wall/img1.ppm')
# print b.shape,b