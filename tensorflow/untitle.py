import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf

a = plt.imread('/home/javen/javenlib/images/leuven/img1.ppm')
b = plt.imread('/home/javen/javenlib/images/leuven/img2.ppm')
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(a)
plt.subplot(2,2,2)
plt.imshow(b)
plt.subplot(2,2,3)
plt.imshow(a)
plt.subplot(2,2,4)
plt.imshow(c)
plt.show()
print a.mean()-b.mean()