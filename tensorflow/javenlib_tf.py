#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def image_resize(image,rows,columns,toGray=True):
    new_image = tf.image.resize_images(image,[rows,columns],method=1)
    new_image = tf.to_float(tf.image.rgb_to_grayscale(new_image))
    return new_image

def KeyPoint_convert_forOpencv2(keypoints):
	length = len(keypoints)
	points2f = np.zeros((length,2))
	for i in range(0,length):
		points2f[i,:] = keypoints[i].pt
	points = np.array(np.around(points2f),dtype='int')
	return points
