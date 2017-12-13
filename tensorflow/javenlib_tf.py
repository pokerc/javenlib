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

def show_patch_set(patch_set):
	"""
	显示patch集中的图像,一幅接一幅，时间间隔为0.5s，图像为rgb图
	:param patch_set: 要显示的patch集
	:return:
	"""
	plt.ion()
	for i in range(len(patch_set)):
		plt.figure()
		plt.imshow(patch_set[i])
		plt.pause(0.5)
		plt.close()
	plt.ioff()

def show_patch_set_gray(patch_set):
	"""
	显示patch集中的图像,一幅接一幅，时间间隔为0.5s,图像为灰度图
	:param patch_set: 要显示的patch集
	:return:
	"""
	plt.ion()
	for i in range(len(patch_set)):
		plt.figure()
		plt.imshow(patch_set[i].reshape(64,64),cmap='gray')
		plt.pause(0.5)
		plt.close()
	plt.ioff()