#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann

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

def get_kp_set_raw(img_path_list):
	"""
	获取输入的几幅图的kp点的集合,每幅图使用sift检测400个kp
	:param img_path_list: 输入的图像的path列表
	:return: 返回kp点的集合
	"""
	# img_path_list = ['/home/javen/javenlib/images/leuven/img1.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img2.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img3.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img4.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img5.ppm']
	number_to_detect = 400
	sift = cv2.SIFT(number_to_detect)
	kp_set_raw = np.zeros(shape=(len(img_path_list),number_to_detect,2),dtype=np.int)
	for img_count in range(len(img_path_list)):
		img = plt.imread(img_path_list[img_count])
		kp_object = sift.detect(img)
		kp_coordinate = KeyPoint_convert_forOpencv2(kp_object)
		while len(kp_coordinate)>400:
			kp_coordinate = np.delete(kp_coordinate,-1,axis=0)
		# print 'kp_coordinate:',kp_coordinate.shape
		kp_set_raw[img_count] = np.copy(kp_coordinate)
	# print 'kp_set_raw:',kp_set_raw.shape,kp_set_raw.dtype
	return kp_set_raw

def get_kp_set_positive(kp_set_raw):
	"""
	使用flann的方法,寻找出kp_set_raw中具有重复性的kp点
	:param kp_set_raw: 输入sift检测出的原始kp点集
	:return: 返回具有可重复性的kp点集
	"""
	print 'kp_set_raw:',kp_set_raw.shape
	flann = pyflann.FLANN()
	origin_data = kp_set_raw[0] #(400,2)
	test_data = kp_set_raw[1] #(400,2)
	matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 1, algorithm="kmeans", branching=32, iterations=7, checks=16)
	print 'matched_indices:',matched_indices.shape,'matched_distances:',matched_distances.shape
	print matched_distances
	count = 0
	for i in range(400):
		if matched_distances[i] < 20:
			count += 1
			print matched_distances[i]
	print 'count:',count


img_path_list = ['/home/javen/javenlib/images/leuven/img1.ppm',
		 '/home/javen/javenlib/images/leuven/img2.ppm',
		 '/home/javen/javenlib/images/leuven/img3.ppm',
		 '/home/javen/javenlib/images/leuven/img4.ppm',
		 '/home/javen/javenlib/images/leuven/img5.ppm']
kp_set_raw = get_kp_set_raw(img_path_list)
get_kp_set_positive(kp_set_raw)
