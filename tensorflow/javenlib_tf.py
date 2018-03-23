#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann
import time
import types
import cmath
from tensorflow.examples.tutorials.mnist import input_data

######################来自原javenlib.py里面的几个函数,主要是求patch重心angle的几个函数##########################

#将43*43的gray image进行外方形与内切圆中间区域的置零操作的函数
def middle_area_set_zero(img):
	#该函数输入为43*43的图像矩阵，函数功能是将外部方形和其内切圆之间的区域置零后输出新的外部方形
	img_inner_circle = np.copy(img)
	for i in range(0,43):
		for j in range(0,43):
			if (i-21)**2+(j-21)**2 > (0.5*29*cmath.sqrt(2).real)**2:
				img_inner_circle[i,j] = 0
	return img_inner_circle

#对进行置零之后的43*43的patch进行求重心angle的操作的函数
def get_patch_angle(img):
	#该函数得到的角度是图像重心与中心的连线偏离x轴的角度，后续进行逆向旋转的时候请注意是否需要加上负号作为旋转函数的参数
	row_num = img.shape[0]
	column_num = img.shape[1]
#	print row_num,column_num,len(img.shape)
	if len(img.shape) == 3:
		img_mean = np.copy(img.mean(2))
	else:
		img_mean = np.copy(img)
#	print img_mean.shape
	sum_x = 0
	sum_y = 0
	sum_image = 0
	for i in range(0,row_num):
		for j in range(0,column_num):
			sum_x += img_mean[i,j]*j
			sum_y += img_mean[i,j]*i
			sum_image += img_mean[i,j]
	center_x = 1.0*sum_x/sum_image
	center_y = 1.0*sum_y/sum_image
#	print center_x,center_y
	delta_x = center_x-(column_num-1)/2.0
	delta_y = center_y-(row_num-1)/2.0
	if delta_x != 0:
		radian = cmath.atan(1.0*delta_y/delta_x)
	if delta_x == 0 and delta_y == 0:
		degree = 0
	elif delta_x ==0 and delta_y > 0:
		degree = 90
	elif delta_x ==0 and delta_y < 0:
		degree = -90
	elif delta_x < 0 and delta_y < 0:
		degree = -180+radian/cmath.pi*180.0
	elif delta_x < 0 and delta_y >= 0:
		degree = 180+radian/cmath.pi*180.0
	else:    #delta_x > 0
		degree = radian/cmath.pi*180.0
	degree = degree.real
#	print 'center degree:',degree
	return degree

#根据计算出的angle,对已经进行置零后的43*43的patch进行旋转操作的函数,方便后续取出里面的29*29的方形patch
def image_rotate(img,degree):
	#图像旋转函数，degree若大于0则表示顺时针旋转，反之表示逆时针旋转
	#使用逆矩阵的方法,由new_position反向去求在原图中的位置,然后取到新图中去,这样可以避免在新图中出现黑色没有赋值到的空点
	row_num = img.shape[0]
	column_num = img.shape[1]
	radian = 1.0*degree/180.0*cmath.pi
#	print 'radian:',radian
	rotate_matrix = np.array([[cmath.cos(radian),-cmath.sin(radian),-0.5*(column_num-1)*cmath.cos(radian)+0.5*(row_num-1)*cmath.sin(radian)+0.5*(column_num-1)],
							  [cmath.sin(radian),cmath.cos(radian),-0.5*(column_num-1)*cmath.sin(radian)-0.5*(row_num-1)*cmath.cos(radian)+0.5*(row_num-1)],
							  [0,0,1]]).real
	rotate_matrix_reverse = np.linalg.inv(rotate_matrix)
	if len(img.shape) == 3:
		rotated_image = np.zeros((row_num, column_num, 3))
		for i in range(row_num):
			for j in range(column_num):
				new_position = np.array([[j],[i],[1]])
				old_position_mapped = np.dot(rotate_matrix_reverse,new_position)
				old_position_row = int(round(old_position_mapped[1]))
				old_position_column = int(round(old_position_mapped[0]))
				# print 'new_position:',new_position,'old_position_mapped:',old_position_mapped,old_position_row,old_position_column
				if old_position_row >= 0 and old_position_row < 43 and old_position_column >= 0 and old_position_column < 43:
					rotated_image[i,j,:] = np.copy(img[old_position_row,old_position_column,:])
	else:
		rotated_image = np.zeros((row_num, column_num))
		for i in range(row_num):
			for j in range(column_num):
				new_position = np.array([[j],[i],[1]])
				old_position_mapped = np.dot(rotate_matrix_reverse,new_position)
				old_position_row = int(round(old_position_mapped[1]))
				old_position_column = int(round(old_position_mapped[0]))
				# print 'new_position:',new_position,'old_position_mapped:',old_position_mapped,old_position_row,old_position_column
				if old_position_row >= 0 and old_position_row < 43 and old_position_column >= 0 and old_position_column < 43:
					rotated_image[i,j] = np.copy(img[old_position_row,old_position_column])
	return rotated_image


###################################################################################################

def get_guassian_mask_1d(sigma=0.6):
	"""
	计算一维高斯模板矩阵
	:param sigma:
	:return:
	"""
	mask_list = []
	for x in range(-2,3,1):
		f = (1. / cmath.sqrt(2 * cmath.pi * (sigma ** 2))) * cmath.exp(-1. * (x ** 2) / (2 * sigma ** 2))
		# print f.real
		mask_list.append([f.real])
	mask_array = np.zeros(shape=(5,1))
	for i in range(5):
		mask_array[i] = np.copy(mask_list[i])
	total = mask_array.sum()
	# print total
	# print mask_array
	mask_array = mask_array/total
	# print mask_array
	return mask_array

def guassian_conv_1d(img,mask):
	"""
	使用两次一维高斯卷积(先水平卷积,再竖直卷积)来计算二维高斯卷积操作,提高计算速度
	:param img: 待卷积模糊(平滑)的原始图像
	:param mask: 一维卷积模板矩阵
	:return: 返回经过卷积模糊(平滑)后的图像
	"""
	# print img.shape
	row_number = img.shape[0]
	column_number = img.shape[1]
	new_img = np.zeros(shape=(row_number + 4, column_number + 4, 3))
	new_img[2:-2, 2:-2, :] = np.copy(img)
	# print new_img.shape
	# print new_img[-4:,-4:,0]
	img_afterconv = np.zeros(shape=(row_number, column_number, 3))
	##首先在水平方向进行一维高斯卷积
	for x in range(2, row_number + 2):
		for y in range(2, column_number + 2):
			pixel_afterconv = np.zeros(shape=(3))
			for i in range(-2,3,1):
				pixel_afterconv += mask[i+2]*new_img[x,y+i]
			img_afterconv[x-2,y-2] = np.copy(pixel_afterconv)
	new_img[2:-2, 2:-2, :] = np.copy(img_afterconv)
	##然后在竖直方向进行一维高斯卷积
	for x in range(2, row_number + 2):
		for y in range(2, column_number + 2):
			pixel_afterconv = np.zeros(shape=(3))
			for j in range(-2,3,1):
				pixel_afterconv += mask[j+2]*new_img[x+j,y]
			img_afterconv[x-2,y-2] = np.copy(pixel_afterconv)
	# plt.subplot(1,2,1)
	# plt.imshow(img)
	# plt.subplot(1,2,2)
	# plt.imshow(img_afterconv)
	# plt.show()
	return img_afterconv

def get_guassian_mask_2d(sigma=0.6):
	"""
	计算高斯核
	:param sigma:高斯分布的标准差,标注差越大,钟型曲线越扁
	:return: 返回5*5的高斯模板矩阵
	"""
	mask_list = []
	for x in range(-2,3,1):
		for y in range(-2,3,1):
			f = (1. / (2 * cmath.pi * (sigma ** 2))) * cmath.exp(-1. * (x ** 2 + y ** 2) / (2 * sigma ** 2))
			mask_list.append([x,y,f.real])
	mask_array = np.zeros(shape=(25,3))
	for i in range(25):
		mask_array[i] = np.copy(mask_list[i])
	# print mask_array,mask_array[:,2].sum()
	total = mask_array[:,2].sum()
	mask_array[:,2] = mask_array[:,2]/total
	# print mask_array
	mask_0_6 = np.zeros(shape=(5,5))
	count = 0
	for i in range(5):
		for j in range(5):
			mask_0_6[i,j] = np.copy(mask_array[count,2])
			count += 1
	# print mask_0_6
	return mask_0_6

def guassian_conv_2d(img,mask):
	"""
	使用二维高斯模板矩阵对图像进行卷积模糊
	:param img: 输入图像,数据类型最好为0-1的float
	:param mask: 高斯模板矩阵
	:return: 返回经过卷积模糊之后的图像
	"""
	# print img.shape
	row_number = img.shape[0]
	column_number = img.shape[1]
	new_img = np.zeros(shape=(row_number+4,column_number+4,3))
	new_img[2:-2,2:-2,:] = np.copy(img)
	# print new_img.shape
	# print new_img[-4:,-4:,0]
	img_afterconv = np.zeros(shape=(row_number,column_number,3))
	for x in range(2,row_number+2):
		for y in range(2,column_number+2):
			pixel_afterconv = np.zeros(shape=(3))
			for i in range(-2,3,1):
				for j in range(-2,3,1):
					pixel_afterconv += mask[i+2,j+2]*new_img[x+i,y+j,:]
			img_afterconv[x-2,y-2,:] = np.copy(pixel_afterconv)
	# plt.subplot(1,2,1)
	# plt.imshow(img)
	# plt.subplot(1,2,2)
	# plt.imshow(img_afterconv)
	# plt.show()
	return img_afterconv

def image_resize(image,rows,columns,toGray=True):
	new_image = tf.image.resize_images(image,[rows,columns],method=1)
	new_image = tf.to_float(tf.image.rgb_to_grayscale(new_image))
	return new_image

def KeyPoint_convert_forOpencv2(keypoints):
	"""
	将list类型的kp object变量,解析为坐标矩阵,即坐标提取
	:param keypoints:
	:return:
	"""
	length = len(keypoints)
	points2f = np.zeros((length,2))
	for i in range(0,length):
		points2f[i,:] = keypoints[i].pt
	points = np.array(np.around(points2f),dtype='int')
	return points

def KeyPoint_reverse_convert_forOpencv2(keypoints):
	"""
	将坐标形式的kp转化为sift算法可用的obj形式的list,可作为后续的sift.compute()的参数
	:param keypoints:
	:return:
	"""
	kp_list = []
	for i in range(len(keypoints)):
		# print 'keypoints[i][0]:',keypoints[i][0],keypoints[i][1]
		kp_obj = cv2.KeyPoint(keypoints[i][0],keypoints[i][1],_size=3.58366942406)
		kp_list.append(kp_obj)
	return kp_list

def KeyPoint_reverse_convert_forORB(keypoints):
	kp_list = []
	for i in range(len(keypoints)):
		# print 'keypoints[i][0]:',keypoints[i][0],keypoints[i][1]
		if keypoints[i][4] < 0:
			kp_obj = cv2.KeyPoint(keypoints[i][0], keypoints[i][1], _size=31,_angle=360+keypoints[i][4],_response=keypoints[i][2],_octave=0,_class_id=-1)
		else:
			kp_obj = cv2.KeyPoint(keypoints[i][0], keypoints[i][1], _size=31, _angle=keypoints[i][4],_response=keypoints[i][2], _octave=0, _class_id=-1)
		kp_list.append(kp_obj)
	return kp_list

def KeyPoint_from_siftObjList_to_4dlist(kp_set_siftObjList):
	"""
	此函数用来将sift检测到的object list类型的kp_set转化为4维的list类型的kp_set
	:param kp_set_siftObjList:
	:return:
	"""
	total_count = len(kp_set_siftObjList)
	kp_set_4dlist=[]
	for i in range(total_count):
		kp_set_4dlist.append([int(round(kp_set_siftObjList[i].pt[0])), int(round(kp_set_siftObjList[i].pt[1])), kp_set_siftObjList[i].response,1])
	return kp_set_4dlist

def KeyPoint_from_4dlist_to_2darray(kp_list_4d):
	"""
	此函数用于将4维的list类型kp_set,转换为2维的array类型的kp_set,即仅保留坐标而去掉score和octave_index信息
	:param kp_list_4d:
	:return:
	"""
	total_count = len(kp_list_4d)
	kp_array_2d = np.zeros(shape=(total_count,2))
	for i in range(total_count):
		kp_array_2d[i] = np.copy(kp_list_4d[i][0:2]).astype(np.int)
	return kp_array_2d

def show_kp_set(img_path,kp_set,pixel_size=5):
	"""
	将特征点在图片上显示出来
	:param img_path: 图片的路径
	:param kp_set: 检测到的特征点的坐标集合
	:return:
	"""
	new_img = np.copy(plt.imread(img_path))
	for i in range(len(kp_set)):
		new_img[kp_set[i,1]-pixel_size:kp_set[i,1]+pixel_size,kp_set[i,0]-pixel_size:kp_set[i,0]+pixel_size,0] = 255
	plt.figure()
	plt.imshow(new_img)
	plt.show()

def show_kp_set_listformat(img,kp_list,pixel_size=5):
	"""
	将一个octave对应的kp在这个octave的img上显示出来,score越大,kp显示的颜色越深
	:param img:
	:param kp_list:
	:param pixel_size:
	:return:
	"""
	img_toshow = np.copy(img)
	for i in range(len(kp_list)):
		img_toshow[int(kp_list[i][1])-pixel_size:int(kp_list[i][1])+pixel_size,int(kp_list[i][0])-pixel_size:int(kp_list[i][0])+pixel_size,0]=int(255*kp_list[i][2]/1.4)
	plt.figure()
	plt.imshow(img_toshow)
	plt.show()

def show_kp_set_listformat_FromDifOctave(img,kp_list,pixel_size=5):
	"""
	用不同的颜色在original图中显示来自3个不同octave的kp
	:param img:
	:param kp_list:
	:param pixel_size:
	:return:
	"""
	img_toshow = np.copy(img)
	for i in range(len(kp_list)):
		if kp_list[i][3] == 0:
			img_toshow[int(kp_list[i][1]) - pixel_size:int(kp_list[i][1]) + pixel_size,int(kp_list[i][0]) - pixel_size:int(kp_list[i][0]) + pixel_size, 1] = 255#int(255 * kp_list[i][2] / 1.4)
		elif kp_list[i][3] == 1:
			img_toshow[int(kp_list[i][1]) - pixel_size:int(kp_list[i][1]) + pixel_size,int(kp_list[i][0]) - pixel_size:int(kp_list[i][0]) + pixel_size, 0] = 255#int(255 * kp_list[i][2] / 1.4)
		elif kp_list[i][3] == 2:
			img_toshow[int(kp_list[i][1]) - pixel_size:int(kp_list[i][1]) + pixel_size,int(kp_list[i][0]) - pixel_size:int(kp_list[i][0]) + pixel_size, :] = 255#int(255 * kp_list[i][2] / 1.4)
		# elif kp_list[i][3] == 3:
		# 	img_toshow[int(kp_list[i][1]) - pixel_size:int(kp_list[i][1]) + pixel_size,int(kp_list[i][0]) - pixel_size:int(kp_list[i][0]) + pixel_size, :] = 255#int(255 * kp_list[i][2] / 1.4)
	plt.figure()
	plt.imshow(img_toshow)
	plt.show()

def show_patch_set(patch_set,time_interval=0.5):
	"""
	显示patch集中的图像,一幅接一幅，时间间隔为0.5s，图像为rgb图
	:param patch_set: 要显示的patch集
	:return:
	"""
	plt.ion()
	plt.figure()
	for i in range(len(patch_set)):
		plt.imshow(patch_set[i])
		plt.pause(time_interval)
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
	:return: 返回kp点的集合(?,400,2)
	"""
	# img_path_list = ['/home/javen/javenlib/images/leuven/img1.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img2.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img3.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img4.ppm',
	# 	 '/home/javen/javenlib/images/leuven/img5.ppm']
	number_to_detect = 900
	number_to_save = 800
	sift = cv2.SIFT(number_to_detect)
	kp_set_raw = np.zeros(shape=(len(img_path_list),number_to_save,2),dtype=np.int)
	for img_count in range(len(img_path_list)):
		img = plt.imread(img_path_list[img_count])
		kp_object = sift.detect(img)
		kp_coordinate = KeyPoint_convert_forOpencv2(kp_object)
		while len(kp_coordinate)>number_to_save:
			kp_coordinate = np.delete(kp_coordinate,-1,axis=0)
		# print 'kp_coordinate:',kp_coordinate.shape
		kp_set_raw[img_count] = np.copy(kp_coordinate)
	# print 'kp_set_raw:',kp_set_raw.shape,kp_set_raw.dtype
	return kp_set_raw

def get_kp_set_positive(kp_set_raw,dist_threshold=18):
	"""
	使用flann的方法,寻找出kp_set_raw中具有重复性的kp点
	:param kp_set_raw: 输入sift检测出的原始kp点集
	:param dist_threshold: 相似距离的冗余度阈值,当距离小于18,认为两点重复,可判断为具有重复性的点
	:return: 返回具有可重复性的kp点集
	"""
	print 'kp_set_raw:',kp_set_raw.shape
	flann = pyflann.FLANN()
	new_test_data = np.copy(kp_set_raw[0])
	for step in range(4):
		test_data = np.copy(new_test_data)
		origin_data = kp_set_raw[step+1]
		matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 1,
													  algorithm="kmeans", branching=32, iterations=7, checks=16)
		count = 0
		new_test_data = np.zeros(shape=(1, 2))
		for i in range(len(test_data)):
			if matched_distances[i] < dist_threshold:
				count += 1
				new_test_data = np.append(new_test_data, test_data[i].reshape(1, 2), axis=0)
				# print matched_distances[i]
		new_test_data = np.delete(new_test_data,0,axis=0)
		# print 'step:',step,'count:',count
		# print 'new_test_data:',new_test_data.shape
	kp_set_positive = np.copy(new_test_data)
	# 进行非局部最大值抑制,即去除聚在一起的冗余的点,保留其中一个即可
	new_test_data = np.copy(kp_set_positive)
	for i in range(len(kp_set_positive)):
		origin_data = np.copy(new_test_data)
		test_data = np.copy(new_test_data)
		matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 2,
													  algorithm="kmeans", branching=32, iterations=7, checks=16)
		for j in range(len(test_data) - 1, -1, -1):
			if matched_distances[j, 1] < 1000:
				new_test_data = np.delete(test_data, j, axis=0)
				break
	kp_set_positive = np.copy(new_test_data)
	return kp_set_positive.astype(np.int)

def get_kp_set_negative(kp_set_raw,dist_threshold=1000):
	"""
	使用flann的方法,寻找出kp_set_raw中具有重复性的kp点
	:param kp_set_raw: 输入sift检测出的原始kp点集
	:param dist_threshold: 相似距离的冗余度阈值,当距离大于3000,可判断为不具有重复性的点
	:return: 返回不具有可重复性的kp点集
	"""
	print 'kp_set_raw:',kp_set_raw.shape
	flann = pyflann.FLANN()
	new_test_data = np.copy(kp_set_raw[0])
	for step in range(4):
		test_data = np.copy(new_test_data)
		origin_data = kp_set_raw[step+1]
		matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 1,
													  algorithm="kmeans", branching=32, iterations=7, checks=16)
		count = 0
		new_test_data = np.zeros(shape=(1, 2))
		for i in range(len(test_data)):
			if matched_distances[i] > dist_threshold:
				count += 1
				new_test_data = np.append(new_test_data, test_data[i].reshape(1, 2), axis=0)
				# print matched_distances[i]
		new_test_data = np.delete(new_test_data,0,axis=0)
		# print 'step:',step,'count:',count
		# print 'new_test_data:',new_test_data.shape
	kp_set_negative = np.copy(new_test_data)
	#进行非局部最大值抑制,即去除聚在一起的冗余的点,保留其中一个即可
	new_test_data = np.copy(kp_set_negative)
	for i in range(len(kp_set_negative)):
		origin_data = np.copy(new_test_data)
		test_data = np.copy(new_test_data)
		matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 2,
													  algorithm="kmeans", branching=32, iterations=7, checks=16)
		for j in range(len(test_data) - 1, -1, -1):
			if matched_distances[j, 1] < 1000:
				new_test_data = np.delete(test_data, j, axis=0)
				break
	kp_set_negative = np.copy(new_test_data)
	return kp_set_negative.astype(np.int)

def get_kp_patch_set_positive(img_path_list,kp_set_positive,scale=8):
	"""
	根据所给的图像，以及positive kp的坐标取出positive patch的集合
	:param img_path_list: 图像路径列表
	:param kp_set_positive: positive kp集合
	:param scale:该参数表示要取出的patch的大小,默认scale为32,也就是取出的patch几何大小为64*64
	:return: 返回positive patch集合（由于函数中要进行kp是否可取patch的边界范围判断，所以可能得到的patch数目比positive kp的数目少）
	"""
	#无法取到patch的kp点的去除
	rows_num,columns_num = plt.imread(img_path_list[0]).shape[0:2]
	for i in range(len(kp_set_positive)-1,-1,-1):
		if kp_set_positive[i,1] < scale or kp_set_positive[i,1] > rows_num-scale or kp_set_positive[i,0] < scale or kp_set_positive[i,0] > columns_num-scale:
			kp_set_positive = np.delete(kp_set_positive,i,axis=0)
	# print 'kp_set_positive:',kp_set_positive.shape,kp_set_positive
	#取出kp_set_positive所对应的patch集合
	kp_patch_set_positive = np.zeros(shape=(1,scale*2,scale*2,3))
	for img_count in range(len(img_path_list)):
		# img = plt.imread(img_path_list[img_count])/255. #二选一从原图提取patch就使用这一行,否则使用下面两行
		img = plt.imread(img_path_list[img_count])
		img_laplacian = cv2.Laplacian(img,ddepth=0,ksize=1)/255.
		for i in range(len(kp_set_positive)):
			kp_patch_set_positive = np.append(kp_patch_set_positive,img_laplacian[kp_set_positive[i,1]-scale:kp_set_positive[i,1]+scale,kp_set_positive[i,0]-scale:kp_set_positive[i,0]+scale,:].reshape(1,scale*2,scale*2,3),axis=0)
	kp_patch_set_positive = np.delete(kp_patch_set_positive,0,axis=0)
	# print kp_patch_set_positive.shape
	return kp_patch_set_positive

def get_kp_patch_set_negative(img_path_list,kp_set_negative,scale=8):
	#无法取到patch的kp点的去除
	rows_num,columns_num = plt.imread(img_path_list[0]).shape[0:2]
	for i in range(len(kp_set_negative)-1,-1,-1):
		if kp_set_negative[i,1] < scale or kp_set_negative[i,1] > rows_num-scale or kp_set_negative[i,0] < scale or kp_set_negative[i,0] > columns_num-scale:
			kp_set_negative = np.delete(kp_set_negative,i,axis=0)
	#取出kp_set_negative所对应的patch集合
	kp_patch_set_negative = np.zeros(shape=(1,scale*2,scale*2,3))
	for img_count in range(len(img_path_list)):
		# img = plt.imread(img_path_list[img_count])/255.
		img = plt.imread(img_path_list[img_count])
		img_laplacian = cv2.Laplacian(img, ddepth=0, ksize=1) / 255.
		for i in range(len(kp_set_negative)):
			kp_patch_set_negative = np.append(kp_patch_set_negative,img_laplacian[kp_set_negative[i,1]-scale:kp_set_negative[i,1]+scale,kp_set_negative[i,0]-scale:kp_set_negative[i,0]+scale,:].reshape(1,scale*2,scale*2,3),axis=0)
	kp_patch_set_negative = np.delete(kp_patch_set_negative,0,axis=0)
	return kp_patch_set_negative

def shuffle_data_and_label(train_data,train_label):
	x = np.arange(len(train_data))
	np.random.shuffle(x)
	shuffled_train_data = train_data[x]
	shuffled_train_label = train_label[x]
	return (shuffled_train_data,shuffled_train_label)

def rgb2gray_train_data(train_data,scale=32):
	"""
	将train_data从rgb模式转换为gray模式
	:param train_data: rgb模式的train_data,其维度必须为(?,64,64,3)的形式
	:return: 返回转换为gray模式的train_data,转换后的维度为(?,64,64,1)
	"""
	if train_data.shape[3] != 3:
		print 'Error: 输入数据的维度必须满足(?,64,64,3)'
		exit()
	train_data_gray = np.dot(train_data[...,:3],[0.2989,0.5870,0.1140])
	return train_data_gray.reshape(len(train_data),scale*2,scale*2,1)

# def NMS_4_points_set(kp_set,dist_threshold=1100):
# 	"""
# 	对一个points set 进行NMS,即非局部最大值抑制,即将点集合中比较彼此很靠近的点堆中只保留其中一个,其实并没有保留最大值,只是保留了其中一个值而已,不算完全的NMS
# 	:param kp_set: 需要进行NMS的点集
# 	:param dist_threshold: 相似距离的冗余度阈值,当距离大于1100,可判断为可以保留的点
# 	:return: 返回进过NMS过滤的点集
# 	"""
# 	# 进行非局部最大值抑制,即去除聚在一起的冗余的点,保留其中一个即可
# 	flann = pyflann.FLANN()
# 	new_test_data = np.copy(kp_set)
# 	for i in range(len(kp_set)):
# 		origin_data = np.copy(new_test_data)
# 		test_data = np.copy(new_test_data)
# 		matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 2,
# 													  algorithm="kmeans", branching=32, iterations=7, checks=16)
# 		for j in range(len(test_data) - 1, -1, -1):
# 			if matched_distances[j, 1] < dist_threshold:
# 				new_test_data = np.delete(test_data, j, axis=0)
# 				break
# 	return new_test_data

def quantity_test(kp_set1,kp_set2,groundtruth_matrix=None,threshold=25):
	if type(groundtruth_matrix) != types.NoneType:
		print 'rotating...'
		kp_set1_after_rotate = np.zeros(shape=(len(kp_set1),3))
		for i in range(len(kp_set1)):
			kp_set1_after_rotate[i] = groundtruth_matrix.dot(np.append(kp_set1[i],1))
		kp_set1 = np.copy(kp_set1_after_rotate[:,:2])
	else:
		print 'No rotation matrix! Going on...'
	flann = pyflann.FLANN()
	total_num = len(kp_set1)+len(kp_set2)
	#首先在1中检测有没有与2重复的
	origin_data = np.copy(kp_set1)
	test_data = np.copy(kp_set2)
	matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 1)
	count1 = 0
	# print 'matched_indices:',matched_indices
	# print 'matched_distances:',matched_distances
	for i in range(len(matched_distances)):
		if matched_distances[i] <= threshold:
			count1 += 1
	#然后在2中检测有没有与1重复的
	origin_data = np.copy(kp_set2)
	test_data = np.copy(kp_set1)
	matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 1)
	count2 = 0
	# print 'matched_indices:', matched_indices
	# print 'matched_distances:', matched_distances
	for i in range(len(matched_distances)):
		if matched_distances[i] <= threshold:
			count2 += 1
	print kp_set1.shape,kp_set2.shape
	print 'count1:',count1,'count2:',count2
	print 'quantity accuracy:',1.*(count1+count2)/total_num

def match_accuracy(img1_kp_pos,img1_kp_des,img2_kp_pos,img2_kp_des,rotation_matrix):
	"""
	用来进行 oxford数据集特征点匹配测试
	:param img1:
	:param img1_kp_pos:
	:param img1_kp_des:
	:param img2:
	:param img2_kp_pos:
	:param img2_kp_des:
	:param rotation_matrix:
	:return:
	"""
	# #多数量的set作test_data
	# flann = pyflann.FLANN()
	# kp_num = len(img1_kp_pos)
	# origin_kp = np.copy(img2_kp_pos)
	# origin_kp_des = np.copy(img2_kp_des)
	# test_kp = np.copy(img1_kp_pos)
	# test_kp_des = np.copy(img1_kp_des)
	# print 'kp_num:',kp_num
	# matched_index,matched_distance = flann.nn(origin_kp_des,test_kp_des,1)
	# match_count = 0
	# for i in range(kp_num):
	# 	matched_kp_4_test_kp = origin_kp[matched_index[i]]
	# 	# print 'matched_kp_4_test_kp:',matched_kp_4_test_kp
	# 	transformed_kp_4_test_kp = rotation_matrix.dot(np.append(test_kp[i],1))
	# 	# print 'transformed_kp_4_test_kp:',transformed_kp_4_test_kp
	# 	if ((matched_kp_4_test_kp[0]-transformed_kp_4_test_kp[0])**2 + (matched_kp_4_test_kp[1]-transformed_kp_4_test_kp[1])**2) <= 25:
	# 		match_count += 1
	# print 'match_count:',match_count
	# print 'match accuracy:',1.0*match_count/kp_num
	# return 1.0*match_count/kp_num

	# 少数量的set作test_data
	flann = pyflann.FLANN()
	kp_num = len(img2_kp_pos)
	origin_kp = np.copy(img1_kp_pos)
	origin_kp_des = np.copy(img1_kp_des)
	test_kp = np.copy(img2_kp_pos)
	test_kp_des = np.copy(img2_kp_des)
	print 'kp_num:', kp_num
	matched_index, matched_distance = flann.nn(origin_kp_des, test_kp_des, 1)
	match_count = 0
	for i in range(kp_num):
		matched_kp_4_test_kp = origin_kp[matched_index[i]]
		# print 'matched_kp_4_test_kp:',matched_kp_4_test_kp
		transformed_kp_4_test_kp = rotation_matrix.dot(np.append(matched_kp_4_test_kp, 1))
		# print 'transformed_kp_4_test_kp:',transformed_kp_4_test_kp
		if ((test_kp[i][0] - transformed_kp_4_test_kp[0]) ** 2 + (
			test_kp[i][1] - transformed_kp_4_test_kp[1]) ** 2) <= 25:
			match_count += 1
	print 'match_count:', match_count
	print 'match accuracy:', 1.0 * match_count / kp_num
	return 1.0 * match_count / kp_num

def get_matrix_from_file(filename):
	"""
	从文件中读取出变换的groundtruth matrix
	:param filename:文件保存路径
	:return:矩阵形式返回变换matrix
	"""
	f = open(filename,'r')
	rotation_matrix = np.zeros((3,3))
	for i in range(0,3):
		x = f.readline().split()
		rotation_matrix[i,0] = float(x[0])
		rotation_matrix[i,1] = float(x[1])
		rotation_matrix[i,2] = float(x[2])
	f.close()
	return rotation_matrix

#扫描式NMS算法
def NMS_4_kp_set(kp_set,row_num,column_num,step=8,n_pixel=16,threshold=0.5,scale=32):
	"""
	非局部最大值抑制,即去除局部区域里的冗余点,保留score最大的点
	:param kp_set: 原来的kp_set数据
	:param row_num: 图像参数行数
	:param column_num: 图像参数列数
	:param step: 扫描的步长
	:param n_pixel: 定义区域的大小,默认为以16*2为边长的正方形
	:param threshold: 抑制得出的点,需要经过第二层score的筛选,默认0.5,即没有进行进一步筛选
	:param scale: 搭配之前提取kp的参数,定义扫描的区域范围
	:return: 返回经过NMS之后的kp_set
	"""
	print 'before NMS:', len(kp_set)
	kp_set_afterNMS_list = []
	for i in range(scale, row_num - scale, step):  # 扫描的步长需要调整
		for j in range(scale, column_num - scale, step):
			# print 'before drop:',len(kp_set)
			point_temp = [0, 0, 0]
			for k in range(len(kp_set) - 1, -1, -1):
				# 判断并删除,从后往前不会影响次序
				if kp_set[k][0] >= j - n_pixel and kp_set[k][0] < j + n_pixel and kp_set[k][1] >= i - n_pixel and kp_set[k][
					1] < i + n_pixel:
					if kp_set[k][2] > point_temp[2]:
						point_temp = kp_set[k][:]
					del kp_set[k]
			# 判断point_temp是否符合要求,进行保留还是舍弃
			if point_temp[2] > threshold:
				kp_set_afterNMS_list.append(point_temp)
			# print 'after drop:',len(kp_set)
	print 'after NMS:', len(kp_set)
	print 'kp got:',len(kp_set_afterNMS_list)
	return kp_set_afterNMS_list

#理论式NMS算法
def NonMaxSuppresion_4_kp_set(kp_set_list,threshold=25):
	"""
	新版NMS,根据score以及IOU来进行局部非最大值的抑制
	:param kp_set_list:输入的kp点的集合
	:param threshold:进行抑制的局部区域的大小设置,threshold=25表示在5个像素的范围内取最大值
	:return:返回经过局部非最大值抑制的kp的集合
	"""
	#输入的kp_set的类型为python的list类型
	#第一步,将kp_set转为ndarray类型,然后按照score从大到小进行排序
	kp_set_array = np.zeros(shape=(len(kp_set_list),6))
	for i in range(len(kp_set_list)):
		kp_set_array[i] = np.copy(kp_set_list[i])
	kp_set_array = kp_set_array[kp_set_array[:,2].argsort()]
		#在转换会list方便操作
	kp_set_list_sorted = []
	for i in range(len(kp_set_array)):
		kp_set_list_sorted.append([kp_set_array[i,0],kp_set_array[i,1],kp_set_array[i,2],kp_set_array[i,3],kp_set_array[i,4],kp_set_array[i,5]])
	# print kp_set_list_sorted[0:5],len(kp_set_list_sorted)
	kp_set_list_afterNMS = []
	#循环迭代的NMS核心
	while len(kp_set_list_sorted) > 0:
		kp_popout = kp_set_list_sorted[-1]
		kp_set_list_afterNMS.append(kp_popout)
		del kp_set_list_sorted[-1]
		if len(kp_set_list_sorted) == 0:
			break
		for i in range(len(kp_set_list_sorted)-1,-1,-1):
			if (kp_set_list_sorted[i][0]-kp_popout[0]) ** 2 + (kp_set_list_sorted[i][1]-kp_popout[1]) ** 2 <= threshold:
				del kp_set_list_sorted[i]
	# print 'kp_set_list_afterNMS',kp_set_list_afterNMS[0],len(kp_set_list_afterNMS)
	return kp_set_list_afterNMS

#从list中选出排名靠前的一定数量的kp,返回值只保留kp的坐标信息
def choose_kp_from_list(kp_set_afterNMS_list,quantity_to_choose=250):
	"""
	从经过NMS之后的kp_set里面按照score挑出得分靠前的一定数量的kp
	:param kp_set_afterNMS_list: 待选的原始kp_set,数据类型是list类型
	:param quantity_to_choose: 需要取出的排名靠前的kp的数量
	:return: 返回score前n名的kp点的信息
	"""
	#要使用numpy的argsort()首先要将list类型转化为numpy array类型
	kp_set_afterNMS_array = np.zeros(shape=(len(kp_set_afterNMS_list),4))
	for i in range(len(kp_set_afterNMS_list)):
		kp_set_afterNMS_array[i] = np.copy(kp_set_afterNMS_list[i])
	# print kp_set_afterNMS_list[0:5]
	# print kp_set_afterNMS_array[0:5]
	kp_set_afterNMS_array = kp_set_afterNMS_array[kp_set_afterNMS_array[:,2].argsort()[-1::-1]]
	# print kp_set_afterNMS_array[0:5]
	if quantity_to_choose > len(kp_set_afterNMS_list):
		print 'quantity to choose is too large! Maxmium quantity is ',len(kp_set_afterNMS_list)
		return kp_set_afterNMS_array[:, 0:2].astype(np.int)
	elif quantity_to_choose == 0:
		return kp_set_afterNMS_array[:,0:2].astype(np.int)
	else:
		return kp_set_afterNMS_array[0:quantity_to_choose,0:2].astype(np.int)

#从list中选出排名靠前的一定数量的kp,并且筛选时加入对应octave边界的筛选,为之后的patch提取做准备,返回值保留kp的位置信息,score信息和octave_index信息,去掉了对应octave的尺寸信息
def choose_kp_from_list_careboundary(kp_set_afterNMS_list,quantity_to_choose=250,boundary_pixel=21):
	count=0
	kp_list_chosen=[]
	i=0
	octave_origin2sample_factor=[0.5,1,2]
	while(count < quantity_to_choose):
		#先要将origin中的坐标映射到octave中的坐标,再判断是否越界
		if kp_set_afterNMS_list[i][1]*octave_origin2sample_factor[int(kp_set_afterNMS_list[i][3])]-boundary_pixel >=0 and kp_set_afterNMS_list[i][1]*octave_origin2sample_factor[int(kp_set_afterNMS_list[i][3])]+boundary_pixel<kp_set_afterNMS_list[i][4] and kp_set_afterNMS_list[i][0]*octave_origin2sample_factor[int(kp_set_afterNMS_list[i][3])]-boundary_pixel >=0 and kp_set_afterNMS_list[i][0]*octave_origin2sample_factor[int(kp_set_afterNMS_list[i][3])]+boundary_pixel<kp_set_afterNMS_list[i][5]:
			kp_list_chosen.append([kp_set_afterNMS_list[i][0],kp_set_afterNMS_list[i][1],kp_set_afterNMS_list[i][2],kp_set_afterNMS_list[i][3]])
			count += 1
		i += 1
	print 'count:',count,'i:',i
	return kp_list_chosen

#根据选出的kp,进行patch的提取并计算重心angle,返回值保留kp的位置信息,score信息和octave_index信息,angle信息,以及经过rotation的28*28patch矩阵
def get_kp_list_withRotatedPatch(img_path,kp_list_chosen):
	img_origin = plt.imread(img_path)
	# print 'img_origin mean:',img_origin.mean()
	#将img_origin进行从0-255转化为0-1的预处理
	if img_origin.mean() > 1:
		img_origin = img_origin/255.
	#将img_origin进行从rgb转化为gray的预处理
	img_gray = np.dot(img_origin,[0.2989,0.5870,0.1140])
	#做减去均值的预处理来抵抗illumination变化的影响
	img_gray = img_gray - img_gray.mean()
	img_gray_octave1 = np.copy(img_gray)
	img_gray_octave0 = cv2.pyrDown(img_gray_octave1)
	img_gray_octave2 = cv2.pyrUp(img_gray_octave1)
	# print img_gray.shape,type(img_gray)
	# print img_gray_octave0.shape,img_gray_octave1.shape,img_gray_octave2.shape
	# plt.figure(1)
	# plt.imshow(img_origin)
	# plt.figure(2)
	# plt.imshow(img_gray_octave2,cmap='gray')
	# plt.show()
	octave_origin2sample_factor=[0.5,1,2]
	total_count = len(kp_list_chosen)
	for i in range(total_count):
		#先取出行和列的坐标
		octave_index = int(kp_list_chosen[i][3])
		row = int(round(kp_list_chosen[i][1]*octave_origin2sample_factor[octave_index]))
		column = int(round(kp_list_chosen[i][0]*octave_origin2sample_factor[octave_index]))
		if kp_list_chosen[i][3] == 0:
			patch = np.copy(img_gray_octave0[row-21:row+22,column-21:column+22])
		elif kp_list_chosen[i][3] == 1:
			patch = np.copy(img_gray_octave1[row - 21:row + 22, column - 21:column + 22])
		elif kp_list_chosen[i][3] == 2:
			patch = np.copy(img_gray_octave2[row - 21:row + 22, column - 21:column + 22])
		# print 'i:',i,'octave_index:',kp_list_chosen[i][3],'patch size:',patch.shape
		#对43*43的patch进行重心的angle计算(方法:先进行置零操作然后计算angle)
		patch_after_set_zero = middle_area_set_zero(patch)
		degree = get_patch_angle(patch_after_set_zero)
		#根据angle,对43*43的patch进行rotation操作
		patch_after_set_zero_and_rotation = image_rotate(patch_after_set_zero,-1*degree)
		#对旋转之后的43*43的patch进行取出其中29*29patch的操作
		patch_29x29 = np.copy(patch_after_set_zero_and_rotation[7:36,7:36])
		# patch_28x28 = np.copy(patch_after_set_zero_and_rotation[7:35, 7:35])
		#将29*29patch转化为28*28patch,具体操作是去掉中心的一行和一列
		patch_28x28 = np.delete(patch_29x29,14,axis=0)
		patch_28x28 = np.delete(patch_28x28,14,axis=1)
		#将求得的angle和28x28patch添加到kp的list中去
		kp_list_chosen[i].append(degree)
		kp_list_chosen[i].append(patch_28x28)
	return kp_list_chosen


#MSE版的use_TILDE
def use_TILDE(img_path_list):
	tf_x = tf.placeholder(tf.float32, [None, 64, 64, 1])  # 输入patch维度为64*64
	tf_y = tf.placeholder(tf.int32, [None, 1])  # input y ,y代表score所以维度为1

	########################################################################
	conv1 = tf.layers.conv2d(
		inputs=tf_x,  # (?,64,64,1)
		filters=16,
		kernel_size=5,
		strides=1,
		padding='valid',
		activation=tf.nn.relu
	)  # -> (?, 60, 60, 16)

	pool1 = tf.layers.max_pooling2d(
		conv1,
		pool_size=2,
		strides=2,
	)  # -> (?,30, 30, 16)

	conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'valid', activation=tf.nn.relu)  # -> (?,26, 26, 32)

	pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,13, 13, 32)

	conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)  # -> (?,11, 11, 32)

	# pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
	#
	flat = tf.reshape(conv3, [-1, 11 * 11 * 32])  # -> (?,32*32*64)

	output = tf.layers.dense(flat, 1)  # output layer

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess,'./save_net/detector_TILDE_model_20171219_mse_20_0_0005')

	#使用列表将两个维度不相同的矩阵打包在一起return
	kp_set_afternms_list = []
	for img_count in range(len(img_path_list)):
		img_test_rgb = plt.imread(img_path_list[img_count])/255.
		img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
		kp_set = np.zeros(shape=(0,2))
		#对图片进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
		row_num = plt.imread(img_path_list[0]).shape[0]
		column_num = plt.imread(img_path_list[0]).shape[1]
		for i in range(32,row_num-32,4): #扫描的步长需要调整
			for j in range(32,column_num-32,4):
				patch = np.copy(img_test_gray[i-32:i+32,j-32:j+32]).reshape(1,64,64,1)
				output_predict = sess.run(output, feed_dict={tf_x:patch})
				if output_predict >= 0.5:
					# print output_predict
					kp_set = np.append(kp_set,[[j,i]],axis=0)
		kp_set = kp_set.astype(np.int)
		# print kp_set.shape#,kp_set
		kp_set_afternms = NMS_4_points_set(kp_set)
		# print 'kp_set_afternms:',kp_set_afternms.shape
		# javenlib_tf.show_kp_set(img_path_list[img_count],kp_set)
		# javenlib_tf.show_kp_set(img_path_list[img_count],kp_set_afternms)
		kp_set_afternms_list.append(kp_set)
	# print 'kp_set_afternms_list:',len(kp_set_afternms_list),kp_set_afternms_list[0].shape,kp_set_afternms_list[1].shape

	# #在图上显示检测出的点
	# javenlib_tf.show_kp_set(img_path_list[0],kp_set_afternms_list[0])
	# javenlib_tf.show_kp_set(img_path_list[1], kp_set_afternms_list[1])
	sess.close()
	return kp_set_afternms_list

#MSE版的use_TILDE增加循环优化
def use_TILDE_optimized(img_path_list):
	tf_x = tf.placeholder(tf.float32, [None, 64, 64, 1])  # 输入patch维度为64*64
	tf_y = tf.placeholder(tf.int32, [None, 1])  # input y ,y代表score所以维度为1

	########################################################################
	conv1 = tf.layers.conv2d(
		inputs=tf_x,  # (?,64,64,1)
		filters=16,
		kernel_size=5,
		strides=1,
		padding='valid',
		activation=tf.nn.relu
	)  # -> (?, 60, 60, 16)

	pool1 = tf.layers.max_pooling2d(
		conv1,
		pool_size=2,
		strides=2,
	)  # -> (?,30, 30, 16)

	conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'valid', activation=tf.nn.relu)  # -> (?,26, 26, 32)

	pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,13, 13, 32)

	conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)  # -> (?,11, 11, 32)

	# pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
	#
	flat = tf.reshape(conv3, [-1, 11 * 11 * 32])  # -> (?,32*32*64)

	output = tf.layers.dense(flat, 1)  # output layer

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, './save_net/detector_TILDE_model_20171219_mse_20_0_0005')

	# 使用列表将两个维度不相同的矩阵打包在一起return
	kp_set_list = []
	for img_count in range(len(img_path_list)):
		img_test_rgb = plt.imread(img_path_list[img_count]) / 255.
		img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
		kp_set = np.zeros(shape=(0, 3))
		# 对图片进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
		row_num = plt.imread(img_path_list[0]).shape[0]
		column_num = plt.imread(img_path_list[0]).shape[1]
		kp_set = []
		count = 0
		for i in range(32, row_num - 32, 4):  # 扫描的步长需要调整
			for j in range(32, column_num - 32, 4):
				patch = img_test_gray[i - 32:i + 32, j - 32:j + 32].reshape(1,64, 64, 1)
				output_predict = sess.run(output, feed_dict={tf_x: patch})
				if output_predict[0,0] >= 0.5:
					count += 1
					kp_set.append([j,i,output_predict[0,0]])
		print 'from image',img_count,'kp count from cnn without NMS:',count
		# #为了方便后续处理,将list在转化为np.ndarray传出去
		# kp_set_2array = np.zeros(shape=(len(kp_set),3))
		# for i in range(len(kp_set)):
		# 	kp_set_2array[i,:] = kp_set[i][:]
		# show_kp_set('/home/javen/javenlib/images/bikes/img1.ppm',kp_set_2array[:,0:2].astype(np.int))

		# #进行NMS扫描去冗余(发现没加实时删除，导致重复率高)
		# print 'before NMS:',len(kp_set_2array)
		# kp_set_2array_afternms = np.zeros(shape=(0,3))
		# for i in range(32, row_num - 32, 8):  # 扫描的步长需要调整
		# 	for j in range(32, column_num - 32, 8):
		# 		point_temp = np.zeros(shape=(1,3))
		# 		for k in range(len(kp_set_2array)):
		# 			if kp_set_2array[k,0] >= j-15 and kp_set_2array[k,0] < j+15 and kp_set_2array[k,1] >= i-15 and kp_set_2array[k,1] < i+15:
		# 				if kp_set_2array[k,2] > point_temp[0,2]:
		# 					point_temp = np.copy(kp_set_2array[k].reshape(1,3))
		# 					# print 'point_temp.shape',point_temp.shape
		# 		if point_temp[0,2] != 0:
		# 			kp_set_2array_afternms = np.append(kp_set_2array_afternms,point_temp,axis=0)
		# print kp_set_2array_afternms[0:5],kp_set_2array_afternms.shape

		# # 进行NMS扫描去冗余,加实时删除操作(发现实时删除时kp_set的删除更新有问题,每次其实只删除了一个点)
		# print 'before NMS:', len(kp_set_2array)
		# kp_set_2array_afternms = np.zeros(shape=(0, 3))
		# kp_set_2array_afterdelete = np.copy(kp_set_2array)
		# for i in range(32, row_num - 32, 8):  # 扫描的步长需要调整
		# 	for j in range(32, column_num - 32, 8):
		# 		point_temp = np.zeros(shape=(1, 3))
		# 		kp_set_2array = np.copy(kp_set_2array_afterdelete)
		# 		print 'before drop:',kp_set_2array.shape
		# 		cc = 0
		# 		for k in range(len(kp_set_2array)):
		# 			if kp_set_2array[k, 0] >= j - 100 and kp_set_2array[k, 0] < j + 100 and kp_set_2array[
		# 				k, 1] >= i - 100 and kp_set_2array[k, 1] < i + 100:
		# 				cc += 1
		# 				if kp_set_2array[k, 2] > point_temp[0, 2]:
		# 					point_temp = np.copy(kp_set_2array[k].reshape(1, 3))
		# 				#删除已经进行区域比较的点
		# 				kp_set_2array_afterdelete = np.delete(kp_set_2array,k,axis=0)
		# 				# print 'point_temp.shape',point_temp.shape
		# 		print 'after drop:',kp_set_2array_afterdelete.shape,'drop quantity:',cc
		# 		if point_temp[0, 2] > 0.6:
		# 			kp_set_2array_afternms = np.append(kp_set_2array_afternms, point_temp, axis=0)
		# print kp_set_2array_afternms[0:5], kp_set_2array_afternms.shape
		# #将筛选后的kp点存入list中
		# kp_set_list.append(kp_set_2array_afternms[:,0:2].astype(np.int))

		# #进行NMS扫描去冗余, 加实时删除操作,此版本的删除策略是倒序遍历删除.第二点就是使用list来提高效率
		# print 'before NMS:',len(kp_set)
		# kp_set_afterNMS_list = []
		# for i in range(32, row_num - 32, 8):  # 扫描的步长需要调整
		# 	for j in range(32, column_num - 32, 8):
		# 		# print 'before drop:',len(kp_set)
		# 		point_temp = [0,0,0]
		# 		for k in range(len(kp_set)-1,-1,-1):
		# 			#判断并删除,从后往前不会影响次序
		# 			if kp_set[k][0] >= j-32 and kp_set[k][0] < j+32 and kp_set[k][1] >= i-32 and kp_set[k][1] < i+32:
		# 				if kp_set[k][2] > point_temp[2]:
		# 					point_temp = kp_set[k][:]
		# 				del kp_set[k]
		# 		#判断point_temp是否符合要求,进行保留还是舍弃
		# 		if point_temp[2] > 0.5:
		# 			kp_set_afterNMS_list.append(point_temp)
		# 		# print 'after drop:',len(kp_set)
		# print 'after NMS:',len(kp_set)
		# # print 'kp got:',len(kp_set_afterNMS_list),kp_set_afterNMS_list[0:5]
		kp_set_afterNMS_list = NMS_4_kp_set(kp_set,row_num,column_num,step=8,n_pixel=16,threshold=0.5)
		kp_set_afterNMS_list = NMS_4_kp_set(kp_set_afterNMS_list, row_num, column_num,step=8,n_pixel=32,threshold=0.6)
		#将list转换为ndarray,并放入作为一个元素放入list中
		kp_set_afterNMS_array = np.zeros(shape=(len(kp_set_afterNMS_list),2),dtype=np.int)
		for i in range(len(kp_set_afterNMS_list)):
			kp_set_afterNMS_array[i] = np.copy(kp_set_afterNMS_list[i][0:2])
		# print 'kp_set_afterNMS_array:',kp_set_afterNMS_array.shape,kp_set_afterNMS_array[0:3]
		kp_set_list.append(kp_set_afterNMS_array)
	#释放gpu资源
	sess.close()
	return kp_set_list

#MSE版的use_TILDE,scale为8,不使用色彩信息
def use_TILDE_scale8(img_path_list):
	"""
	使用训练好的CNN detector,返回值为带有score的kp点的集合,后续可以使用score从中挑选出一定数量的kp点
	:param img_path: 需要检测的图像的path,多幅图同时检测
	:return: 返回带有score的kp点集合
	"""
	scale = 8
	tf_x = tf.placeholder(tf.float32, [None, scale*2, scale*2, 1])  # 输入patch维度为64*64
	tf_y = tf.placeholder(tf.int32, [None, 1])  # input y ,y代表score所以维度为1

	########################################################################
	conv1 = tf.layers.conv2d(
		inputs=tf_x,  # (?,16,16,1)
		filters=8,
		kernel_size=5,
		strides=1,
		padding='same',
		activation=tf.nn.relu
	)  # -> (?, 16, 16, 8)

	pool1 = tf.layers.max_pooling2d(
		conv1,
		pool_size=2,
		strides=2,
	)  # -> (?,8, 8, 8)

	conv2 = tf.layers.conv2d(pool1, 16, 5, 1, 'same', activation=tf.nn.relu)  # -> (?,8, 8, 16)

	pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,4, 4, 16)

	# conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)  # -> (?,11, 11, 32)

	# pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
	#
	flat = tf.reshape(pool2, [-1, 4*4*16])  # -> (?,256)

	output = tf.layers.dense(flat, 1)  # output layer

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, './save_net/detector_TILDE_model_20180102_mse_100_0_0005')
	# saver.restore(sess, './save_net/detector_TILDE_model_20180323_mse_500_0_0005')  # 使用laplacian做输入patch检测

	# 使用列表将两个维度不相同的矩阵打包在一起return
	kp_set_list = []
	for image_count in range(len(img_path_list)):
		img_test_rgb = plt.imread(img_path_list[image_count])
		# img_test_rgb = np.copy(cv2.Laplacian(img_test_rgb,ddepth=0,ksize=1))
		# plt.figure(1)
		# plt.imshow(img_test_rgb)
		# plt.show()
		if img_test_rgb.mean() > 1:
			img_test_rgb = np.copy(img_test_rgb / 255.)
		img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
		kp_set = np.zeros(shape=(0, 3))
		# 对图片进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
		row_num = plt.imread(img_path_list[image_count]).shape[0]
		column_num = plt.imread(img_path_list[image_count]).shape[1]
		kp_set = []
		count = 0
		for i in range(scale, row_num - scale, 4):  # 扫描的步长需要调整
			for j in range(scale, column_num - scale, 4):
				patch = img_test_gray[i - scale:i + scale, j - scale:j + scale].reshape(1, scale*2, scale*2, 1)
				output_predict = sess.run(output, feed_dict={tf_x: patch})
				if output_predict[0, 0] >= 0.6:
					count += 1
					kp_set.append([j, i, output_predict[0, 0],1,row_num,column_num])
		print 'kp count from cnn without NMS:', count
		# kp_set_afterNMS_list = NMS_4_kp_set(kp_set, row_num, column_num, step=8, n_pixel=32, threshold=0.75)
		kp_set_afterNMS_list = NonMaxSuppresion_4_kp_set(kp_set,threshold=25)
		print 'NMS之后,保留:',len(kp_set_afterNMS_list),kp_set_afterNMS_list[-5:-1]
		kp_set_list.append(kp_set_afterNMS_list)
	# 释放gpu资源
	sess.close()
	print '一次结束!'
	return kp_set_list

#MSE版的use_TILDE,scale为8,不使用色彩信息,加入scale invariant处理(使用图像金字塔)
def use_TILDE_scale8_withpyramid(img_path_list):
	"""
	使用训练好的CNN detector,返回值为带有score的kp点的集合,后续可以使用score从中挑选出一定数量的kp点,加入图像金字塔来适应scale-invariant
	:param img_path: 需要检测的图像的path,多幅图同时检测
	:return: 返回带有score的kp点集合
	"""

	scale = 8
	tf_x = tf.placeholder(tf.float32, [None, scale*2, scale*2, 1])  # 输入patch维度为(?,16,16,1)
	tf_y = tf.placeholder(tf.int32, [None, 1])  # input y ,y代表score所以维度为1

	########################################################################
	conv1 = tf.layers.conv2d(
		inputs=tf_x,  # (?,16,16,1)
		filters=8,
		kernel_size=5,
		strides=1,
		padding='same',
		activation=tf.nn.relu
	)  # -> (?, 16, 16, 8)

	pool1 = tf.layers.max_pooling2d(
		conv1,
		pool_size=2,
		strides=2,
	)  # -> (?,8, 8, 8)

	conv2 = tf.layers.conv2d(pool1, 16, 5, 1, 'same', activation=tf.nn.relu)  # -> (?,8, 8, 16)

	pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,4, 4, 16)

	# conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)  # -> (?,11, 11, 32)

	# pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
	#
	flat = tf.reshape(pool2, [-1, 4*4*16])  # -> (?,256)

	output = tf.layers.dense(flat, 1)  # output layer

	# graph1 = tf.Graph()
	sess = tf.Session()
	saver = tf.train.Saver()
	# saver.restore(sess, './save_net/detector_TILDE_model_20180102_mse_100_0_0005')
	saver.restore(sess, './save_net/detector_TILDE_model_20180323_mse_500_0_0005') #使用laplacian做输入patch检测

	# 使用列表将两个维度不相同的矩阵打包在一起return
	kp_set_list = []
	for image_count in range(len(img_path_list)):
		img_test_rgb = plt.imread(img_path_list[image_count])
		#将像素值从0-255转化成0-1
		if img_test_rgb.mean() > 1:
			img_test_rgb = np.copy(img_test_rgb / 255.)
		#将图像从rgb转换成gray
		img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
		#kp_set = np.zeros(shape=(0, 3))
		print 'img_test_gray.shape:', img_test_gray.shape
		#获得一个图片的3个scale的金字塔
		img_test_gray_pyramid_list = get_pyramid_of_image(img_test_gray)
		print '三个scale的像素：',img_test_gray_pyramid_list[0].shape,img_test_gray_pyramid_list[1].shape,img_test_gray_pyramid_list[2].shape#,img_test_gray_pyramid_list[3].shape#,img_test_gray_pyramid_list[4].shape
		# 对金字塔每个octave进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
		kp_set = []
		octave_factor=[2,1,0.5] #用来恢复原图坐标的比例因子
		for octave_count in range(3):
			row_num = img_test_gray_pyramid_list[octave_count].shape[0]
			column_num = img_test_gray_pyramid_list[octave_count].shape[1]
			count = 0
			for i in range(scale, row_num - scale, 4):  # 扫描的步长需要调整
				for j in range(scale, column_num - scale, 4):
					patch = img_test_gray_pyramid_list[octave_count][i - scale:i + scale, j - scale:j + scale].reshape(1, scale*2, scale*2, 1)
					output_predict = sess.run(output, feed_dict={tf_x: patch})
					if output_predict[0, 0] >= 0.6:
						count += 1
						kp_set.append([j*octave_factor[octave_count], i*octave_factor[octave_count], output_predict[0, 0],octave_count,row_num,column_num])
			print 'kp count from cnn without NMS:', 'octave:',octave_count,'count:',count
		print 'NMS之前 total count:',len(kp_set)
		# kp_set_afterNMS_list = NMS_4_kp_set(kp_set, row_num, column_num, step=8, n_pixel=32, threshold=0.75)
		kp_set_afterNMS_list = NonMaxSuppresion_4_kp_set(kp_set,threshold=25)
		print 'NMS之后,保留:',len(kp_set_afterNMS_list),kp_set_afterNMS_list[0:5]
		kp_set_list.append(kp_set_afterNMS_list)
	# 释放gpu资源
	sess.close()
	print '一次结束!'
	return kp_set_list

def get_pyramid_of_image(img):
	"""
	建立图像金字塔,由于达到两级上采样时其图像分辨率过大,所向上采样只用到一级,向下采样保留两级的下采样
	:param img: 需要建立图像octave金字塔的原图像
	:return: 返回建立好的金字塔list,list中的每个元素都是一幅图像矩阵
	"""
	row_num = img.shape[0]
	column_num = img.shape[1]
	print row_num,column_num
	img_scaleup1 = cv2.pyrUp(img)	#.reshape(row_num*2,column_num*2,1)
	img_scaleup2 = cv2.pyrUp(img_scaleup1)	#.reshape(row_num*4,column_num*4,1)
	img_scaledown1 = cv2.pyrDown(img)	#.reshape(row_num/2,column_num/2,1)
	img_scaledown2 = cv2.pyrDown(img_scaledown1)	#.reshape(row_num/4,column_num/4,1)
	print img_scaleup2.shape,img_scaleup1.shape,img.shape,img_scaledown1.shape,img_scaledown2.shape
	img_pyramid_list = [np.expand_dims(img_scaledown1,axis=2),img,np.expand_dims(img_scaleup1,axis=2)]
	return img_pyramid_list

def parse_kp_from_totallist(kp_set_list):
	"""
	将经过NMS之后合并到一起的kp_set按照octave_index分开存储到不同的list中去,以便用于在图像中显示观察
	:param kp_set_list: 被混合的kp_set_list
	:return: 分组过后的kp_set list
	"""
	total_count = len(kp_set_list)
	# print total_count
	octave0_kp_list = []
	octave1_kp_list = []
	octave2_kp_list = []
	# octave3_kp_list = []
	for i in range(total_count):
		if kp_set_list[i][3] == 0:
			octave0_kp_list.append([kp_set_list[i][0]/4,kp_set_list[i][1]/4,kp_set_list[i][2],kp_set_list[i][3]])
		elif kp_set_list[i][3] == 1:
			octave1_kp_list.append([kp_set_list[i][0]/2, kp_set_list[i][1]/2, kp_set_list[i][2], kp_set_list[i][3]])
		elif kp_set_list[i][3] == 2:
			octave2_kp_list.append([kp_set_list[i][0], kp_set_list[i][1], kp_set_list[i][2], kp_set_list[i][3]])
		# elif kp_set_list[i][3] == 3:
		# 	octave3_kp_list.append([kp_set_list[i][0]*2, kp_set_list[i][1]*2, kp_set_list[i][2], kp_set_list[i][3]])
	# print 'octave0_kp_list:',len(octave0_kp_list),len(octave1_kp_list),len(octave2_kp_list),len(octave3_kp_list)
	octave_all_kp_list = [octave0_kp_list,octave1_kp_list,octave2_kp_list]
	return octave_all_kp_list

def use_CNN_descriptor_generator(kp_set_list):
	"""
	通过CNN对输入的kp_set根据其28*28patch算出其descriptor,并附加在kp_set_list中然后输出
	输入的kp_set_list的结构为每一个kp点格式为[j,i,score,octave_index,angle,28*28patch_ndarray]
	输出的kp_set_list的结构在每个kp点后面附加了求出的descriptor
	:param kp_set_list_in:
	:return:
	"""

	mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
	train_images = mnist.train.images.reshape(55000, 28, 28, 1)
	train_labels = mnist.train.labels
	test_images = mnist.test.images.reshape(10000, 28, 28, 1)
	test_labels = mnist.test.labels
	print '训练数据和测试数据的shape：', train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
	# plt.imshow(train_images[0].reshape(28,28),cmap='gray')
	# plt.show()

	# 构建网络结构
	# 读取准备好的数据,mnist数据集本身就经过了预处理所以不再需要做与处理了
	train_x = np.copy(train_images)
	train_y = np.copy(train_labels)

	#为了避免两个graph造成的冲突,创建一个新的graph不再使用默认的graph,然后结合with来完成session的计算任务
	g1 = tf.Graph()
	with g1.as_default():
		tf_x = tf.placeholder(tf.float32, [None, 28, 28, 1])  # 输入patch维度为28*28
		tf_y = tf.placeholder(tf.int32, [None, 10])  # input y ,y代表预测标签所以维度为10

		########################################################################
		conv1 = tf.layers.conv2d(
			inputs=tf_x,  # (?,28,28,1)
			filters=32,
			kernel_size=5,
			strides=1,
			padding='same',  # same为保持原size,valid为去除边界size会变小
			activation=tf.nn.relu
		)  # -> (?, 28, 28, 32)

		pool1 = tf.layers.max_pooling2d(
			conv1,
			pool_size=2,
			strides=2,
		)  # -> (?,14, 14, 32)

		conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)  # -> (?,14, 14, 64)

		pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,7, 7, 64)

		flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # -> (?,7*7*64)

		full_connect1 = tf.layers.dense(flat, 512)  # -> (?,512)

		output = tf.layers.dense(full_connect1, 10)  # output layer (?,10)

	# 输出构建的网络结构的各层的维度情况
	print conv1
	print pool1
	print conv2
	print pool2
	print flat
	print full_connect1
	print output

	#将训练好的模型加载进来
	sess = tf.Session(graph=g1)
	with sess:
		saver = tf.train.Saver()
		saver.restore(sess,'./save_net_descriptor_generator/descriptor_generator_model_20180302_softmax_10circle_LR0_0005')

		#计算输入kp_set_list中每个kp的28*28patch所对应的descriptor
		count = len(kp_set_list)
		# print 'use CNN 中的count:',count,kp_set_list_in[0][5].shape
		for i in range(count):
			patch = np.copy(kp_set_list[i][5])
			descriptor = sess.run(flat,feed_dict={tf_x:patch.reshape(1,28,28,1)})
			# print 'descriptor shape:',descriptor.shape,descriptor.reshape(512).shape
			kp_set_list[i].append(descriptor)

		# # 测试训练集准确率
		# count = 0
		# total_num = 55000
		# for i in range(total_num):
		# 	output_predict = sess.run(output, feed_dict={tf_x: train_x[i].reshape(1, 28, 28, 1)})
		# 	# print 'step:',i,'output_predict',output_predict.shape,output_predict
		# 	if np.argmax(output_predict) == np.argmax(train_y[i]):
		# 		count += 1
		# print '训练集准确率:', 1. * count / total_num
        #
		# # 测试测试集准确率
		# count = 0
		# total_num = 10000
		# for i in range(total_num):
		# 	output_predict = sess.run(output, feed_dict={tf_x: test_images[i].reshape(1, 28, 28, 1)})
		# 	# print 'step:',i,'output_predict',output_predict.shape,output_predict
		# 	if np.argmax(output_predict) == np.argmax(test_labels[i]):
		# 		count += 1
		# print '测试集准确率:', 1. * count / total_num
	sess.close()
	return kp_set_list

def kp_list_2_pos_des_array(kp_set_list):
	"""
	将得到的kp_set的list进行提取,抽出其中的position矩阵和descriptor矩阵
	:param kp_set_list: kp_set_list的格式为[j,i,score,octave_index,angle,28*28patcharray,descriptor]
	:return:返回pos和des两个numpy矩阵
	"""
	count = len(kp_set_list)
	print 'kp list [6] shape:',kp_set_list[0][6].shape
	des_size = kp_set_list[0][6].shape[1]
	print 'des_size:',des_size
	position_array = np.zeros(shape=(count,2),dtype=np.int)
	descriptor_array = np.zeros(shape=(count,des_size))
	for i in range(count):
		position_array[i,0] = kp_set_list[i][0]
		position_array[i,1] = kp_set_list[i][1]
		descriptor_array[i] = np.copy(kp_set_list[i][6].reshape(des_size))
	# print position_array[0:2]
	# print descriptor_array.shape
	return [position_array,descriptor_array]

def extract_kp_pos_array_from_kp_set_list(kp_set_list):
	"""
	将得到的kp_set_list进行提取,只取出其中的position矩阵
	:param kp_set_list:
	:return:
	"""
	count = len(kp_set_list)
	kp_pos_array = np.zeros(shape=(count,2),dtype=np.int)
	for i in range(count):
		kp_pos_array[i,0] = int(kp_set_list[i][0])
		kp_pos_array[i,1] = int(kp_set_list[i][1])
	return kp_pos_array

def transform_from_array_2_list(kp_pos_array,row_num,column_num):
	"""
	将position矩阵转换成可以用与CNN des计算的kp_list
	:param kp_pos_array:
	:param row_num:
	:param column_num:
	:return:
	"""
	kp_list = []
	count = kp_pos_array.shape[0]
	for i in range(count):
		y = kp_pos_array[i,1]
		x = kp_pos_array[i,0]
		if x-21 >= 0 and x+21 < column_num and y-21 >= 0 and y+21 < row_num:
			kp_list.append([kp_pos_array[i,0],kp_pos_array[i,1],0.8,1])
	return kp_list

###################################################################################
###################################################################################
###################################################################################
# img_path_list = ['/home/javen/javenlib/images/bikes/img1.ppm',
# 				 '/home/javen/javenlib/images/bikes/img2.ppm']
# img1 = plt.imread(img_path_list[0])
# img2 = plt.imread(img_path_list[1])
# kp_set_list = use_TILDE_scale8_withpyramid(img_path_list) #包含两幅图的kp_set
# print len(kp_set_list)
# print len(kp_set_list[0]),len(kp_set_list[1])
# kp_list_chosenIMG1 = choose_kp_from_list_careboundary(kp_set_list[0],quantity_to_choose=250,boundary_pixel=21)
# print '考虑octave边界后选出的kp list:',len(kp_list_chosenIMG1),kp_list_chosenIMG1[0]
# print kp_list_chosenIMG1
# kp_list_withRotatedPatchIMG1 = get_kp_list_withRotatedPatch(img_path_list[0],kp_list_chosenIMG1)
# print '取出28*28patch后的结构:',kp_list_withRotatedPatchIMG1[0]
# kp_list_withDescriptorIMG1 = use_CNN_descriptor_generator(kp_list_withRotatedPatchIMG1)
# print 'kp_list_withDescriptorIMG1:',len(kp_list_withDescriptorIMG1),kp_list_withDescriptorIMG1[0][6].shape,kp_list_withRotatedPatchIMG1[0][5].shape
# pos,des = kp_list_2_pos_des_array(kp_list_withDescriptorIMG1)
# print pos.shape,des.shape
# print pos
# print des

# show_kp_set_listformat(octave0_image,octave0_kp_list)
# show_kp_set_listformat(octave1_image,octave1_kp_list)
# show_kp_set_listformat(octave2_image,octave2_kp_list)
# show_kp_set_listformat(octave3_image,octave3_kp_list)
# print 'hello:',len(kp_set_list),len(kp_set_list[0]),len(kp_set_list[1])

# img1_gray = np.dot(img1,[0.2989,0.5870,0.1140])/255.
# patch = np.copy(img1_gray[450-21:450+22,480-21:480+22])
# plt.figure(1)
# plt.imshow(patch,cmap='gray')
# patch_after_zero = middle_area_set_zero(patch)
# print patch_after_zero.shape
# plt.figure(2)
# plt.imshow(patch_after_zero,cmap='gray')
#
# degree = get_patch_angle(patch_after_zero)
# print 'degree:',degree
# patch_after_zero_and_rotation = image_rotate(patch_after_zero,-1*degree)
# plt.figure(3)
# plt.imshow(patch_after_zero_and_rotation,cmap='gray')
# plt.show()

# x = np.zeros(shape=(43,43))
# for i in range(43):
# 	x[i,:] = 1
# x[:8,38:] = 100
# print x
# degree = get_patch_angle(x)
# print 'degree:',degree
# x_after_rotation = image_rotate(x,-1*degree)
# for j in range(43):
# 	print x_after_rotation[j]
# degree2 = get_patch_angle(x_after_rotation)
# print degree2