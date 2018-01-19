#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann
import time
import types
import cmath

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
		kp_obj = cv2.KeyPoint(keypoints[i,0],keypoints[i,1],_size=3.58366942406)
		kp_list.append(kp_obj)
	return kp_list

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

def get_kp_patch_set_positive(img_path_list,kp_set_positive,scale=32):
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
		img = plt.imread(img_path_list[img_count])/255.
		for i in range(len(kp_set_positive)):
			kp_patch_set_positive = np.append(kp_patch_set_positive,img[kp_set_positive[i,1]-scale:kp_set_positive[i,1]+scale,kp_set_positive[i,0]-scale:kp_set_positive[i,0]+scale,:].reshape(1,scale*2,scale*2,3),axis=0)
	kp_patch_set_positive = np.delete(kp_patch_set_positive,0,axis=0)
	# print kp_patch_set_positive.shape
	return kp_patch_set_positive

def get_kp_patch_set_negative(img_path_list,kp_set_negative,scale=32):
	#无法取到patch的kp点的去除
	rows_num,columns_num = plt.imread(img_path_list[0]).shape[0:2]
	for i in range(len(kp_set_negative)-1,-1,-1):
		if kp_set_negative[i,1] < scale or kp_set_negative[i,1] > rows_num-scale or kp_set_negative[i,0] < scale or kp_set_negative[i,0] > columns_num-scale:
			kp_set_negative = np.delete(kp_set_negative,i,axis=0)
	#取出kp_set_negative所对应的patch集合
	kp_patch_set_negative = np.zeros(shape=(1,scale*2,scale*2,3))
	for img_count in range(len(img_path_list)):
		img = plt.imread(img_path_list[img_count])/255.
		for i in range(len(kp_set_negative)):
			kp_patch_set_negative = np.append(kp_patch_set_negative,img[kp_set_negative[i,1]-scale:kp_set_negative[i,1]+scale,kp_set_negative[i,0]-scale:kp_set_negative[i,0]+scale,:].reshape(1,scale*2,scale*2,3),axis=0)
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

def NonMaxSuppresion_4_kp_set(kp_set_list,threshold=25):
	"""
	新版NMS,根据score以及IOU来进行局部非最大值的抑制
	:param kp_set_list:输入的kp点的集合
	:param threshold:进行抑制的局部区域的大小设置,threshold=25表示在5个像素的范围内取最大值
	:return:返回经过局部非最大值抑制的kp的集合
	"""
	#输入的kp_set的类型为python的list类型
	#第一步,将kp_set转为ndarray类型,然后按照score从大到小进行排序
	kp_set_array = np.zeros(shape=(len(kp_set_list),3))
	for i in range(len(kp_set_list)):
		kp_set_array[i] = np.copy(kp_set_list[i])
	kp_set_array = kp_set_array[kp_set_array[:,2].argsort()]
		#在转换会list方便操作
	kp_set_list_sorted = []
	for i in range(len(kp_set_array)):
		kp_set_list_sorted.append([kp_set_array[i,0],kp_set_array[i,1],kp_set_array[i,2]])
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
			if (kp_set_list_sorted[i][0]-kp_popout[0]) ** 2 + (kp_set_list_sorted[i][1]-kp_popout[1]) ** 2 < threshold:
				del kp_set_list_sorted[i]
	print 'kp_set_list_afterNMS',kp_set_list_afterNMS[0],len(kp_set_list_afterNMS)
	return kp_set_list_afterNMS

def choose_kp_from_list(kp_set_afterNMS_list,quantity_to_choose=0):
	"""
	从经过NMS之后的kp_set里面按照score挑出得分靠前的一定数量的kp
	:param kp_set_afterNMS_list: 待选的原始kp_set,数据类型是list类型
	:param quantity_to_choose: 需要取出的排名靠前的kp的数量
	:return: 返回score前n名的kp点的信息
	"""
	#要使用numpy的argsort()首先要将list类型转化为numpy array类型
	kp_set_afterNMS_array = np.zeros(shape=(len(kp_set_afterNMS_list),3))
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
def use_TILDE_scale10(img_path_list):
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

	# 使用列表将两个维度不相同的矩阵打包在一起return
	kp_set_list = []
	for image_count in range(len(img_path_list)):
		img_test_rgb = plt.imread(img_path_list[image_count])
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
					kp_set.append([j, i, output_predict[0, 0]])
		print 'kp count from cnn without NMS:', count
		# kp_set_afterNMS_list = NMS_4_kp_set(kp_set, row_num, column_num, step=8, n_pixel=32, threshold=0.75)
		kp_set_afterNMS_list = NonMaxSuppresion_4_kp_set(kp_set,threshold=25)
		print 'NMS之后,保留:',len(kp_set_afterNMS_list),kp_set_afterNMS_list[-5:-1]
		kp_set_list.append(kp_set_afterNMS_list)
	# 释放gpu资源
	sess.close()
	print '一次结束!'
	return kp_set_list

def get_pyramid_of_image(img):
	img_scaleup1 = cv2.pyrUp(img)
	img_scaleup2 = cv2.pyrUp(img_scaleup1)
	img_scaledown1 = cv2.pyrDown(img)
	img_scaledown2 = cv2.pyrDown(img_scaledown1)
	print img_scaleup2.shape,img_scaleup1.shape,img.shape,img_scaledown1.shape,img_scaledown2.shape
	img_pyramid_list = [img_scaledown2,img_scaledown1,img,img_scaleup1,img_scaleup2]
	return img_pyramid_list

img_path_list = ['/home/javen/javenlib/images/bikes/img1.ppm',
                 '/home/javen/javenlib/images/bikes/img2.ppm']
img = plt.imread(img_path_list[0])/255.
x = get_pyramid_of_image(img[:,:,0])
print type(x[0])
