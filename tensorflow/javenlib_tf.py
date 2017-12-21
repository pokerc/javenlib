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

def get_kp_patch_set_positive(img_path_list,kp_set_positive):
	"""
	根据所给的图像，以及positive kp的坐标取出positive patch的集合
	:param img_path_list: 图像路径列表
	:param kp_set_positive: positive kp集合
	:return: 返回positive patch集合（由于函数中要进行kp是否可取patch的边界范围判断，所以可能得到的patch数目比positive kp的数目少）
	"""
	#无法取到patch的kp点的去除
	rows_num,columns_num = plt.imread(img_path_list[0]).shape[0:2]
	for i in range(len(kp_set_positive)-1,-1,-1):
		if kp_set_positive[i,1] < 32 or kp_set_positive[i,1] > rows_num-32 or kp_set_positive[i,0] < 32 or kp_set_positive[i,0] > columns_num-32:
			kp_set_positive = np.delete(kp_set_positive,i,axis=0)
	# print 'kp_set_positive:',kp_set_positive.shape,kp_set_positive
	#取出kp_set_positive所对应的patch集合
	kp_patch_set_positive = np.zeros(shape=(1,64,64,3))
	for img_count in range(len(img_path_list)):
		img = plt.imread(img_path_list[img_count])/255.
		for i in range(len(kp_set_positive)):
			kp_patch_set_positive = np.append(kp_patch_set_positive,img[kp_set_positive[i,1]-32:kp_set_positive[i,1]+32,kp_set_positive[i,0]-32:kp_set_positive[i,0]+32,:].reshape(1,64,64,3),axis=0)
	kp_patch_set_positive = np.delete(kp_patch_set_positive,0,axis=0)
	# print kp_patch_set_positive.shape
	return kp_patch_set_positive

def get_kp_patch_set_negative(img_path_list,kp_set_negative):
	#无法取到patch的kp点的去除
	rows_num,columns_num = plt.imread(img_path_list[0]).shape[0:2]
	for i in range(len(kp_set_negative)-1,-1,-1):
		if kp_set_negative[i,1] < 32 or kp_set_negative[i,1] > rows_num-32 or kp_set_negative[i,0] < 32 or kp_set_negative[i,0] > columns_num-32:
			kp_set_negative = np.delete(kp_set_negative,i,axis=0)
	#取出kp_set_negative所对应的patch集合
	kp_patch_set_negative = np.zeros(shape=(1,64,64,3))
	for img_count in range(len(img_path_list)):
		img = plt.imread(img_path_list[img_count])/255.
		for i in range(len(kp_set_negative)):
			kp_patch_set_negative = np.append(kp_patch_set_negative,img[kp_set_negative[i,1]-32:kp_set_negative[i,1]+32,kp_set_negative[i,0]-32:kp_set_negative[i,0]+32,:].reshape(1,64,64,3),axis=0)
	kp_patch_set_negative = np.delete(kp_patch_set_negative,0,axis=0)
	return kp_patch_set_negative

def shuffle_data_and_label(train_data,train_label):
	x = np.arange(len(train_data))
	np.random.shuffle(x)
	shuffled_train_data = train_data[x]
	shuffled_train_label = train_label[x]
	return (shuffled_train_data,shuffled_train_label)

def rgb2gray_train_data(train_data):
	"""
	将train_data从rgb模式转换为gray模式
	:param train_data: rgb模式的train_data,其维度必须为(?,64,64,3)的形式
	:return: 返回转换为gray模式的train_data,转换后的维度为(?,64,64,1)
	"""
	if train_data.shape[3] != 3:
		print 'Error: 输入数据的维度必须满足(?,64,64,3)'
		exit()
	train_data_gray = np.dot(train_data[...,:3],[0.2989,0.5870,0.1140])
	return train_data_gray.reshape(len(train_data),64,64,1)

def NMS_4_points_set(kp_set):
	"""
	对一个points set 进行NMS,即非局部最大值抑制,即将点集合中比较彼此很靠近的点堆中只保留其中一个,其实并没有保留最大值,只是保留了其中一个值而已,不算完全的NMS
	:param kp_set: 需要进行NMS的点集
	:return: 返回进过NMS过滤的点集
	"""
	# 进行非局部最大值抑制,即去除聚在一起的冗余的点,保留其中一个即可
	flann = pyflann.FLANN()
	new_test_data = np.copy(kp_set)
	for i in range(len(kp_set)):
		origin_data = np.copy(new_test_data)
		test_data = np.copy(new_test_data)
		matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 2,
													  algorithm="kmeans", branching=32, iterations=7, checks=16)
		for j in range(len(test_data) - 1, -1, -1):
			if matched_distances[j, 1] < 1024:
				new_test_data = np.delete(test_data, j, axis=0)
				break
	return new_test_data

# #提取图片集合中的positive patches和negative patches
# img_path_list = ['/home/javen/javenlib/images/leuven/img1.ppm',
# 		 '/home/javen/javenlib/images/leuven/img2.ppm',
# 		 '/home/javen/javenlib/images/leuven/img3.ppm',
# 		 '/home/javen/javenlib/images/leuven/img4.ppm',
# 		 '/home/javen/javenlib/images/leuven/img5.ppm']
#
# kp_set_raw = get_kp_set_raw(img_path_list)
# kp_set_positive = get_kp_set_positive(kp_set_raw)
# print 'kp_set_positive:',kp_set_positive.shape
# kp_patch_set_positive = get_kp_patch_set_positive(img_path_list,kp_set_positive)
# print 'kp_patch_set_positive:',kp_patch_set_positive.shape
# # show_patch_set(kp_patch_set_positive)
# # show_kp_set(img_path_list[4],kp_set_positive)
# #保存positive patch集合
# # np.save('/home/javen/javenlib/tensorflow/TILDE_data/'+'leuven'+'_positive_patches.npy',kp_patch_set_positive)
#
# kp_set_negative = get_kp_set_negative(kp_set_raw)
# print 'kp_set_nagative:',kp_set_negative.shape
# kp_patch_set_negative = get_kp_patch_set_negative(img_path_list,kp_set_negative)
# print 'kp_patch_set_negative:',kp_patch_set_negative.shape
# # show_patch_set(kp_patch_set_negative)
# #保存negative patch集合
# # np.save('/home/javen/javenlib/tensorflow/TILDE_data/'+'leuven'+'_negative_patches.npy',kp_patch_set_negative)
