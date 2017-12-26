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

def NMS_4_points_set(kp_set,dist_threshold=1100):
	"""
	对一个points set 进行NMS,即非局部最大值抑制,即将点集合中比较彼此很靠近的点堆中只保留其中一个,其实并没有保留最大值,只是保留了其中一个值而已,不算完全的NMS
	:param kp_set: 需要进行NMS的点集
	:param dist_threshold: 相似距离的冗余度阈值,当距离大于1100,可判断为可以保留的点
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
			if matched_distances[j, 1] < dist_threshold:
				new_test_data = np.delete(test_data, j, axis=0)
				break
	return new_test_data

def quantity_test(kp_set1,kp_set2,groundtruth_matrix=None):
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
		if matched_distances[i] <= 1500:
			count1 += 1
	#然后在2中检测有没有与1重复的
	origin_data = np.copy(kp_set2)
	test_data = np.copy(kp_set1)
	matched_indices, matched_distances = flann.nn(origin_data.astype(np.float64), test_data.astype(np.float64), 1)
	count2 = 0
	# print 'matched_indices:', matched_indices
	# print 'matched_distances:', matched_distances
	for i in range(len(matched_distances)):
		if matched_distances[i] <= 1500:
			count2 += 1
	print kp_set1.shape,kp_set2.shape
	print 'count1:',count1,'count2:',count2
	print 'accuracy:',1.*(count1+count2)/total_num

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
	flann = pyflann.FLANN()
	kp_num = len(img1_kp_pos)
	origin_kp = np.copy(img2_kp_pos)
	origin_kp_des = np.copy(img2_kp_des)
	test_kp = np.copy(img1_kp_pos)
	test_kp_des = np.copy(img1_kp_des)
	print 'kp_num:',kp_num
	matched_index,matched_distance = flann.nn(origin_kp_des,test_kp_des,1)
	match_count = 0
	for i in range(kp_num):
		matched_kp_4_test_kp = origin_kp[matched_index[i]]
		# print 'matched_kp_4_test_kp:',matched_kp_4_test_kp
		transformed_kp_4_test_kp = rotation_matrix.dot(np.append(test_kp[i],1))
		# print 'transformed_kp_4_test_kp:',transformed_kp_4_test_kp
		if ((matched_kp_4_test_kp[0]-transformed_kp_4_test_kp[0])**2 + (matched_kp_4_test_kp[1]-transformed_kp_4_test_kp[1])**2) <= 16:
			match_count += 1
	print 'match_count:',match_count
	print 'match accuracy:',1.0*match_count/kp_num
	return 1.0*match_count/kp_num

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
		for i in range(32,row_num-32,1): #扫描的步长需要调整
			for j in range(32,column_num-32,1):
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
	return kp_set_afternms_list

#MSE版的use_TILDE增加多线程优化
def use_TILDE_multitread(img_path_list):
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
	kp_set_afternms_list = []
	for img_count in range(len(img_path_list)):
		img_test_rgb = plt.imread(img_path_list[img_count]) / 255.
		img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
		kp_set = np.zeros(shape=(0, 2))
		# 对图片进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
		row_num = plt.imread(img_path_list[0]).shape[0]
		column_num = plt.imread(img_path_list[0]).shape[1]
		for i in range(32, row_num - 32, 1):  # 扫描的步长需要调整
			for j in range(32, column_num - 32, 1):
				patch = np.copy(img_test_gray[i - 32:i + 32, j - 32:j + 32]).reshape(1, 64, 64, 1)
				output_predict = sess.run(output, feed_dict={tf_x: patch})
				if output_predict >= 0.5:
					# print output_predict
					kp_set = np.append(kp_set, [[j, i]], axis=0)
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

	#释放gpu资源
	sess.close()
	return kp_set_afternms_list

# img_path_list = ['/home/javen/javenlib/images/bikes/img1.ppm',
#                  '/home/javen/javenlib/images/bikes/img2.ppm']
# img = plt.imread(img_path_list[0])
# kp = use_TILDE(img_path_list)
# kp1 = kp[0]
# print kp1.shape