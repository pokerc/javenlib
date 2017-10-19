#!/usr/bin/python
#encoding=utf-8
import numpy as np
import caffe
import matplotlib.pyplot as plt
import sys
import os
import cv2
import cmath
import pyflann

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

def KeyPoint_convert_forOpencv2(keypoints):
	length = len(keypoints)
	points2f = np.zeros((length,2))
	for i in range(0,length):
		points2f[i,:] = keypoints[i].pt
	points = np.array(np.around(points2f),dtype='int')
	return points

def ph():
    print 'hello ph function!'
	

def convert_meanvalue():
	blob = caffe.proto.caffe_pb2.BlobProto()
	bin_mean = open('/home/javen/javenlib/lenet5_profiles/mean.binaryproto', 'rb' ).read()
	blob.ParseFromString(bin_mean)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	npy_mean = arr[0]
	np.save('/home/javen/javenlib/lenet5_profiles/minist_mean.npy', npy_mean )

def area_set_zero(img):
	#该函数输入为43*43×3的图像矩阵，函数功能是将外部方形和其内切圆之间的区域置零后输出新的外部方形
	img_inner_circle = np.copy(img)
	for i in range(0,43):
		for j in range(0,43):
			if (i-21)**2+(j-21)**2 > (0.5*29*cmath.sqrt(2).real)**2:
				img_inner_circle[i,j,:] = 0
	return img_inner_circle

def get_center_direction(img):
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

def image_rotate(img,degree):
	#图像旋转函数，degree若大于0则表示顺时针旋转，反之表示逆时针旋转
	row_num = img.shape[0]
	column_num = img.shape[1]
	radian = 1.0*degree/180.0*cmath.pi
#	print 'radian:',radian
	rotate_matrix = np.array([[cmath.cos(radian),-cmath.sin(radian),-0.5*(column_num-1)*cmath.cos(radian)+0.5*(row_num-1)*cmath.sin(radian)+0.5*(column_num-1)],
							  [cmath.sin(radian),cmath.cos(radian),-0.5*(column_num-1)*cmath.sin(radian)-0.5*(row_num-1)*cmath.cos(radian)+0.5*(row_num-1)],
							  [0,0,1]]).real
#	print 'rotate_matrix:',rotate_matrix
	old_position = np.zeros((3,1))
	old_position[2] = 1
	new_position = np.zeros((3,1))
	if len(img.shape) == 3:
		rotated_image = np.zeros((row_num,column_num,3))
		for i in range(0,row_num):
			for j in range(0,column_num):
				old_position[0] = j
				old_position[1] = i
				new_position = np.around(np.dot(rotate_matrix,old_position))
				if new_position[1]>=0 and new_position[1]<row_num and new_position[0]>=0 and new_position[0]<column_num:
					rotated_image[int(new_position[1]),int(new_position[0]),:] = img[i,j,:]
		for i in range(1,row_num-1):
			for j in range(1,column_num-1):
				if rotated_image[i,j,0] == 0:
					rotated_image[i,j,0] = 0.25*(rotated_image[i-1,j,0]+rotated_image[i+1,j,0]+rotated_image[i,j-1,0]+rotated_image[i,j+1,0])
					rotated_image[i,j,1] = 0.25*(rotated_image[i-1,j,1]+rotated_image[i+1,j,1]+rotated_image[i,j-1,1]+rotated_image[i,j+1,1])
					rotated_image[i,j,2] = 0.25*(rotated_image[i-1,j,2]+rotated_image[i+1,j,2]+rotated_image[i,j-1,2]+rotated_image[i,j+1,2])
	else:
		rotated_image = np.zeros((row_num,column_num))
		for i in range(0,row_num):
			for j in range(0,column_num):
				old_position[0] = j
				old_position[1] = i
				new_position = np.around( np.dot(rotate_matrix,old_position) )
				if new_position[1]>=0 and new_position[1]<row_num and new_position[0]>=0 and new_position[0]<column_num:
					rotated_image[int(new_position[1]),int(new_position[0])] = img[i,j]
#	print rotated_image[:,:,0]
	return rotated_image

def lenet5_compute(img,kp_pos,size_outter_square=43,size_inner_square=29,layer_name='pool2'):
	#此函数输入的img需要是使用caffe.io.load_image()读取的数据
	kp_num = len(kp_pos)
#	print kp_num
	caffe.set_mode_cpu()
	model_def = '/home/javen/javenlib/lenet5_profiles/lenet.prototxt'
	model_weights = '/home/javen/javenlib/lenet5_profiles/lenet_iter_10000.caffemodel'
	net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
	# load the mean ImageNet image (as distributed with Caffe) for subtraction
	mu = np.load('/home/javen/javenlib/lenet5_profiles/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1).mean(0).reshape(1)  # average over pixels to obtain the mean (BGR) pixel values
#	mu[0]=0
	print 'mu shape:',mu.shape,mu
#	print 'mean-subtracted values:', zip('BGR', mu)

	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
#	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

	# set the size of the input (we can skip this if we're happy
	#  with the default; we can also change it later, e.g., for different batch sizes)
	net.blobs['data'].reshape(kp_num,        # batch size
                                1,         # 3-channel (BGR) images
                               28, 28)  # image size is 227x227
	data_input = np.zeros((kp_num,1,28,28))
	for i in range(0,kp_num):
		outter_square = np.copy(img[kp_pos[i,1]-21:kp_pos[i,1]+21+1,kp_pos[i,0]-21:kp_pos[i,0]+21+1,:])
		#添加将中间区域置0的处理函数area_set_zero()
		degree = get_center_direction(area_set_zero(outter_square))
		rotated_outter_square = image_rotate(outter_square,-1*degree) #旋转后的43*43*3
#		print outter_square[:,:,0],'\n',degree,'°','\n',rotated_outter_square[:,:,0]
		inner_square = np.copy(rotated_outter_square[7:35+1,7:35+1,:].mean(2).reshape(29,29,1))
#		print 'inner_square:',inner_square.shape,inner_square[:,:]
		transformed_inner_square = transformer.preprocess('data',inner_square)
#		print 'trans:',transformed_inner_square
#		print 'inner square:',inner_square.shape,'\n','transformed inner square:',transformed_inner_square.shape
		data_input[i,:,:,:] = transformed_inner_square
	print data_input.shape

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = data_input

#	### perform classification
	output = net.forward()
#	print net.blobs['data'].data[499,0,:,:]
#	output_pool2 = output['pool2'][0]
	if layer_name == 'pool2':
		kp_des = net.blobs['pool2'].data
		kp_des = kp_des.reshape(kp_num,800)
	elif layer_name == 'conv2':
		kp_des = net.blobs['conv2'].data.reshape(kp_num,50*8*8)
	elif layer_name == 'ip1':
		kp_des = net.blobs['ip1'].data.reshape(kp_num,500)
	else:
		kp_des = net.blobs['ip2'].data.reshape(kp_num,10)
	print kp_des.shape
	return kp_des

def get_matrix_from_file(filename):
	f = open(filename,'r')
	rotation_matrix = np.zeros((3,3))
	for i in range(0,3):
		x = f.readline().split()
		rotation_matrix[i,0] = float(x[0])
		rotation_matrix[i,1] = float(x[1])
		rotation_matrix[i,2] = float(x[2])
	f.close()
	return rotation_matrix

def match_accuracy(img1,img1_kp_pos,img1_kp_des,img2,img2_kp_pos,img2_kp_des,rotation_matrix):
	flann = pyflann.FLANN()
#	if len(img1_kp_pos) <= len(img2_kp_pos):
#		kp_num = len(img1_kp_pos)
#	else:
#		kp_num = len(img2_kp_pos)
	kp_num = len(img2_kp_pos)
	print 'kp_num:',kp_num
	result,dists = flann.nn(img1_kp_des,img2_kp_des,1)#,algorithm="kmeans",branching=32,iterations=7, checks=16)
	pair_list = np.zeros((kp_num,3))
	pair_list[:,0] = dists
	pair_list[:,1] = result
	pair_list[:,2] = np.arange(kp_num)
	sorted_index = np.argsort(pair_list[:,0]) #按从小到大的顺序排序后的pair的下标
	match_count = 0
#	kp_num = 1000
	for i in range(0,kp_num):
		pair_index = sorted_index[i]
		kp1_index = int(pair_list[pair_index,1])
		kp2_index = int(pair_list[pair_index,2])
		kp1_pos = img1_kp_pos[kp1_index]
		kp2_pos = img2_kp_pos[kp2_index]
		rotated_kp1_pos = rotation_matrix.dot(np.append(kp1_pos,1))
		if kp2_pos[0] >= rotated_kp1_pos[0]-1 and kp2_pos[0] <= rotated_kp1_pos[0]+1 and kp2_pos[1] >= rotated_kp1_pos[1]-1 and kp2_pos[1] <= rotated_kp1_pos[1]+1:
			match_count += 1
	print 'match accuracy:',1.0*match_count/kp_num
	return 1.0*match_count/kp_num



























