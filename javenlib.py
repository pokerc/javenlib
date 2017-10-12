#!/usr/bin/python
#encoding=utf-8
import numpy as np
import caffe
import matplotlib.pyplot as plt
import sys
import os
import cv2

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
	return points2f

def ph():
    print 'hello ph function!'
	

def convert_meanvalue():
	blob = caffe.proto.caffe_pb2.BlobProto()
	bin_mean = open('/home/javen/javenlib/lenet5_profiles/mean.binaryproto', 'rb' ).read()
	blob.ParseFromString(bin_mean)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	npy_mean = arr[0]
	np.save('/home/javen/javenlib/lenet5_profiles/minist_mean.npy', npy_mean )


def lenet5_compute(img,kp_pos):
	caffe.set_mode_cpu()
	model_def = '/home/javen/javenlib/lenet5_profiles/lenet.prototxt'
	model_weights = '/home/javen/javenlib/lenet5_profiles/lenet_iter_10000.caffemodel'
	net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
	# load the mean ImageNet image (as distributed with Caffe) for subtraction
	mu = np.load('/home/javen/javenlib/lenet5_profiles/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1).mean(0).reshape(1)  # average over pixels to obtain the mean (BGR) pixel values
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
#	net.blobs['data'].reshape(50,        # batch size
#                          3,         # 3-channel (BGR) images
#                          227, 227)  # image size is 227x227

	image = caffe.io.load_image('/home/javen/caffe-master/examples/images/cat.jpg')
	image = image.mean(2).reshape((360,480,1))
	print 'before transform:',image.shape
	transformed_image = transformer.preprocess('data', image)
	print 'after transform:',transformed_image.shape
#	plt.imshow(image)

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()
	print net.blobs['pool2'].data[0]
#	output_pool2 = output['pool2'][0]





























