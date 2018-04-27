#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann

#将各个图片集所提取出来的patch拼接成一个patch矩阵作为训练集
image_classname_set = ['begin_0_kitti_campus_gray',
                       'begin_3_kitti_campus_gray',
                       'begin_6_kitti_campus_gray',
                       'begin_9_kitti_campus_gray',
                       'begin_12_kitti_campus_gray',
                       'begin_15_kitti_campus_gray',
                       'begin_18_kitti_campus_gray',
                       'begin_21_kitti_campus_gray',
                       'begin_24_kitti_campus_gray',
                       'begin_27_kitti_campus_gray',
                       'begin_30_kitti_campus_gray',
                       'begin_60_kitti_campus_gray',
                       'begin_63_kitti_campus_gray',
                       'begin_66_kitti_campus_gray',
                       'begin_69_kitti_campus_gray',
                       'begin_72_kitti_campus_gray',
                       'begin_75_kitti_campus_gray',
                       'begin_78_kitti_campus_gray',
                       'begin_81_kitti_campus_gray',
                       'begin_84_kitti_campus_gray',
                       'begin_87_kitti_campus_gray',
                       'begin_90_kitti_campus_gray',
                       'begin_120_kitti_campus_gray',
                       'begin_123_kitti_campus_gray',
                       'begin_126_kitti_campus_gray',
                       'begin_129_kitti_campus_gray',
                       'begin_132_kitti_campus_gray',
                       'begin_135_kitti_campus_gray',
                       'begin_138_kitti_campus_gray',
                       'begin_141_kitti_campus_gray',
                       'begin_144_kitti_campus_gray',
                       'begin_147_kitti_campus_gray',
                       'begin_150_kitti_campus_gray'
                       ]

scale =  8 #scale为patch的尺寸参数
#将每组图提取出来的patch按照positive和negative拼接起来,变成两大类
positive_sample = np.zeros(shape=(0,scale*2,scale*2,1))
negative_sample = np.zeros(shape=(0,scale*2,scale*2,1))
for i in range(len(image_classname_set)):
    positive_patch = np.load(image_classname_set[i]+'_positive_patches.npy')
    negative_patch = np.load(image_classname_set[i]+'_negative_patches.npy')
    # print 'positive:',positive_patch.shape
    # print 'negative:',negative_patch.shape
    positive_sample = np.append(positive_sample,positive_patch,axis=0)
    negative_sample = np.append(negative_sample,negative_patch,axis=0)

print 'positive_sample:',positive_sample.shape
print 'negative_sample:',negative_sample.shape
#将positive_sample与negative_sample合并为train_data,并创建label,使用的时候再进行打乱处理
train_data = np.append(positive_sample,negative_sample,axis=0)
# print train_data.shape
train_label = np.zeros(shape=(0,1))
for i in range(len(train_data)):
    if i < len(positive_sample):
        train_label = np.append(train_label,[[1]],axis=0)
    else:
        train_label = np.append(train_label,[[0]],axis=0)
# print train_label.shape

#将拼接完成的data和label保存
np.save('train_data_20180416.npy',train_data)
np.save('train_label_20180416.npy',train_label)
