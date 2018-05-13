#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann

#将各个图片集所提取出来的patch拼接成一个patch矩阵作为训练集
image_classname_set = ['begin_0_kitti_residential_gray_0061',
                       'begin_3_kitti_residential_gray_0061',
                       'begin_6_kitti_residential_gray_0061',
                       'begin_9_kitti_residential_gray_0061',
                       'begin_12_kitti_residential_gray_0061',
                       'begin_15_kitti_residential_gray_0061',
                       'begin_18_kitti_residential_gray_0061',
                       'begin_21_kitti_residential_gray_0061',
                       'begin_24_kitti_residential_gray_0061',
                       'begin_27_kitti_residential_gray_0061',
                       'begin_30_kitti_residential_gray_0061',
                       'begin_33_kitti_residential_gray_0061',
                       'begin_36_kitti_residential_gray_0061',
                       'begin_39_kitti_residential_gray_0061',
                       'begin_42_kitti_residential_gray_0061',
                       'begin_45_kitti_residential_gray_0061',
                       'begin_48_kitti_residential_gray_0061',
                       'begin_51_kitti_residential_gray_0061',
                       'begin_54_kitti_residential_gray_0061',
                       'begin_57_kitti_residential_gray_0061',
                       'begin_60_kitti_residential_gray_0061',
                       'begin_63_kitti_residential_gray_0061',
                       'begin_66_kitti_residential_gray_0061',
                       'begin_69_kitti_residential_gray_0061',
                       'begin_72_kitti_residential_gray_0061',
                       'begin_75_kitti_residential_gray_0061',
                       'begin_78_kitti_residential_gray_0061',
                       'begin_81_kitti_residential_gray_0061',
                       'begin_84_kitti_residential_gray_0061',
                       'begin_87_kitti_residential_gray_0061',
                       'begin_90_kitti_residential_gray_0061',
                       'begin_93_kitti_residential_gray_0061',
                       'begin_96_kitti_residential_gray_0061',
                       'begin_99_kitti_residential_gray_0061',
                       'begin_102_kitti_residential_gray_0061',
                       'begin_105_kitti_residential_gray_0061',
                       'begin_108_kitti_residential_gray_0061',
                       'begin_111_kitti_residential_gray_0061',
                       'begin_114_kitti_residential_gray_0061',
                       'begin_117_kitti_residential_gray_0061',
                       'begin_120_kitti_residential_gray_0061',
                       'begin_123_kitti_residential_gray_0061',
                       'begin_126_kitti_residential_gray_0061',
                       'begin_132_kitti_residential_gray_0061',
                       'begin_135_kitti_residential_gray_0061',
                       'begin_138_kitti_residential_gray_0061',
                       'begin_141_kitti_residential_gray_0061',
                       'begin_144_kitti_residential_gray_0061',
                       'begin_147_kitti_residential_gray_0061',
                       'begin_150_kitti_residential_gray_0061',
                       'begin_153_kitti_residential_gray_0061',
                       'begin_156_kitti_residential_gray_0061',
                       'begin_159_kitti_residential_gray_0061',
                       'begin_165_kitti_residential_gray_0061',
                       'begin_168_kitti_residential_gray_0061',
                       'begin_171_kitti_residential_gray_0061',
                       'begin_174_kitti_residential_gray_0061',
                       'begin_177_kitti_residential_gray_0061',
                       'begin_180_kitti_residential_gray_0061',
                       'begin_183_kitti_residential_gray_0061',
                       'begin_186_kitti_residential_gray_0061',
                       'begin_189_kitti_residential_gray_0061',
                       'begin_192_kitti_residential_gray_0061',
                       'begin_195_kitti_residential_gray_0061',
                       'begin_198_kitti_residential_gray_0061'
                       ]

scale =  8 #scale为patch的尺寸参数
#将每组图提取出来的patch按照positive和negative拼接起来,变成两大类
positive_sample = np.zeros(shape=(0,scale*2,scale*2,1))
negative_sample = np.zeros(shape=(0,scale*2,scale*2,1))
for i in range(len(image_classname_set)):
    positive_patch = np.load(image_classname_set[i]+'_positive_patches.npy')
    negative_patch = np.load(image_classname_set[i]+'_negative_patches.npy')
    # print 'positive:',positive_patch.shape,positive_patch.shape[3],positive_patch.mean()
    # if(positive_patch.shape[3] == 3):
    #     positive_patch = np.dot(positive_patch,[0.2989,0.5870,0.1140]).reshape(-1,16,16,1)
    #     negative_patch = np.dot(negative_patch,[0.2989,0.5870,0.1140]).reshape(-1,16,16,1)
    #     print 'positive patch shape:',positive_patch.shape
    # print 'negative:',negative_patch.shape
    positive_sample = np.append(positive_sample,positive_patch,axis=0)
    negative_sample = np.append(negative_sample,negative_patch,axis=0)

# positive_sample = np.append(positive_sample,positive_sample,axis=0) #positive的patch数量不够,将其自身复制一份

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
np.save('train_data_20180419.npy',train_data)
np.save('train_label_20180419.npy',train_label)
