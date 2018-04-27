#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann

#将各个图片集所提取出来的patch拼接成一个patch矩阵作为训练集
image_classname_set = ['begin_0_kitti_city_gray_0093',
                       'begin_3_kitti_city_gray_0093',
                       'begin_6_kitti_city_gray_0093',
                       'begin_9_kitti_city_gray_0093',
                       'begin_12_kitti_city_gray_0093',
                       'begin_15_kitti_city_gray_0093',
                       'begin_18_kitti_city_gray_0093',
                       'begin_21_kitti_city_gray_0093',
                       'begin_24_kitti_city_gray_0093',
                       'begin_27_kitti_city_gray_0093',
                       'begin_30_kitti_city_gray_0093',
                       'begin_33_kitti_city_gray_0093',
                       'begin_36_kitti_city_gray_0093',
                       'begin_39_kitti_city_gray_0093',
                       'begin_42_kitti_city_gray_0093',
                       'begin_45_kitti_city_gray_0093',
                       'begin_48_kitti_city_gray_0093',
                       'begin_51_kitti_city_gray_0093',
                       'begin_54_kitti_city_gray_0093',
                       'begin_57_kitti_city_gray_0093',
                       'begin_60_kitti_city_gray_0093',
                       'begin_100_kitti_city_gray_0093',
                       'begin_103_kitti_city_gray_0093',
                       'begin_106_kitti_city_gray_0093',
                       'begin_109_kitti_city_gray_0093',
                       'begin_112_kitti_city_gray_0093',
                       'begin_115_kitti_city_gray_0093',
                       'begin_118_kitti_city_gray_0093',
                       'begin_121_kitti_city_gray_0093',
                       'begin_124_kitti_city_gray_0093',
                       'begin_127_kitti_city_gray_0093',
                       'begin_130_kitti_city_gray_0093',
                       'begin_133_kitti_city_gray_0093',
                       'begin_136_kitti_city_gray_0093',
                       'begin_139_kitti_city_gray_0093',
                       'begin_142_kitti_city_gray_0093',
                       'begin_145_kitti_city_gray_0093',
                       'begin_148_kitti_city_gray_0093',
                       'begin_151_kitti_city_gray_0093',
                       'begin_154_kitti_city_gray_0093',
                       'begin_157_kitti_city_gray_0093',
                       'begin_160_kitti_city_gray_0093',
                       'begin_300_kitti_city_gray_0093',
                       'begin_303_kitti_city_gray_0093',
                       'begin_306_kitti_city_gray_0093',
                       'begin_309_kitti_city_gray_0093',
                       'begin_312_kitti_city_gray_0093',
                       'begin_315_kitti_city_gray_0093',
                       'begin_318_kitti_city_gray_0093',
                       'begin_321_kitti_city_gray_0093',
                       'begin_324_kitti_city_gray_0093',
                       'begin_327_kitti_city_gray_0093',
                       'begin_330_kitti_city_gray_0093',
                       'begin_333_kitti_city_gray_0093',
                       'begin_336_kitti_city_gray_0093',
                       'begin_339_kitti_city_gray_0093',
                       'begin_342_kitti_city_gray_0093',
                       'begin_345_kitti_city_gray_0093',
                       'begin_348_kitti_city_gray_0093',
                       'begin_351_kitti_city_gray_0093',
                       'begin_354_kitti_city_gray_0093',
                       'begin_357_kitti_city_gray_0093',
                       'begin_360_kitti_city_gray_0093',
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
