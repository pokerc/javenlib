#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann

#将各个图片集所提取出来的patch拼接成一个patch矩阵作为训练集
image_classname_set = ['begin_50_kitti_city_gray_0014',
                       'begin_53_kitti_city_gray_0014',
                       'begin_56_kitti_city_gray_0014',
                       'begin_59_kitti_city_gray_0014',
                       'begin_62_kitti_city_gray_0014',
                       'begin_65_kitti_city_gray_0014',
                       'begin_68_kitti_city_gray_0014',
                       'begin_71_kitti_city_gray_0014',
                       'begin_74_kitti_city_gray_0014',
                       'begin_77_kitti_city_gray_0014',
                       'begin_80_kitti_city_gray_0014',
                       'begin_83_kitti_city_gray_0014',
                       'begin_86_kitti_city_gray_0014',
                       'begin_89_kitti_city_gray_0014',
                       'begin_92_kitti_city_gray_0014',
                       'begin_95_kitti_city_gray_0014',
                       'begin_98_kitti_city_gray_0014',
                       'begin_101_kitti_city_gray_0014',
                       'begin_104_kitti_city_gray_0014',
                       'begin_107_kitti_city_gray_0014',
                       'begin_110_kitti_city_gray_0014',
                       'begin_113_kitti_city_gray_0014',
                       'begin_116_kitti_city_gray_0014',
                       'begin_119_kitti_city_gray_0014',
                       'begin_122_kitti_city_gray_0014',
                       'begin_125_kitti_city_gray_0014',
                       'begin_128_kitti_city_gray_0014',
                       'begin_131_kitti_city_gray_0014',
                       'begin_134_kitti_city_gray_0014',
                       'begin_137_kitti_city_gray_0014',
                       'begin_140_kitti_city_gray_0014',
                       'begin_143_kitti_city_gray_0014',
                       'begin_146_kitti_city_gray_0014',
                       'begin_149_kitti_city_gray_0014'
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
