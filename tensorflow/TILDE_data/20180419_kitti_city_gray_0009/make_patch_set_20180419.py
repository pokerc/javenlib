#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann

#将各个图片集所提取出来的patch拼接成一个patch矩阵作为训练集
image_classname_set = ['begin_0_kitti_city_gray_0009',
                       'begin_3_kitti_city_gray_0009',
                       'begin_6_kitti_city_gray_0009',
                       'begin_9_kitti_city_gray_0009',
                       'begin_10_kitti_city_gray_0009',
                       'begin_13_kitti_city_gray_0009',
                       'begin_16_kitti_city_gray_0009',
                       'begin_19_kitti_city_gray_0009',
                       'begin_22_kitti_city_gray_0009',
                       'begin_25_kitti_city_gray_0009',
                       'begin_28_kitti_city_gray_0009',
                       'begin_31_kitti_city_gray_0009',
                       'begin_34_kitti_city_gray_0009',
                       'begin_37_kitti_city_gray_0009',
                       'begin_40_kitti_city_gray_0009',
                       'begin_43_kitti_city_gray_0009',
                       'begin_46_kitti_city_gray_0009',
                       'begin_49_kitti_city_gray_0009',
                       'begin_52_kitti_city_gray_0009',
                       'begin_55_kitti_city_gray_0009',
                       'begin_58_kitti_city_gray_0009',
                       'begin_61_kitti_city_gray_0009',
                       'begin_155_kitti_city_gray_0009',
                       'begin_158_kitti_city_gray_0009',
                       'begin_161_kitti_city_gray_0009',
                       'begin_164_kitti_city_gray_0009',
                       'begin_167_kitti_city_gray_0009',
                       'begin_170_kitti_city_gray_0009',
                       'begin_173_kitti_city_gray_0009',
                       'begin_176_kitti_city_gray_0009',
                       'begin_179_kitti_city_gray_0009',
                       'begin_182_kitti_city_gray_0009',
                       'begin_185_kitti_city_gray_0009',
                       'begin_188_kitti_city_gray_0009',
                       'begin_191_kitti_city_gray_0009',
                       'begin_194_kitti_city_gray_0009',
                       'begin_197_kitti_city_gray_0009',
                       'begin_200_kitti_city_gray_0009',
                       'begin_203_kitti_city_gray_0009',
                       'begin_206_kitti_city_gray_0009',
                       'begin_209_kitti_city_gray_0009',
                       'begin_212_kitti_city_gray_0009',
                       'begin_215_kitti_city_gray_0009',
                       'begin_218_kitti_city_gray_0009',
                       'begin_221_kitti_city_gray_0009',
                       'begin_224_kitti_city_gray_0009',
                       'begin_227_kitti_city_gray_0009',
                       'begin_230_kitti_city_gray_0009',
                       'begin_240_kitti_city_gray_0009',
                       'begin_243_kitti_city_gray_0009',
                       'begin_246_kitti_city_gray_0009',
                       'begin_249_kitti_city_gray_0009',
                       'begin_252_kitti_city_gray_0009',
                       'begin_255_kitti_city_gray_0009',
                       'begin_258_kitti_city_gray_0009',
                       'begin_261_kitti_city_gray_0009',
                       'begin_264_kitti_city_gray_0009',
                       'begin_267_kitti_city_gray_0009',
                       'begin_270_kitti_city_gray_0009',
                       'begin_273_kitti_city_gray_0009',
                       'begin_276_kitti_city_gray_0009',
                       'begin_279_kitti_city_gray_0009',
                       'begin_282_kitti_city_gray_0009',
                       'begin_285_kitti_city_gray_0009',
                       'begin_288_kitti_city_gray_0009',
                       'begin_291_kitti_city_gray_0009',
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

# #将拼接完成的data和label保存
# np.save('train_data_20180419.npy',train_data)
# np.save('train_label_20180419.npy',train_label)
