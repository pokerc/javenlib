#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pyflann

#将各个图片集所提取出来的patch拼接成一个patch矩阵作为训练集
image_classname_set = ['begin_0_kitti_city_gray_0095',
                       'begin_3_kitti_city_gray_0095',
                       'begin_6_kitti_city_gray_0095',
                       'begin_9_kitti_city_gray_0095',
                       'begin_12_kitti_city_gray_0095',
                       'begin_15_kitti_city_gray_0095',
                       'begin_18_kitti_city_gray_0095',
                       'begin_21_kitti_city_gray_0095',
                       'begin_24_kitti_city_gray_0095',
                       'begin_27_kitti_city_gray_0095',
                       'begin_30_kitti_city_gray_0095',
                       'begin_33_kitti_city_gray_0095',
                       'begin_36_kitti_city_gray_0095',
                       'begin_39_kitti_city_gray_0095',
                       'begin_44_kitti_city_gray_0095',
                       'begin_45_kitti_city_gray_0095',
                       'begin_48_kitti_city_gray_0095',
                       'begin_51_kitti_city_gray_0095',
                       'begin_54_kitti_city_gray_0095',
                       'begin_57_kitti_city_gray_0095',
                       'begin_60_kitti_city_gray_0095',
                       'begin_120_kitti_city_gray_0095',
                       'begin_123_kitti_city_gray_0095',
                       'begin_126_kitti_city_gray_0095',
                       'begin_129_kitti_city_gray_0095',
                       'begin_132_kitti_city_gray_0095',
                       'begin_135_kitti_city_gray_0095',
                       'begin_138_kitti_city_gray_0095',
                       'begin_141_kitti_city_gray_0095',
                       'begin_144_kitti_city_gray_0095',
                       'begin_147_kitti_city_gray_0095',
                       'begin_150_kitti_city_gray_0095',
                       'begin_153_kitti_city_gray_0095',
                       'begin_156_kitti_city_gray_0095',
                       'begin_159_kitti_city_gray_0095',
                       'begin_230_kitti_city_gray_0095',
                       'begin_233_kitti_city_gray_0095',
                       'begin_236_kitti_city_gray_0095',
                       'begin_239_kitti_city_gray_0095',
                       'begin_242_kitti_city_gray_0095',
                       'begin_245_kitti_city_gray_0095',
                       'begin_248_kitti_city_gray_0095',
                       'begin_251_kitti_city_gray_0095',
                       'begin_254_kitti_city_gray_0095',
                       'begin_257_kitti_city_gray_0095',
                       'begin_260_kitti_city_gray_0095',
                       'begin_263_kitti_city_gray_0095',
                       'begin_266_kitti_city_gray_0095',
                       'begin_269_kitti_city_gray_0095'
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
