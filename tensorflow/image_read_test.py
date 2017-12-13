#encoding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import javenlib_tf


sess = tf.Session()
# img_plt = plt.imread('./animal/train/cat/cat1.jpg')
# print img_plt.shape
# img_plt = img_plt/255.
#
# # img_resized = tf.image.resize_image_with_crop_or_pad(img_plt,120,120)
# img_resized = tf.image.resize_images(img_plt,[120,120])
# img_resized = tf.to_float(img_resized)
# sess = tf.Session()
# # sess.run(img_resized)
#
# # print 'x:',x.shape
# print img_resized.shape,type(img_resized.eval(session=sess))
# img_resized_2np = img_resized.eval(session=sess)
# print img_resized_2np.shape

# #分两个窗口来显示两幅图像
# plt.figure()
# plt.imshow(img_plt)
# plt.figure()
# plt.imshow(img_resized_2np)
# plt.show()

# #一个窗口中显示两幅图像
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(img_plt)
# plt.subplot(1,2,2)
# plt.imshow(img_resized_2np)
# plt.show()

# a = './animal/test/cat/cat'
# for count in range(1,10):
#     b = a+str(count)+'.jpg'
#     img_plt = plt.imread(b)
#     if img_plt.shape[1] != 256:
#         print 'no'
#         img_plt = tf.image.resize_images(img_plt,[256,256]).eval(session=sess)
#         print img_plt.shape,type(img_plt)
#         plt.figure()
#         plt.imshow(img_plt)
#     print b
# plt.show()


# img_plt = plt.imread('./animal/test/cat/cat0.jpg')
# img_plt = tf.image.resize_images(img_plt,[150,150],method=1).eval(session=sess)
# img_plt = tf.image.rgb_to_grayscale(img_plt).eval(session=sess)
# img_plt = img_plt.reshape((150,150))
# print img_plt.shape
# plt.figure()
# plt.imshow(img_plt,cmap='gray')
# plt.show()

##########################################################################
# #制作train数据
# animal_train_data = np.zeros(shape=(100,256,256,1))
# animal_train_label = np.zeros(shape=(100,2))
# count = 0
# for step in range(50):
#     path = './animal/train/cat/cat'+str(step)+'.jpg'
#     img_plt = plt.imread(path)
#     print img_plt.shape
#     if img_plt.shape[1] != 256:
#         img_plt = tf.image.resize_images(img_plt,[256,256],method=1).eval(session=sess)
#     img_plt = tf.image.rgb_to_grayscale(img_plt).eval(session=sess)
#     print img_plt.shape,img_plt.dtype
#     animal_train_data[count] = np.copy(img_plt)
#     animal_train_label[count,0] = 1
#     count += 1
#
#     path = './animal/train/dog/dog' + str(step) + '.jpg'
#     img_plt = plt.imread(path)
#     print img_plt.shape
#     if img_plt.shape[1] != 256:
#         img_plt = tf.image.resize_images(img_plt, [256, 256], method=1).eval(session=sess)
#     img_plt = tf.image.rgb_to_grayscale(img_plt).eval(session=sess)
#     print img_plt.shape, img_plt.dtype
#     animal_train_data[count] = np.copy(img_plt)
#     animal_train_label[count,1] = 1
#     count += 1
# # print animal_train_data.shape,animal_train_data.dtype
# # plt.figure()
# # print animal_train_data[0].shape
# # plt.imshow(animal_train_data[0].reshape((256,256)),cmap='gray')
# # plt.show()
# # print animal_train_label.shape,animal_train_label[0:9]
# np.save('./animal/animal_train_data.npy',animal_train_data)
# np.save('./animal/animal_train_label.npy',animal_train_label)

###########################################################################
# #制作test数据
# animal_test_data = np.zeros(shape=(20,256,256,1))
# animal_test_label = np.zeros(shape=(20,2))
# count = 0
# for step in range(10):
#     path = './animal/test/cat/cat'+str(step)+'.jpg'
#     img_plt = plt.imread(path)
#     print img_plt.shape
#     if img_plt.shape[1] != 256:
#         img_plt = tf.image.resize_images(img_plt,[256,256],method=1).eval(session=sess)
#     img_plt = tf.image.rgb_to_grayscale(img_plt).eval(session=sess)
#     print img_plt.shape,img_plt.dtype
#     animal_test_data[count] = np.copy(img_plt)
#     animal_test_label[count,0] = 1
#     count += 1
#
#     path = './animal/test/dog/dog' + str(step) + '.jpg'
#     img_plt = plt.imread(path)
#     print img_plt.shape
#     if img_plt.shape[1] != 256:
#         img_plt = tf.image.resize_images(img_plt, [256, 256], method=1).eval(session=sess)
#     img_plt = tf.image.rgb_to_grayscale(img_plt).eval(session=sess)
#     print img_plt.shape, img_plt.dtype
#     animal_test_data[count] = np.copy(img_plt)
#     animal_test_label[count,1] = 1
#     count += 1
# plt.figure()
# plt.imshow(animal_test_data[16].reshape(256,256),cmap='gray')
# plt.show()
# print animal_test_label
# np.save('./animal/animal_test_data.npy',animal_test_data)
# np.save('./animal/animal_test_label.npy',animal_test_label)

# ##############################################################################
# #各方法得到的图像数据类型测试 opencv读取图形的数据类型也是uint8
# img1_uint8 = plt.imread('/home/javen/LIFT-master/data/testimg/img1.jpg')
# print 'img1_uint8:',img1_uint8.shape,img1_uint8.dtype
# img1_float64 = img1_uint8/255.
# print 'img1_float64:',img1_float64.shape,img1_float64.dtype
# img1_gray_float64 = tf.image.rgb_to_grayscale(img1_float64).eval(session=sess).reshape(1200,1920)
# print 'img1_gray:',img1_gray_float64.shape,img1_gray_float64.dtype
# # plt.ion()
# # plt.figure(1)
# # plt.imshow(img1_uint8)
# # plt.pause(0.3)
# # plt.close(1)
# # plt.figure(2)
# # plt.imshow(img1_float64)
# # plt.pause(0.3)
# # plt.close(2)
# # plt.figure(3)
# # plt.imshow(img1_gray_float64,cmap='gray')
# # plt.ioff()
# # plt.show()
# x = np.array([300,400])
# print x
# new_img1 = np.copy(img1_gray_float64)
# new_img1[x[0]-32:x[0]+32,x[1]-32:x[1]+32] = 1
# plt.figure()
# plt.imshow(new_img1,cmap='gray')
# plt.figure()
# plt.imshow(img1_float64)
# plt.show()
#################################################################

img1 = plt.imread('/home/javen/javenlib/images/leuven/img1.ppm')
img2 = plt.imread('/home/javen/javenlib/images/leuven/img2.ppm')
img3 = plt.imread('/home/javen/javenlib/images/leuven/img3.ppm')
img4 = plt.imread('/home/javen/javenlib/images/leuven/img4.ppm')
img5 = plt.imread('/home/javen/javenlib/images/leuven/img5.ppm')
img6 = plt.imread('/home/javen/javenlib/images/leuven/img6.ppm')
# img1 = tf.image.rgb_to_grayscale(
#     plt.imread('/home/javen/javenlib/images/leuven/img1.ppm')).eval(session=sess)
# print img1.shape,img1.dtype
#
#
# sift = cv2.SIFT(400)
# kp = sift.detect(img1)
# points = javenlib_tf.KeyPoint_convert_forOpencv2(kp)
#
# new_img1 = np.copy(img1)
# # new_img1[points[0][1]-32:points[0][1]+32,points[0][0]-32:points[0][0]+32]=255
# # plt.figure()
# # plt.imshow(new_img1)
# # plt.show()
# coordinate_x = 1
# coordinate_y = 0
# img1_patch_data = np.zeros(shape=(200,64,64,3))
# count = 0
# plt.ion()
# for i in range(20):
#     if points[i][1] > 32 and points[i][1] < 600-32 and points[i][0] > 32 and points[i][0] < 900-32:
#         count += 1
#         img1_patch_data[i] = np.copy(img1[points[i][1]-32:points[i][1]+32,points[i][0]-32:points[i][0]+32])
#         new_img1 = np.copy(img1)
#         new_img1[points[i][1]-32:points[i][1]+32,points[i][0]-32:points[i][0]+32,0] = 200
#         plt.figure(1)
#         plt.imshow(new_img1)
#         plt.figure(2)
#         plt.imshow(img1_patch_data[i]/255.)
#         plt.pause(5)
#         plt.close(1)
#         plt.close(2)
#
#
# print count

sift = cv2.SIFT(400)
img1_kp = sift.detect(img1)
img1_kp_coordinate = javenlib_tf.KeyPoint_convert_forOpencv2(img1_kp)
img2_kp = sift.detect(img2)
img2_kp_coordinate = javenlib_tf.KeyPoint_convert_forOpencv2(img2_kp)
img3_kp = sift.detect(img3)
img3_kp_coordinate = javenlib_tf.KeyPoint_convert_forOpencv2(img3_kp)
img4_kp = sift.detect(img4)
img4_kp_coordinate = javenlib_tf.KeyPoint_convert_forOpencv2(img4_kp)
img5_kp = sift.detect(img5)
img5_kp_coordinate = javenlib_tf.KeyPoint_convert_forOpencv2(img5_kp)
img6_kp = sift.detect(img6)
img6_kp_coordinate = javenlib_tf.KeyPoint_convert_forOpencv2(img6_kp)

x = np.array([[1,2],[3,4],[5,3],[1,3]])
y = np.array([[3,4],[2,4]])

where_tuple = np.where(x[:,0]==1)
# where_tuple_index0 = where_tuple[0][0]
# where_tuple_index1 = where_tuple[1][0]
print where_tuple
print where_tuple[0]
print len(where_tuple[0])
# print where_tuple_index0,where_tuple_index1


###########################################################
# 将取出具有重复性的点的算法写为一个函数
def get_kp_set(img1_kp_coordinate,img2_kp_coordinate):
    kp_set = np.zeros(shape=(1, 2))
    print '########################同一列'
    count = 0
    for i in range(400):
        where_tuple = np.where(img2_kp_coordinate[:,0] == img1_kp_coordinate[i,0])
        if len(where_tuple[0]) == 0:
            continue
        print 'where_tuple[0]:',where_tuple[0],'img1_row:',i,img1_kp_coordinate[i]
        for j in range(len(where_tuple[0])):
            row_index = where_tuple[0][j]
            if (img2_kp_coordinate[row_index,0]>=img1_kp_coordinate[i,0]-1 and img2_kp_coordinate[row_index,0]<=img1_kp_coordinate[i,0]+1) and (img2_kp_coordinate[row_index,1]>=img1_kp_coordinate[i,1]-1 and img2_kp_coordinate[row_index,1]<=img1_kp_coordinate[i,1]+1):
                count += 1
                print 'ok:',img1_kp_coordinate[i],img2_kp_coordinate[row_index],'img2_row:',row_index
                kp_set = np.vstack([kp_set, img1_kp_coordinate[i]])
    print 'count0:',count

    print '########################前一列'
    count = 0
    for i in range(400):
        where_tuple = np.where(img2_kp_coordinate == img1_kp_coordinate[i][0]-1)
        if len(where_tuple[0]) == 0:
            continue
        for j in range(len(where_tuple[0])):
            row_index = where_tuple[0][j]
            if (img2_kp_coordinate[row_index,0]>=img1_kp_coordinate[i,0]-1 and img2_kp_coordinate[row_index,0]<=img1_kp_coordinate[i,0]+1) and (img2_kp_coordinate[row_index,1]>=img1_kp_coordinate[i,1]-1 and img2_kp_coordinate[row_index,1]<=img1_kp_coordinate[i,1]+1):
                count += 1
                print 'ok:',img1_kp_coordinate[i],img2_kp_coordinate[row_index]
                kp_set = np.vstack([kp_set,img1_kp_coordinate[i]])
    print 'count-1:',count

    print '########################后一列'
    count = 0
    for i in range(400):
        where_tuple = np.where(img2_kp_coordinate == img1_kp_coordinate[i][0]+1)
        if len(where_tuple[0]) == 0:
            continue
        for j in range(len(where_tuple[0])):
            row_index = where_tuple[0][j]
            if (img2_kp_coordinate[row_index,0]>=img1_kp_coordinate[i,0]-1 and img2_kp_coordinate[row_index,0]<=img1_kp_coordinate[i,0]+1) and (img2_kp_coordinate[row_index,1]>=img1_kp_coordinate[i,1]-1 and img2_kp_coordinate[row_index,1]<=img1_kp_coordinate[i,1]+1):
                count += 1
                print 'ok:',img1_kp_coordinate[i],img2_kp_coordinate[row_index]
                kp_set = np.vstack([kp_set, img1_kp_coordinate[i]])
    print 'count+1:',count
    for i in range(len(kp_set) - 1, -1, -1): #将无法取出patch的边缘kp点进行去除处理
        if kp_set[i, 0] < 32 or kp_set[i, 0] > 900 - 32 or kp_set[i, 1] < 32 or kp_set[i, 1] > 600 - 32:
            kp_set = np.delete(kp_set, i, axis=0)
    return kp_set.astype(np.int)
###########################################################


###########################################################
#将取出具有重复性的点周围的patch写为一个函数
def get_kp_patch_set(img,kp_set):
    kp_patch_set = np.zeros(shape=(len(kp_set), 64, 64, 3))
    for i in range(len(kp_set)):
        kp_patch_set[i] = np.copy(img[kp_set[i, 1] - 32:kp_set[i, 1] + 32, kp_set[i, 0] - 32:kp_set[i, 0] + 32])
    return kp_patch_set/255.
###########################################################

kp_set = get_kp_set(img1_kp_coordinate,img2_kp_coordinate)
print kp_set.shape
print kp_set

kp_patch_set = get_kp_patch_set(img1,kp_set)
a = np.append(kp_patch_set,kp_patch_set,axis=0)
print a.shape

kp_set_12 = get_kp_set(img1_kp_coordinate,img2_kp_coordinate)
kp_patch_set_12 = get_kp_patch_set(img1,kp_set_12)
kp_set_34 = get_kp_set(img3_kp_coordinate,img4_kp_coordinate)
kp_patch_set_34 = get_kp_patch_set(img3,kp_set_34)
kp_set_56 = get_kp_set(img5_kp_coordinate,img6_kp_coordinate)
kp_patch_set_56 = get_kp_patch_set(img5,kp_set_56)

kp_patch_set = np.copy(kp_patch_set_12)
kp_patch_set = np.append(kp_patch_set,kp_patch_set_34,axis=0)
kp_patch_set = np.append(kp_patch_set,kp_patch_set_56,axis=0)
print kp_patch_set.shape

# #保存为positive samples,即具有重复性的patch集
# np.save('./TILDE_data/positive_samples.npy',kp_patch_set)


# plt.ion()
# for i in range(len(kp_patch_set)):
#     plt.figure()
#     plt.imshow(kp_patch_set[i])
#     plt.pause(0.5)
#     plt.close()
# plt.ioff()

def get_negative_patch_set(img,kp_set_positive):
    """
    从给定的img中取出不具有重复性的点的patch集
    :param img:
    :return: 返回不具有重复性的patch集
    """
    sift = cv2.SIFT(400)
    kp = sift.detect(img)
    kp_coordinate = javenlib_tf.KeyPoint_convert_forOpencv2(kp)
    # print kp_coordinate
    kp_set_negative = np.zeros(shape=(1,2))
    for i in range(len(kp_coordinate)):
        where_tuple = np.where(kp_set_positive[:,0]==kp_coordinate[i,0])
        if len(where_tuple[0]) == 0:
            kp_set_negative = np.append(kp_set_negative,kp_coordinate[i].reshape(1,2),axis=0)
    for i in range(len(kp_set_negative) - 1, -1, -1): #将无法取出patch的边缘kp点进行去除处理
        if kp_set_negative[i, 0] < 32 or kp_set_negative[i, 0] > 900 - 32 or kp_set_negative[i, 1] < 32 or kp_set_negative[i, 1] > 600 - 32:
            kp_set_negative = np.delete(kp_set_negative, i, axis=0)
    kp_set_negative = kp_set_negative.astype(np.int)
    print 'kp_set_negative:', kp_set_negative.shape
    kp_patch_set_negative = np.zeros(shape=(len(kp_set_negative),64,64,3))
    for i in range(len(kp_set_negative)):
        kp_patch_set_negative[i] = np.copy(img[kp_set_negative[i, 1] - 32:kp_set_negative[i, 1] + 32, kp_set_negative[i, 0] - 32:kp_set_negative[i, 0] + 32])
    return kp_patch_set_negative/255.

kp_set_positive = np.append(kp_set_12,kp_set_34,axis=0)
kp_set_positive = np.append(kp_set_positive,kp_set_56,axis=0)
kp_patch_set_negative = get_negative_patch_set(img1,kp_set_positive)

# #保存kp_patch_set_negative到文件
# np.save('./TILDE_data/negative_samples.npy',kp_patch_set_negative)

# #显示kp_patch_set_negative
# plt.ion()
# for i in range(len(kp_patch_set_negative)):
#     plt.figure()
#     plt.imshow(kp_patch_set_negative[i])
#     plt.pause(0.5)
#     plt.close()
# plt.ioff()









