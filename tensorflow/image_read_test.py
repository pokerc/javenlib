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

sift = cv2.SIFT(1400)
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

x = np.array([[1,2],[3,4],[5,3]])
y = np.array([[3,4],[2,4]])

where_tuple = np.where(x==3)
# where_tuple_index0 = where_tuple[0][0]
# where_tuple_index1 = where_tuple[1][0]
print where_tuple
print len(where_tuple[0])
# print where_tuple_index0,where_tuple_index1




for i in range(1400):
    where_tuple = np.where(img2_kp_coordinate == img1_kp_coordinate[i][0])
    if len(where_tuple[0]) == 0:
        continue
    for j in range(len(where_tuple[0])):
        row_index = where_tuple[0][j]
        if img2_kp_coordinate[row_index,0]==img1_kp_coordinate[i,0] and img2_kp_coordinate[row_index,1]==img1_kp_coordinate[i,1]:
            print 'ok:',img1_kp_coordinate[i],img2_kp_coordinate[row_index]
    # where_tuple_index0 = where_tuple[0][0]
    # where_tuple_index1 = where_tuple[1][0]

    # print where_tuple















