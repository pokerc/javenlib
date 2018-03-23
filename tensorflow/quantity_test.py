#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf
import cv2

##############################################################
##############################################################
##############################################################
sift = cv2.SIFT(500)
img_path_list = ['/home/javen/javenlib/images/bark/img1.ppm',
                 '/home/javen/javenlib/images/bark/img2.ppm']
tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/javenlib/images/bark/H1to2p')
imga = plt.imread(img_path_list[0])
imgb = plt.imread(img_path_list[1])

img_kp_set_afternms_list = javenlib_tf.use_TILDE_scale8_withpyramid(img_path_list)
print 'cnn done!'
imga_kp_cnn = javenlib_tf.choose_kp_from_list(img_kp_set_afternms_list[0],quantity_to_choose=250)
# imga_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn)
# imga_kp_cnn_obj,imga_kp_cnn_des = sift.compute(imga,imga_kp_cnn_obj)

imgb_kp_cnn = javenlib_tf.choose_kp_from_list(img_kp_set_afternms_list[1],quantity_to_choose=250)
# imgb_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn)
# imgb_kp_cnn_obj,imgb_kp_cnn_des = sift.compute(imgb,imgb_kp_cnn_obj)

imga_kp_sift_obj = sift.detect(imga)
imga_kp_sift_4dlist = javenlib_tf.KeyPoint_from_siftObjList_to_4dlist(imga_kp_sift_obj)
imga_kp_sift_4dlist_afterNMS = javenlib_tf.NonMaxSuppresion_4_kp_set(imga_kp_sift_4dlist,threshold=25)
imga_kp_sift_chosen = javenlib_tf.choose_kp_from_list(imga_kp_sift_4dlist_afterNMS,quantity_to_choose=250)
# imga_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj)
# # imga_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_sift)
# imga_kp_sift_obj,imga_kp_sift_des = sift.compute(imga,imga_kp_sift_obj)

imgb_kp_sift_obj = sift.detect(imgb)
imgb_kp_sift_4dlist = javenlib_tf.KeyPoint_from_siftObjList_to_4dlist(imgb_kp_sift_obj)
imgb_kp_sift_4dlist_afterNMS = javenlib_tf.NonMaxSuppresion_4_kp_set(imgb_kp_sift_4dlist,threshold=25)
imgb_kp_sift_chosen = javenlib_tf.choose_kp_from_list(imgb_kp_sift_4dlist_afterNMS,quantity_to_choose=250)
# imgb_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj)
# # imgb_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_sift)
# imgb_kp_sift_obj,imgb_kp_sift_des = sift.compute(imgb,imgb_kp_sift_obj)

javenlib_tf.quantity_test(imga_kp_sift_chosen,imgb_kp_sift_chosen,groundtruth_matrix=tranform_matrix)
javenlib_tf.quantity_test(imga_kp_cnn,imgb_kp_cnn,groundtruth_matrix=tranform_matrix)

# print imga_kp_cnn_des.shape,imgb_kp_cnn_des.shape,imga_kp_sift_des.shape,imgb_kp_sift_des.shape
# print 'sift shape:',imga_kp_sift.shape,imgb_kp_sift.shape
# javenlib_tf.match_accuracy(imga_kp_sift,imga_kp_sift_des,imgb_kp_sift,imgb_kp_sift_des,tranform_matrix)
# print 'cnn shape:',imga_kp_cnn.shape,imgb_kp_cnn.shape
# javenlib_tf.match_accuracy(imga_kp_cnn,imga_kp_cnn_des,imgb_kp_cnn,imgb_kp_cnn_des,tranform_matrix)

javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_sift_chosen)
javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_cnn,pixel_size=14)


# ###########################################################################################
# ##################################   针对EFDataset 进行修改   ###############################
# ###########################################################################################
# sift = cv2.SIFT(1500)
# img_path_list = ['/home/javen/datasets/EFDataset/rushmore/test/image_color/img1.png',
#                  '/home/javen/datasets/EFDataset/rushmore/test/image_color/img2.png']
# tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/datasets/EFDataset/rushmore/test/homography/H1to2p')
#
# imga = (plt.imread(img_path_list[0])*255).astype(np.uint8)
# imgb = (plt.imread(img_path_list[1])*255).astype(np.uint8)
# img_kp_set_afternms_list = javenlib_tf.use_TILDE_scale10(img_path_list)
# print 'cnn done!'
# imga_kp_cnn = javenlib_tf.choose_kp_from_list(img_kp_set_afternms_list[0],quantity_to_choose=0)
# imga_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_cnn)
# imga_kp_cnn_obj,imga_kp_cnn_des = sift.compute(imga,imga_kp_cnn_obj)
#
# imgb_kp_cnn = javenlib_tf.choose_kp_from_list(img_kp_set_afternms_list[1],quantity_to_choose=0)
# imgb_kp_cnn_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_cnn)
# imgb_kp_cnn_obj,imgb_kp_cnn_des = sift.compute(imgb,imgb_kp_cnn_obj)
#
# imga_kp_sift_obj = sift.detect(imga)
# imga_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imga_kp_sift_obj)
# imga_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imga_kp_sift)
# imga_kp_sift_obj,imga_kp_sift_des = sift.compute(imga,imga_kp_sift_obj)
#
# imgb_kp_sift_obj = sift.detect(imgb)
# imgb_kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(imgb_kp_sift_obj)
# imgb_kp_sift_obj = javenlib_tf.KeyPoint_reverse_convert_forOpencv2(imgb_kp_sift)
# imgb_kp_sift_obj,imgb_kp_sift_des = sift.compute(imgb,imgb_kp_sift_obj)
#
# javenlib_tf.quantity_test(imga_kp_sift,imgb_kp_sift,groundtruth_matrix=tranform_matrix)
# javenlib_tf.quantity_test(imga_kp_cnn,imgb_kp_cnn,groundtruth_matrix=tranform_matrix)
#
# javenlib_tf.show_kp_set(img_path_list[0],imga_kp_cnn)
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_cnn)
# javenlib_tf.show_kp_set(img_path_list[0],imga_kp_sift)
# javenlib_tf.show_kp_set(img_path_list[1],imgb_kp_sift)
#
# # print imga_kp_cnn_des.shape,imgb_kp_cnn_des.shape,imga_kp_sift_des.shape,imgb_kp_sift_des.shape
# # print 'sift shape:',imga_kp_sift.shape,imgb_kp_sift.shape
# # javenlib_tf.match_accuracy(imga_kp_sift,imga_kp_sift_des,imgb_kp_sift,imgb_kp_sift_des,tranform_matrix)
# # print 'cnn shape:',imga_kp_cnn.shape,imgb_kp_cnn.shape
# # javenlib_tf.match_accuracy(imga_kp_cnn,imga_kp_cnn_des,imgb_kp_cnn,imgb_kp_cnn_des,tranform_matrix)

