#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf
import cmath
from tensorflow.examples.tutorials.mnist import input_data



# imga = plt.imread('/home/javen/javenlib/images/graf_rotate/img1.ppm')[320-260:320+260,400-260:400+260]
# plt.imshow(imga)
# # plt.show()
# plt.imsave('/home/javen/javenlib/images/graf_rotate/img1_0.jpg',imga,format='jpg')


# #单应性矩阵的计算
# tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/javenlib/images/bikes/H1to3p')
# img_path_list = ['/home/javen/javenlib/images/kitti/0000000000.png',
#                  '/home/javen/javenlib/images/kitti/0000000001.png']
#
# MIN_MATCH_COUNT = 10
# img1 = cv2.imread('/home/javen/javenlib/images/kitti/0000000000.png')          # queryImage
# img2 = cv2.imread('/home/javen/javenlib/images/kitti/0000000001.png')          # trainImage
# img3 = cv2.imread('/home/javen/Downloads/kitti_City/2011_09_26_drive_0001_extract/image_01/data/0000000000.png')
# img4 = plt.imread('/home/javen/javenlib/images/kitti/0000000000.png')

# # Initiate SIFT detector
# sift = cv2.SIFT()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(des1,des2,k=2)
#
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
#
# print len(good)
#
# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# print M
# # print tranform_matrix
# # print M-tranform_matrix
#
# k = javenlib_tf.get_homography_from2picture(img_path_list)
# print k

# print img1.shape,len(img1.shape)
# print img3.shape,len(img3.shape),type(img3)
# print img4.shape,len(img4.shape)
# print img3[:5,:5,0]
# print img3[:5,:5,1]
# print img3[:5,:5,2]

# a = cv2.imread('/home/javen/javenlib/images/kitti_city_gray/0000000000.png')
# a_path = '/home/javen/javenlib/images/kitti_city_gray/0000000000.png'
# b = cv2.imread('/home/javen/javenlib/images/kitti_city_rgb/0000000000.png')
# b_path = '/home/javen/javenlib/images/kitti_city_rgb/0000000000.png'
# sift = cv2.SIFT(250);
# kp1,des1 = sift.detectAndCompute(a,None)
# kp2,des2 = sift.detectAndCompute(b,None)
# print len(kp1),des1.shape,len(kp2),des2.shape
# kp1 = javenlib_tf.KeyPoint_convert_forOpencv2(kp1)
# kp2 = javenlib_tf.KeyPoint_convert_forOpencv2(kp2)
# # print kp1[:10]
# javenlib_tf.show_kp_set(b_path,kp1,10)
# javenlib_tf.show_kp_set(b_path,kp2,10)
# print a.shape

def get_homography_from2picture_sift(img_path_list,numberToDetect = 100):
	MIN_MATCH_COUNT = 10
	img1 = cv2.imread(img_path_list[0])  # queryImage
	img2 = cv2.imread(img_path_list[1])  # trainImage

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=numberToDetect)

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=150)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append(m)

	# print len(good)

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	return M

def get_homography_from2picture_surf(img_path_list):
	MIN_MATCH_COUNT = 10
	img1 = cv2.imread(img_path_list[0])  # queryImage
	img2 = cv2.imread(img_path_list[1])  # trainImage

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SURF_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)
	# print 'surf!!!',des1.shape,des1.dtype
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=150)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append(m)

	# print len(good)

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	return M

def get_homography_from2picture_orb(img_path_list):
	MIN_MATCH_COUNT = 10
	img1 = cv2.imread(img_path_list[0])  # queryImage
	img2 = cv2.imread(img_path_list[1])  # trainImage

	# Initiate SIFT detector
	sift = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)
	des1 = des1.astype(np.float32)
	des2 = des2.astype(np.float32)
	# print 'orb!!!',des1.shape,des2.shape,type(des1),des1.dtype
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=150)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append(m)

	# print len(good)

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	return M

# sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
# surf = cv2.xfeatures2d.SURF_create()
# orb = cv2.ORB_create(nfeatures=250)
# kaze = cv2.KAZE_create()
#
#
# img = cv2.imread('/home/javen/javenlib/images/kitti_city_gray_0005/0000000002.png')
# surf2 = cv2.xfeatures2d.SURF_create(hessianThreshold=2000,extended=True)
# kp_surf2,des_surf2 = surf2.detectAndCompute(img,None)
# print 'surf2 length:',len(kp_surf2),des_surf2.shape

# kaze = cv2.KAZE_create(extended=True,threshold=0.026)
# kp_kaze,des_kaze = kaze.detectAndCompute(img,None)
# print 'kaze length:',len(kp_kaze),des_kaze.shape
# # plt.imshow(img)
# # plt.show()
# kp_sift,des_sift = sift.detectAndCompute(img,None)
# kp_sift = javenlib_tf.KeyPoint_convert_forOpencv2(kp_sift)
# print len(kp_sift),des_sift.shape
# kp_surf_obj,des_surf = surf.detectAndCompute(img,None)
# kp_surf = javenlib_tf.KeyPoint_convert_forOpencv2(kp_surf_obj)
# print kp_surf.shape,des_surf.shape
# # surf.extended=True
# des_surf,des_surf = surf.compute(img,kp_surf_obj[0:500])
# print des_surf.shape
# # print len(kp_surf),des_surf.shape
# # kp_orb,des_orb = orb.detectAndCompute(img,None)
# # kp_orb = javenlib_tf.KeyPoint_convert_forOpencv2(kp_orb)
# # print len(kp_orb),des_orb.shape
# #
# # print kp_orb.shape
# # javenlib_tf.show_kp_set('/home/javen/javenlib/images/kitti_residential_gray_0061/0000000201.png',kp_sift)
# # javenlib_tf.show_kp_set('/home/javen/javenlib/images/kitti_residential_gray_0061/0000000201.png',kp_surf)
# # javenlib_tf.show_kp_set('/home/javen/javenlib/images/kitti_residential_gray_0061/0000000201.png',kp_orb)
#
img_path_list = ['/home/javen/javenlib/images/wall/img1.ppm',
				 '/home/javen/javenlib/images/wall/img6.ppm']
tranform_matrix = javenlib_tf.get_matrix_from_file('/home/javen/javenlib/images/wall/H1to6p')
# m_sift_50 = get_homography_from2picture_sift(img_path_list,numberToDetect=50)
# print m_sift_50
# m_sift_100 = get_homography_from2picture_sift(img_path_list,numberToDetect=100)
# print m_sift_100
# m_sift_250 = get_homography_from2picture_sift(img_path_list,numberToDetect=250)
# print m_sift_250
# m_sift_500 = get_homography_from2picture_sift(img_path_list,numberToDetect=500)
# print m_sift_500
m_sift_1000 = get_homography_from2picture_sift(img_path_list,numberToDetect=1000)
print m_sift_1000
# m_surf = get_homography_from2picture_surf(img_path_list)
# print m_surf
# m_orb  = get_homography_from2picture_orb(img_path_list)
# print m_orb
#
print '差：','\n'
# print m_sift_50-tranform_matrix
# print m_sift_100-tranform_matrix
# print m_sift_250-tranform_matrix
# print m_sift_500-tranform_matrix
print m_sift_1000-tranform_matrix

# diff_50 = m_sift_50-tranform_matrix
# diff_100 = m_sift_100-tranform_matrix
# diff_250 = m_sift_250-tranform_matrix
# diff_500 = m_sift_500-tranform_matrix
diff_1000 = m_sift_1000-tranform_matrix

# print m_surf-tranform_matrix
# print m_orb-tranform_matrix

a = np.array([[-2,0,0],[0,-3,0],[0,0,-4]])
print a**2,sum(sum(a**2))
print 'sum(sum(diff_50**2))','sum(sum(diff_100**2))','sum(sum(diff_250**2))','sum(sum(diff_500**2))',sum(sum(diff_1000**2))