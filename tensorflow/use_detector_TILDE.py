#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf

################################################################################
################################################################################
################################################################################
#MSE版的use_TILDE
def use_TILDE():
    tf_x = tf.placeholder(tf.float32, [None, 64, 64, 1])  # 输入patch维度为64*64
    tf_y = tf.placeholder(tf.int32, [None, 1])  # input y ,y代表score所以维度为1

    ########################################################################
    conv1 = tf.layers.conv2d(
        inputs=tf_x,  # (?,64,64,1)
        filters=16,
        kernel_size=5,
        strides=1,
        padding='valid',
        activation=tf.nn.relu
    )  # -> (?, 60, 60, 16)

    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2,
    )  # -> (?,30, 30, 16)

    conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'valid', activation=tf.nn.relu)  # -> (?,26, 26, 32)

    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,13, 13, 32)

    conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)  # -> (?,11, 11, 32)

    # pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
    #
    flat = tf.reshape(conv3, [-1, 11 * 11 * 32])  # -> (?,32*32*64)

    output = tf.layers.dense(flat, 1)  # output layer

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,'./save_net/detector_TILDE_model_20171219_mse_20_0_0005')


    img_test_rgb = plt.imread('/home/javen/javenlib/images/bikes/img1.ppm')/255.
    img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
    kp_set = np.zeros(shape=(0,2))
    #对图片进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
    for i in range(32,600-32,16): #扫描的步长需要调整
        for j in range(32,900-32,16):
            patch = np.copy(img_test_gray[i-32:i+32,j-32:j+32]).reshape(1,64,64,1)
            output_predict = sess.run(output, feed_dict={tf_x:patch})
            if output_predict>=0.5:
                kp_set = np.append(kp_set,[[j,i]],axis=0)
    kp_set = kp_set.astype(np.int)
    print kp_set.shape#,kp_set
    kp_set_afternms = javenlib_tf.NMS_4_points_set(kp_set)
    print 'kp_set_afternms:',kp_set_afternms.shape
    kp_set = np.copy(kp_set_afternms)
    #在图上显示检测出的点
    # plt.ion()
    new_img = np.copy(img_test_rgb)
    for i in range(len(kp_set)):
        if kp_set[i,1]-5 < 0 or kp_set[i,1]+5 > new_img.shape[0] or kp_set[i,0]-5 < 0 or kp_set[i,0]+5 > new_img.shape[1]:
            continue
        new_img[kp_set[i,1]-5:kp_set[i,1]+5,kp_set[i,0]-5:kp_set[i,0]+5,0] = 1.
        # plt.figure()
        # plt.imshow(new_img)
        # plt.pause(0.3)
        # plt.close()
    # plt.ioff()

    plt.figure()
    plt.imshow(new_img)
    plt.show()

use_TILDE()


################################################################################
################################################################################
################################################################################
# #softmax版的use_TILDE
# def use_TILDE():
#     tf_x = tf.placeholder(tf.float32, [None, 64, 64, 1])  # 输入patch维度为64*64
#     tf_y = tf.placeholder(tf.int32, [None, 2])  # input y ,y代表score所以维度为1
#
#     ########################################################################
#     conv1 = tf.layers.conv2d(
#         inputs=tf_x,  # (?,64,64,1)
#         filters=16,
#         kernel_size=5,
#         strides=1,
#         padding='valid',
#         activation=tf.nn.relu
#     )  # -> (?, 60, 60, 16)
#
#     pool1 = tf.layers.max_pooling2d(
#         conv1,
#         pool_size=2,
#         strides=2,
#     )  # -> (?,30, 30, 16)
#
#     conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'valid', activation=tf.nn.relu)  # -> (?,26, 26, 32)
#
#     pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,13, 13, 32)
#
#     conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)  # -> (?,11, 11, 32)
#
#     # pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
#     #
#     flat = tf.reshape(conv3, [-1, 11 * 11 * 32])  # -> (?,32*32*64)
#
#     output = tf.layers.dense(flat, 2)  # output layer
#
#     sess = tf.Session()
#     saver = tf.train.Saver()
#     saver.restore(sess,'./save_net/detector_TILDE_model_20171219_softmax_20_0_001')
#
#
#     img_test_rgb = plt.imread('/home/javen/javenlib/images/bikes/img1.ppm')/255.
#     img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
#     kp_set = np.zeros(shape=(0,2))
#     #对图片进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
#     for i in range(32,600-32,32): #扫描的步长需要调整
#         for j in range(32,900-32,32):
#             patch = np.copy(img_test_gray[i-32:i+32,j-32:j+32]).reshape(1,64,64,1)
#             output_predict = sess.run(output, feed_dict={tf_x:patch})
#             # print output_predict
#             if np.argmax(output_predict) == 0:
#                 kp_set = np.append(kp_set,[[j,i]],axis=0)
#     kp_set = kp_set.astype(np.int)
#     print kp_set.shape#,kp_set
#
#     #在图上显示检测出的点
#     # plt.ion()
#     new_img = np.copy(img_test_rgb)
#     for i in range(len(kp_set)):
#         if kp_set[i,1]-5 < 0 or kp_set[i,1]+5 > new_img.shape[0] or kp_set[i,0]-5 < 0 or kp_set[i,0]+5 > new_img.shape[1]:
#             continue
#         new_img[kp_set[i,1]-5:kp_set[i,1]+5,kp_set[i,0]-5:kp_set[i,0]+5,0] = 1.
#         # plt.figure()
#         # plt.imshow(new_img)
#         # plt.pause(0.3)
#         # plt.close()
#     # plt.ioff()
#
#     plt.figure()
#     plt.imshow(new_img)
#     plt.show()
#
# use_TILDE()

