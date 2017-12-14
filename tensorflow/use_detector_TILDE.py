#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf

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
    saver.restore(sess,'./save_net/detector_TILDE_model')

    #预处理要forward计算的数据
    kp_patch_set_positive = np.load('./TILDE_data/positive_samples.npy')
    kp_patch_set_negative = np.load('./TILDE_data/negative_samples.npy')
    train_x_rgb = np.zeros(shape=(1, 64, 64, 3))
    train_x = np.zeros(shape=(1, 64, 64, 1))
    train_y = np.zeros(shape=(1, 1))
    print train_x_rgb.shape
    for i in range(100):
        if i % 2 == 0:
            train_x_rgb = np.append(train_x_rgb, kp_patch_set_positive[i / 2].reshape(1, 64, 64, 3), axis=0)
            train_y = np.append(train_y, [[1]], axis=0)
        else:
            train_x_rgb = np.append(train_x_rgb, kp_patch_set_negative[i / 2].reshape(1, 64, 64, 3), axis=0)
            train_y = np.append(train_y, [[0]], axis=0)
    train_x_rgb = np.delete(train_x_rgb, 0, axis=0)
    train_y = np.delete(train_y, 0, axis=0)
    # print train_y.shape
    # print train_x.shape
    # javenlib_tf.show_patch_set(train_x)

    # rgb转化为gray
    for i in range(100):
        train_x = np.append(train_x, tf.image.rgb_to_grayscale(train_x_rgb[i]).eval(session=sess).reshape(1, 64, 64, 1),
                            axis=0)
    train_x = np.delete(train_x, 0, axis=0)

    # 测试准确率
    count = 0
    for i in range(100):
        output_predict = sess.run(output, feed_dict={tf_x: train_x[i].reshape(1, 64, 64, 1)})
        print 'step:', i, 'output_predict', output_predict
        if (output_predict >= 0.5 and train_y[i] == 1) or (output_predict <= 0.5 and train_y[i] == 0):
            count += 1
    print '准确率:', count / 100.

    img_test_rgb = plt.imread('/home/javen/javenlib/images/bikes/img1.ppm')/255.
    img_test_gray = tf.image.rgb_to_grayscale(img_test_rgb).eval(session=sess)
    kp_set = np.zeros(shape=(1,2))
    #对图片进行扫描,用训练好的TILDE网络来判断某一个点是不是具有可重复性的kp
    for i in range(32,600-32,32): #扫描的步长需要调整
        for j in range(32,900-32,32):
            patch = np.copy(img_test_gray[i-32:i+32,j-32:j+32]).reshape(1,64,64,1)
            output_predict = sess.run(output, feed_dict={tf_x:patch})
            if output_predict>=0.5:
                kp_set = np.append(kp_set,[[j,i]],axis=0)
    kp_set = np.delete(kp_set,0,axis=0).astype(np.int)
    print kp_set.shape,kp_set

    #在图上显示检测出的点
    # plt.ion()
    new_img = np.copy(img_test_rgb)
    for i in range(len(kp_set)):
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