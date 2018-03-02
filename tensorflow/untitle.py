#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
# import javenlib_tf
import cmath
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
train_images = mnist.train.images.reshape(55000, 28, 28, 1)
train_labels = mnist.train.labels
test_images = mnist.test.images.reshape(10000, 28, 28, 1)
test_labels = mnist.test.labels
print '训练数据和测试数据的shape：', train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
# plt.imshow(train_images[0].reshape(28,28),cmap='gray')
# plt.show()

# 构建网络结构
# 读取准备好的数据,mnist数据集本身就经过了预处理所以不再需要做与处理了
train_x = np.copy(train_images)
train_y = np.copy(train_labels)

g1 = tf.Graph()
g2 = tf.Graph()
with g1.as_default():
    tf_x = tf.placeholder(tf.float32, [None, 28, 28, 1])  # 输入patch维度为28*28
    tf_y = tf.placeholder(tf.int32, [None, 10])  # input y ,y代表预测标签所以维度为10

    ########################################################################
    conv1 = tf.layers.conv2d(
        inputs=tf_x,  # (?,28,28,1)
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',  # same为保持原size,valid为去除边界size会变小
        activation=tf.nn.relu
    )  # -> (?, 28, 28, 32)

    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2,
    )  # -> (?,14, 14, 32)

    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)  # -> (?,14, 14, 64)

    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)  # -> (?,7, 7, 64)

    flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # -> (?,7*7*64)

    full_connect1 = tf.layers.dense(flat, 512)  # -> (?,512)

    output = tf.layers.dense(full_connect1, 10)  # output layer (?,10)

# 输出构建的网络结构的各层的维度情况
print conv1
print pool1
print conv2
print pool2
print flat
print full_connect1
print output

#将训练好的模型加载进来
sess = tf.Session(graph=g1)
with sess:
    saver = tf.train.Saver()
    saver.restore(sess,'./save_net_descriptor_generator/descriptor_generator_model_20180302_softmax_10circle_LR0_0005')

    # 测试训练集准确率
    count = 0
    total_num = 55000
    for i in range(total_num):
        output_predict = sess.run(output, feed_dict={tf_x: train_x[i].reshape(1, 28, 28, 1)})
        # print 'step:',i,'output_predict',output_predict.shape,output_predict
        if np.argmax(output_predict) == np.argmax(train_y[i]):
            count += 1
    print '训练集准确率:', 1. * count / total_num

    # 测试测试集准确率
    count = 0
    total_num = 10000
    for i in range(total_num):
        output_predict = sess.run(output, feed_dict={tf_x: test_images[i].reshape(1, 28, 28, 1)})
        # print 'step:',i,'output_predict',output_predict.shape,output_predict
        if np.argmax(output_predict) == np.argmax(test_labels[i]):
            count += 1
    print '测试集准确率:', 1. * count / total_num