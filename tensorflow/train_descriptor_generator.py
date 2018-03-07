#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf
import cmath
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
np.random.seed(1)


LR = 0.0005
sess = tf.Session()

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
train_images = mnist.train.images.reshape(55000,28,28,1)
train_labels = mnist.train.labels
test_images = mnist.test.images.reshape(10000,28,28,1)
test_labels = mnist.test.labels
print 'mean value:',train_images.mean()
meanvalue = train_images.mean()
train_images = train_images - meanvalue
print '训练数据和测试数据的shape：',train_images.shape,train_labels.shape,test_images.shape,test_labels.shape
# plt.figure(1)
# plt.imshow(train_images[0].reshape(28,28),cmap='gray')
# plt.figure(2)
# plt.imshow((train_images[0]-meanvalue).reshape(28,28),cmap='gray')
# plt.show()

#构建网络结构
#读取准备好的数据,mnist数据集本身就经过了预处理所以不再需要做与处理了
train_x = np.copy(train_images)
train_y = np.copy(train_labels)

tf_x = tf.placeholder(tf.float32, [None, 28,28,1]) #输入patch维度为28*28
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y ,y代表预测标签所以维度为10

########################################################################
conv1 = tf.layers.conv2d(
    inputs=tf_x,    # (?,28,28,1)
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same', #same为保持原size,valid为去除边界size会变小
    activation=tf.nn.relu
)           # -> (?, 28, 28, 32)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (?,14, 14, 32)

conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)    # -> (?,14, 14, 64)

pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (?,7, 7, 64)


flat = tf.reshape(pool2, [-1, 7*7*64])          # -> (?,7*7*64)

full_connect1 = tf.layers.dense(flat,512)       # -> (?,512)

output = tf.layers.dense(full_connect1, 10)              # output layer (?,10)

#输出构建的网络结构的各层的维度情况
print conv1
print pool1
print conv2
print pool2
print flat
print full_connect1
print output

#目标函数与梯度下降方法设置
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
# loss = tf.losses.mean_squared_error(tf_y,output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess.run(tf.global_variables_initializer())

###########################################################
#train开始
print 'train_x:',train_x.shape
total_num = train_x.shape[0] #训练数据的总条数
batch_size = 50 #每次输入训练的数据条数
steps_in_onecircle = total_num/batch_size #一次训练集全迭代需要的循环次数
for step in range(10*steps_in_onecircle):
    data_batch = np.copy(train_x[batch_size * (step % steps_in_onecircle):batch_size * (step % steps_in_onecircle) + batch_size])
    label_batch = np.copy(train_y[batch_size*(step%steps_in_onecircle):batch_size*(step%steps_in_onecircle)+batch_size])
    loss_before = sess.run(loss, feed_dict={tf_x:data_batch, tf_y:label_batch})
    # print 'step:',step,'loss_before:',loss_before
    sess.run(train_op, feed_dict={tf_x:data_batch, tf_y:label_batch})
    loss_after = sess.run(loss, feed_dict={tf_x:data_batch, tf_y:label_batch})
    print 'step:',step,'loss_after:',loss_after
    # output_predict = sess.run(output, feed_dict={tf_x: train_x[0].reshape(1, 28, 28, 1)})
    # print 'index 0 output_predict:', output_predict.shape, output_predict,type(output_predict),train_y[0],np.argmax(output_predict),np.argmax(train_y[0])

#测试训练集准确率
count = 0
for i in range(total_num):
    output_predict = sess.run(output,feed_dict={tf_x:train_x[i].reshape(1,28,28,1)})
    # print 'step:',i,'output_predict',output_predict.shape,output_predict
    if np.argmax(output_predict) == np.argmax(train_y[i]):
        count += 1
print '训练集准确率:',1.*count/total_num

#测试测试集准确率
count = 0
total_num = 10000
for i in range(total_num):
    output_predict = sess.run(output,feed_dict={tf_x:test_images[i].reshape(1,28,28,1)})
    # print 'step:',i,'output_predict',output_predict.shape,output_predict
    if np.argmax(output_predict) == np.argmax(test_labels[i]):
        count += 1
print '测试集准确率:',1.*count/total_num


#将训练好的模型保存
saver = tf.train.Saver()
saver.save(sess,'./save_net_descriptor_generator/descriptor_generator_model_20180306_softmax_10circle_LR0_0005')