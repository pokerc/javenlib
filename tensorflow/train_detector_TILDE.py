#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import javenlib_tf

tf.set_random_seed(1)
np.random.seed(1)


LR = 0.001
sess = tf.Session()

################################################################################
################################################################################
################################################################################
# #MSE版本
# #读取准备好的未打乱的train数据
# train_data = np.load('/home/javen/javenlib/tensorflow/TILDE_data/train_data_20171219.npy')
# train_label = np.load('/home/javen/javenlib/tensorflow/TILDE_data/train_label_20171219.npy')
# #对数据进行打乱操作
# train_data,train_label = javenlib_tf.shuffle_data_and_label(train_data,train_label) #(?,64,64,3)
# #将rgb数据转化为gray
# train_data = javenlib_tf.rgb2gray_train_data(train_data) #(?,64,64,1)
# #将数据类型转换为float32和int32
# # print train_data.dtype
# # print train_label.dtype
#
# train_x = np.copy(train_data)
# train_y = np.copy(train_label)
#
# tf_x = tf.placeholder(tf.float32, [None, 64,64,1]) #输入patch维度为64*64
# tf_y = tf.placeholder(tf.int32, [None, 1])            # input y ,y代表score所以维度为1
#
# ########################################################################
# conv1 = tf.layers.conv2d(
#     inputs=tf_x,    # (?,64,64,1)
#     filters=16,
#     kernel_size=5,
#     strides=1,
#     padding='valid',
#     activation=tf.nn.relu
# )           # -> (?, 60, 60, 16)
#
# pool1 = tf.layers.max_pooling2d(
#     conv1,
#     pool_size=2,
#     strides=2,
# )           # -> (?,30, 30, 16)
#
# conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'valid', activation=tf.nn.relu)    # -> (?,26, 26, 32)
#
# pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (?,13, 13, 32)
#
# conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)    # -> (?,11, 11, 32)
#
# # pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
# #
# flat = tf.reshape(conv3, [-1, 11*11*32])          # -> (?,11*11*32)
#
# output = tf.layers.dense(flat, 1)              # output layer
# ########################################################################
#
# #目标函数与梯度下降方法设置
# # loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
# loss = tf.losses.mean_squared_error(tf_y,output)
# train_op = tf.train.AdamOptimizer(LR).minimize(loss)
#
# sess.run(tf.global_variables_initializer())
#
#
# print conv1
# print pool1
# print conv2
# print pool2
# print conv3
# print flat
# print output
#
# #train开始
# print 'train_x:',train_x.shape
# total_num = train_x.shape[0] #训练数据的总条数
# batch_size = 50 #每次输入训练的数据条数
# steps_in_onecircle = total_num/batch_size #一次训练集全迭代需要的循环次数
# for step in range(20*steps_in_onecircle):
#     data_batch = np.copy(train_x[batch_size * (step % steps_in_onecircle):batch_size * (step % steps_in_onecircle) + batch_size])
#     label_batch = np.copy(train_y[batch_size*(step%steps_in_onecircle):batch_size*(step%steps_in_onecircle)+batch_size])
#     loss_before = sess.run(loss, feed_dict={tf_x:data_batch, tf_y:label_batch})
#     print 'step:',step,'loss_before:',loss_before
#     sess.run(train_op, feed_dict={tf_x:data_batch, tf_y:label_batch})
#     loss_after = sess.run(loss, feed_dict={tf_x:data_batch, tf_y:label_batch})
#     print 'step:',step,'loss_after:',loss_after
#
# #测试准确率
# count = 0
# for i in range(total_num):
#     output_predict = sess.run(output,feed_dict={tf_x:train_x[i].reshape(1,64,64,1)})
#     print 'step:',i,'output_predict',output_predict.shape
#     if (output_predict>=0.5 and train_y[i]==1) or (output_predict<=0.5 and train_y[i]==0):
#         count += 1
# print '准确率:',1.*count/total_num
#
# #将训练好的模型保存
# saver = tf.train.Saver()
# saver.save(sess,'./save_net/detector_TILDE_model_20171219_mse_20_0_0005')


################################################################################
################################################################################
################################################################################
#将loss function改为softmax的版本,因为需要改写label的数据维度,所以将之前的复制,在下面做修改
#读取准备好的未打乱的train数据
train_data = np.load('/home/javen/javenlib/tensorflow/TILDE_data/train_data_20171219.npy')
train_label = np.load('/home/javen/javenlib/tensorflow/TILDE_data/train_label_20171219.npy')
#将label做softmax所需要的形式
train_label_4softmax = np.zeros(shape=(0,2))
for i in range(len(train_label)):
    if train_label[i] == 1:
        train_label_4softmax = np.append(train_label_4softmax,[[1,0]],axis=0)
    else:
        train_label_4softmax = np.append(train_label_4softmax, [[0, 1]], axis=0)
train_label = np.copy(train_label_4softmax)
#对数据进行打乱操作
train_data,train_label = javenlib_tf.shuffle_data_and_label(train_data,train_label) #(?,64,64,3)
#将rgb数据转化为gray
train_data = javenlib_tf.rgb2gray_train_data(train_data) #(?,64,64,1)
#将数据类型转换为float32和int32
# print train_data.dtype
# print train_label.dtype

train_x = np.copy(train_data)
train_y = np.copy(train_label)

tf_x = tf.placeholder(tf.float32, [None, 64,64,1]) #输入patch维度为64*64
tf_y = tf.placeholder(tf.int32, [None, 2])            # input y ,y代表score所以维度为1

########################################################################
conv1 = tf.layers.conv2d(
    inputs=tf_x,    # (?,64,64,1)
    filters=16,
    kernel_size=5,
    strides=1,
    padding='valid',
    activation=tf.nn.relu
)           # -> (?, 60, 60, 16)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (?,30, 30, 16)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'valid', activation=tf.nn.relu)    # -> (?,26, 26, 32)

pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (?,13, 13, 32)

conv3 = tf.layers.conv2d(pool2, 32, 3, 1, 'valid', activation=tf.nn.relu)    # -> (?,11, 11, 32)

# pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)
#
flat = tf.reshape(conv3, [-1, 11*11*32])          # -> (?,11*11*32)

output = tf.layers.dense(flat, 2)              # output layer
########################################################################

#目标函数与梯度下降方法设置
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
# loss = tf.losses.mean_squared_error(tf_y,output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess.run(tf.global_variables_initializer())

#train开始
print 'train_x:',train_x.shape
total_num = train_x.shape[0] #训练数据的总条数
batch_size = 50 #每次输入训练的数据条数
steps_in_onecircle = total_num/batch_size #一次训练集全迭代需要的循环次数
for step in range(20*steps_in_onecircle):
    data_batch = np.copy(train_x[batch_size * (step % steps_in_onecircle):batch_size * (step % steps_in_onecircle) + batch_size])
    label_batch = np.copy(train_y[batch_size*(step%steps_in_onecircle):batch_size*(step%steps_in_onecircle)+batch_size])
    loss_before = sess.run(loss, feed_dict={tf_x:data_batch, tf_y:label_batch})
    print 'step:',step,'loss_before:',loss_before
    sess.run(train_op, feed_dict={tf_x:data_batch, tf_y:label_batch})
    loss_after = sess.run(loss, feed_dict={tf_x:data_batch, tf_y:label_batch})
    print 'step:',step,'loss_after:',loss_after

#测试准确率
count = 0
for i in range(total_num):
    output_predict = sess.run(output,feed_dict={tf_x:train_x[i].reshape(1,64,64,1)})
    # print 'step:',i,'output_predict',output_predict,'argmax:',np.argmax(output_predict)
    if np.argmax(output_predict) == np.argmax(train_y[i]):
        count += 1
print '准确率:',1.*count/total_num

# #将训练好的模型保存
# saver = tf.train.Saver()
# saver.save(sess,'./save_net/detector_TILDE_model_20171219_softmax_20_0_001')

#释放gpu资源
sess.close()
