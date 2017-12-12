#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 10
LR = 0.001

train_x = np.load('./animal/animal_train_data.npy')/255.
train_y = np.load('./animal/animal_train_label.npy')
test_x = np.load('./animal/animal_test_data.npy')/255.
test_y = np.load('./animal/animal_test_label.npy')

tf_x = tf.placeholder(tf.float32, [None, 64,64,1]) #输入patch维度为64*64
tf_y = tf.placeholder(tf.int32, [None, 1])            # input y ,y代表score所以维度为1

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

output = tf.layers.dense(flat, 1)              # output layer
########################################################################

#目标函数与梯度下降方法设置
# loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
loss = tf.losses.mean_squared_error(tf_y,output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


print conv1
print pool1
print conv2
print pool2
print conv3
print flat
print output

saver = tf.train.Saver()
saver.save(sess,'./save_net/detector_TILDE_model')

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



