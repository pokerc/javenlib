import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001

mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

# print(image.shape)
# print(tf_x.shape)

########################################################################
conv1 = tf.layers.conv2d(
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)

pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)

flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )

output = tf.layers.dense(flat, 10)              # output layer
########################################################################

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(600):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b = sess.run(loss, {tf_x: b_x, tf_y: b_y})
    a = sess.run(train_op,{tf_x:b_x,tf_y:b_y})
    c = sess.run(loss, {tf_x: b_x, tf_y: b_y})
    conv11 = sess.run(conv1,{tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        print "step:",step
        print "loss:",a,b,c,conv11.shape
        print conv11[0]
