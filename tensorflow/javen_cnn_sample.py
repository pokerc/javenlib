#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import javenlib_tf

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 10
LR = 0.001

train_x = np.load('./animal/animal_train_data.npy')/255.
train_y = np.load('./animal/animal_train_label.npy')
test_x = np.load('./animal/animal_test_data.npy')/255.
test_y = np.load('./animal/animal_test_label.npy')


# mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)
# test_x = mnist.test.images[:2000]
# test_y = mnist.test.labels[:2000]

tf_x = tf.placeholder(tf.float32, [None, 256,256,1])
# image = tf.reshape(tf_x, [-1, 256, 256, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 2])            # input y

# print image.shape,image
# print tf_x.shape,tf_x

########################################################################
conv1 = tf.layers.conv2d(
    inputs=tf_x,    # (?,256,256,1)
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (?,256, 256, 16)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (?,128, 128, 16)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (?,128, 128, 32)

pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (?,64, 64, 32)

conv3 = tf.layers.conv2d(pool2, 64, 5, 1, 'same', activation=tf.nn.relu)    # -> (?,64, 64, 64)

pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (?,32, 32, 64)

flat = tf.reshape(pool3, [-1, 32*32*64])          # -> (?,32*32*64)

output = tf.layers.dense(flat, 2)              # output layer
########################################################################

#目标函数与梯度下降方法设置
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
# loss = tf.losses.mean_squared_error(tf_y,output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

##########################################################################
# count = 0
# for step in range(50):
#     #cat与dog的图片交叉训练
#     path = './animal/train/cat/cat'+str(step)+'.jpg'
#     image = plt.imread(path)/255.
#     image = javenlib_tf.image_resize(image,256,256).eval(session=sess).reshape((1,256,256,1))
#     label = tf.constant([[1,0]],dtype=tf.int32).eval(session=sess).reshape((1,2))
#     # print image.shape,image.dtype,label.shape,label.dtype
#     sess.run(train_op,{tf_x:image,tf_y:label})
#     result = sess.run(output,{tf_x:image,tf_y:label})[0].argmax()
#     if result == 0:
#         count += 1
#
#     path = './animal/train/dog/dog' + str(step) + '.jpg'
#     image = plt.imread(path) / 255.
#     image = javenlib_tf.image_resize(image, 256, 256).eval(session=sess).reshape((1, 256, 256, 1))
#     label = tf.constant([[0, 1]], dtype=tf.int32).eval(session=sess).reshape((1, 2))
#     # print image.shape,image.dtype,label.shape,label.dtype
#     sess.run(train_op, {tf_x: image, tf_y: label})
#     result = sess.run(output, {tf_x: image, tf_y: label})[0].argmax()
#     if result == 1:
#         count += 1
#     print sess.run(loss,{tf_x:image,tf_y:label})
#
# print '准确预测的数量:',count

print train_x.shape,train_y.shape,test_x.shape,test_y.shape
for step in range(40):
    index_start = (10*(step%10))
    index_end = (10*(step%10)+10)
    print index_start,index_end
    a,b = sess.run([train_op,loss],feed_dict={tf_x:train_x[index_start:index_end],tf_y:train_y[index_start:index_end]})
    print b
    if step == 9:
        output_10 = sess.run(output,feed_dict={tf_x:test_x})
    if step == 39:
        output_20 = sess.run(output,feed_dict={tf_x:test_x})

count = 0
for i in range(20):
    if output_10[i].argmax() == test_y[i].argmax():
        count += 1
print 'output_10:',output_10,count,count/20.

count = 0
for i in range(20):
    if output_20[i].argmax() == test_y[i].argmax():
        count += 1
print 'output_20:',output_20,count,count/20.