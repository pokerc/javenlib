#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#variable可以有的操作
# tf.add()


# state = tf.Variable([[1,2]],name='counter')
# new_value = tf.Variable([0,0],name='new_value')
# print new_value
# one = tf.constant(1)
#
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# sess.run(tf.assign(new_value[0],state[0,0]))
# print state
# print one
# print sess.run(new_value)



a = tf.Variable([[1,2,3,4,5]])
b = tf.Variable([[6,6,6,6,6]])
c = tf.constant(1)
init  = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print sess.run(tf.multiply(a,b))