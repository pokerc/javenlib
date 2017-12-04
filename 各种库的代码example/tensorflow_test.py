#/usr/bin/python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#################################
# #sample1
# #create data
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1+0.3

# ###create tensorflow structure start###
# weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# biases = tf.Variable(tf.zeros([1]))

# y = weights*x_data + biases

# loss = tf.reduce_mean(tf.square(y-y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()
# ###create tensorflow structure end###

# sess = tf.Session()
# sess.run(init)     #very important

# for step in range(201):
# 	sess.run(train)
# 	if step % 20 == 0:
# 		print(step,sess.run(weights),sess.run(biases))


#######################################
# #sample2 structure
# matrix1 = tf.constant([[3,3]]) #dimension(1,2)
# matrix2 = tf.constant([[2],
# 					   [2]])   #dimension(2,1)
# product = tf.matmul(matrix1,matrix2)	#matrix multiply

# #method 1
# sess = tf.Session()
# result = sess.run(product)
# print result
# sess.close()

# #method 2
# with tf.Session() as sess:
# 	result2 = sess.run(product)
# 	print result2


##################################
# #sample3 Variable of tensorflow
# state = tf.Variable(0,name='counter')
# print state.name
# one = tf.constant(1)

# new_value = tf.add(state,one)
# update = tf.assign(state,new_value)

# #init = tf.initialize_all_variables()   #must have initialization if any variables been defined
# init = tf.global_variables_initializer() #tf.initialize_all_variables() will be deprecated since 2017.03.02, so use this to instead
# with tf.Session() as sess:
# 	sess.run(init)
# 	for n in range(3):
# 		sess.run(update)
# 		print sess.run(state)


# ###################################
# #sample4 placeholder
# #placeholder: if you want to transmit your data into tensorflow type, it is a transitional value to store value for assigning variable temporally
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)

# output = tf.multiply(input1,input2)
# with tf.Session() as sess:
# 	print sess.run(output,feed_dict={input1:[7.],input2:[2.]})
# 	#print sess.run(output)

###################################
#sample5 add_layer
def  add_layer(inputs,in_size,out_size,activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size])+0.1)
	Wx_plus_b = tf.matmul(inputs,Weights)+biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs
						# biases = tf.Variable(tf.zeros([1,5])+0.1)
						# init = tf.global_variables_initializer()
						# sess = tf.Session()
						# sess.run(init)
						# print sess.run(biases)


x_data = np.linspace(-1,1,300,dtype='float32')[:,np.newaxis]
# x_data = np.linspace(-1,1,300).reshape(300,1)
print x_data.dtype
noise = np.random.normal(0,0.05,x_data.shape)
# print noise
y_data = np.square(x_data)-0.5+noise
# print y_data,y_data.shape

# xs = tf.placeholder([None,1])
# ys = tf.placeholder([None,1])

l1 = add_layer(x_data,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
for i in range(1000):
	sess.run(train_step)
	if i%50==0:
		# print sess.run(loss)
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction)
		lines = ax.plot(x_data,prediction_value,'r-',lw=5)
		plt.pause(0.5)







