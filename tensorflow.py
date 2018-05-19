import tensorflow as tf
import numpy as np
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b

# sess = tf.Session()

# print(sess.run(adder_node, {a: 3, b:4.5}))
# print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

# ---

# Model y = 1 - x

# Parameters
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# input and output
x = tf.placeholder(tf.float32)
# print(W.dtype)
# print(x.dtype)
# print(b.dtype)
y = W * x + b
yhat = tf.placeholder(tf.float32)

# loss function
loss = tf.reduce_sum(tf.square(y - yhat))

# optimizer
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# training data -- could try changing amount of data and see how many epochs it takes
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init) # reset values to wrong

for i in range(1000):
  sess.run(train, {x:x_train, yhat:y_train})

curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, yhat:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
