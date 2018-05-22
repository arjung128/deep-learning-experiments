import tensorflow as tf

sess = tf.Session()

# ReLU
print("ReLU:", sess.run(tf.nn.relu([-3., 3., 10.])))

# ReLU capped at 6
print("ReLU capped:", sess.run(tf.nn.relu6([-3., 3., 10.])))

# sigmoid
print("sigmoid:", sess.run(tf.nn.sigmoid([-1., 0., 1.])))
print(sess.run(tf.nn.sigmoid([1.])) - sess.run(tf.nn.sigmoid([-1.]))) # not 0.5

# tanh
print("tanh:", sess.run(tf.nn.tanh([-1., 0., 1.])))

# softsign
print("softsign:", sess.run(tf.nn.softsign([-1., 0., 1.])))

# softplus
print("softplus:", sess.run(tf.nn.softplus([-1., 0., 1.])))

# ELU
print("ELU:", sess.run(tf.nn.elu([-1., 0., 1.])))
