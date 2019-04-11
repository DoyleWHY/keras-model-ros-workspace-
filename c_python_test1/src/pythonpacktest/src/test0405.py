#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import the tensorflow library
import tensorflow as tf
import numpy as np
tf.reset_default_graph()

# create the input placeholder
X = tf.placeholder(tf.float32, shape=[None, 784], name="X")

# create network parameters
weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[784, 200], initializer=weight_initer)
bias_initer =tf.constant(0., shape=[200], dtype=tf.float32)
b = tf.get_variable(name="Bias", dtype=tf.float32, initializer=bias_initer)

# create MatMul node
x_w = tf.matmul(X, W, name="MatMul")
# create Add node
x_w_b = tf.add(x_w, b, name="Add")
# create ReLU node
h = tf.nn.relu(x_w_b, name="ReLU")

# ____step 1:____ create the scalar summary
first_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=b)

# create saver object
saver = tf.train.Saver()

# Add an Op to initialize variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # initialize variables
    sess.run(init_op)
    # create the dictionary:
    d = {X: np.random.rand(100, 784)}

    # ____step 2:____ creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # feed it to placeholder a via the dict
    print(sess.run(h, feed_dict=d))

    # ____step 3:____ evaluate the scalar summary
#    summary = sess.run(first_summary)
    # ____step 4:____ add the summary to the writer (i.e. to the event file)
#    writer.add_summary(summary)

    # save the variable in the disk
    saved_path = saver.save(sess, './saved_variable')
    print('model saved in {}'.format(saved_path))
