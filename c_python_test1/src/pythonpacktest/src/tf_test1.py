#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

import numpy as np
a=np.ones((3,4))
print(a)

nums = [5,6,7]
print(nums)
