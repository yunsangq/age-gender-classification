from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import *


def inference(images, nlabels, pkeep):
    weight_decay = 0.0005
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    # conv1
    conv1 = convolution2d(images, 96, [7, 7], [4, 4], padding='VALID',
                          weights_regularizer=weights_regularizer,
                          biases_initializer=tf.constant_initializer(0.),
                          weights_initializer=tf.random_normal_initializer(stddev=0.01),
                          trainable=True,
                          scope='conv1')
    pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
    norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75, name='norm1')

    # conv2
    conv2 = convolution2d(norm1, 256, [5, 5], [1, 1], padding='SAME',
                          weights_regularizer=weights_regularizer,
                          biases_initializer=tf.constant_initializer(0.),
                          weights_initializer=tf.random_normal_initializer(stddev=0.01),
                          trainable=True,
                          scope='conv2')
    pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
    norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75, name='norm2')

    # conv3
    conv3 = convolution2d(norm2, 384, [3, 3], [1, 1], padding='SAME',
                          weights_regularizer=weights_regularizer,
                          biases_initializer=tf.constant_initializer(0.),
                          weights_initializer=tf.random_normal_initializer(stddev=0.01),
                          trainable=True,
                          scope='conv3')
    pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
    flat = tf.reshape(pool3, [-1, 384 * 6 * 6], name='reshape')

    # fully_connected
    full1 = fully_connected(flat, 512,
                            weights_regularizer=weights_regularizer,
                            biases_initializer=tf.constant_initializer(1.),
                            weights_initializer=tf.random_normal_initializer(stddev=0.005),
                            trainable=True,
                            scope='full1')
    drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
    full2 = fully_connected(drop1, 512,
                            weights_regularizer=weights_regularizer,
                            biases_initializer=tf.constant_initializer(1.),
                            weights_initializer=tf.random_normal_initializer(stddev=0.005),
                            trainable=True,
                            scope='full2')
    drop2 = tf.nn.dropout(full2, pkeep, name='drop2')

    weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
    biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
    output = tf.add(tf.matmul(drop2, weights), biases, name='output')
    return output

