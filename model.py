from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_checkpoint(checkpoint_path):
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    weight_decay = 0.0005

    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.random_normal_initializer(stddev=0.01))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.constant_initializer(0.0))

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def fc_layer(input_layer, keep_prob, output_channel):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, output_channel], is_fc_layer=True,
                            initializer=tf.random_normal_initializer(stddev=0.005))
    fc_b = create_variables(name='fc_bias', shape=[output_channel], initializer=tf.constant_initializer(1.0))

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    relu = tf.nn.relu(fc_h)
    drop = tf.nn.dropout(relu, keep_prob)
    return drop


def conv_relu_layer(input_layer, filter_shape, stride, padding):
    out_channel = filter_shape[-1]
    w = create_variables(name='weights', shape=filter_shape,
                         initializer=tf.random_normal_initializer(stddev=0.01))
    b = create_variables(name='bias', shape=[out_channel],
                         initializer=tf.constant_initializer(0.0))

    conv_layer = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding=padding)
    conv_layer = tf.add(conv_layer, b)
    conv_layer = tf.nn.relu(conv_layer)
    pool = tf.nn.max_pool(conv_layer,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='VALID')
    norm = tf.nn.local_response_normalization(pool, 5, alpha=0.0001, beta=0.75)

    return norm


def inference(input_tensor_batch, nlabels, keep_prob, reuse):
    layers = []
    with tf.variable_scope('conv1', reuse=reuse):
        conv1 = conv_relu_layer(input_tensor_batch, [7, 7, 3, 96], 4, 'VALID')
        layers.append(conv1)

    with tf.variable_scope('conv2', reuse=reuse):
        conv2 = conv_relu_layer(layers[-1], [5, 5, 96, 256], 1, 'SAME')
        layers.append(conv2)

    with tf.variable_scope('conv3', reuse=reuse):
        conv3 = conv_relu_layer(layers[-1], [3, 3, 256, 384], 1, 'SAME')
        layers.append(conv3)

    with tf.variable_scope('flat', reuse=reuse):
        flat = tf.reshape(layers[-1], [-1, 384 * 6 * 6], name='reshape')
        layers.append(flat)

    with tf.variable_scope('fc1', reuse=reuse):
        fc = fc_layer(layers[-1], keep_prob, 512)
        layers.append(fc)

    with tf.variable_scope('fc2', reuse=reuse):
        fc = fc_layer(layers[-1], keep_prob, 512)
        layers.append(fc)

    with tf.variable_scope('output', reuse=reuse):
        fc = output_layer(layers[-1], nlabels)
        layers.append(fc)

    return layers[-1]
