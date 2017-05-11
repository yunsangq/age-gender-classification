import tensorflow as tf
import os

BN_EPSILON = 0.001


def get_checkpoint(checkpoint_path, requested_step=None):
    if requested_step is not None:
        model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, 'checkpoint', requested_step)
        if os.path.exists(model_checkpoint_path) is None:
            print('No checkpoint file found at [%s]' % checkpoint_path)
            exit(-1)
            print(model_checkpoint_path)
        print(model_checkpoint_path)
        return model_checkpoint_path, requested_step

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
    weight_decay = 0.0001

    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def batch_normalization_layer(input_layer, dimension):
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.random_normal_initializer(stddev=0.01))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.constant_initializer(0.0))

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def conv_bn_relu_layer(input_layer, filter_shape, stride, padding, name):
    with tf.variable_scope(name):
        out_channel = filter_shape[-1]
        w = create_variables(name='weights', shape=filter_shape,
                             initializer=tf.random_normal_initializer(stddev=0.01))
        b = create_variables(name='bias', shape=[out_channel],
                             initializer=tf.constant_initializer(0.0))

        conv_layer = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding=padding)
        conv_layer = tf.add(conv_layer, b)
        bn = batch_normalization_layer(conv_layer, out_channel)
        relu = tf.nn.relu(bn)
    return relu


def bn_relu_conv_layer(input_layer, filter_shape, stride, padding, name):
    with tf.variable_scope(name):
        out_channel = filter_shape[-1]
        w = create_variables(name='weights', shape=filter_shape,
                             initializer=tf.random_normal_initializer(stddev=0.01))
        b = create_variables(name='bias', shape=[out_channel],
                             initializer=tf.constant_initializer(0.0))

        bn = batch_normalization_layer(input_layer, filter_shape[2])
        relu = tf.nn.relu(bn)
        conv_layer = tf.nn.conv2d(relu, w, strides=[1, stride, stride, 1], padding=padding)
        conv_layer = tf.add(conv_layer, b)
    return conv_layer


def conv_bn_layer(input_layer, filter_shape, stride, padding, name):
    with tf.variable_scope(name):
        out_channel = filter_shape[-1]
        w = create_variables(name='weights', shape=filter_shape,
                             initializer=tf.random_normal_initializer(stddev=0.01))
        b = create_variables(name='bias', shape=[out_channel],
                             initializer=tf.constant_initializer(0.0))

        conv_layer = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding=padding)
        conv_layer = tf.add(conv_layer, b)
        bn = batch_normalization_layer(conv_layer, out_channel)
    return bn


def inference(input_tensor_batch, nlabels, keep_prob, reuse):
    layers = []
    with tf.variable_scope('conv1', reuse=reuse):
        input_data = tf.pad(input_tensor_batch, [[0, 0], [3, 3], [3, 3], [0, 0]])
        conv1 = bn_relu_conv_layer(input_data, [7, 7, 3, 64], 2, 'VALID', name='conv')
        pool = tf.nn.max_pool(conv1,
                              ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
        layers.append(pool)

    ''' 34-layer pre-activation
    with tf.variable_scope('conv2_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch1')

        res2a = tf.add(branch2b, branch1)
        layers.append(res2a)

    with tf.variable_scope('conv2_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')

        res2b = tf.add(layers[-1], branch2b)
        layers.append(res2b)

    with tf.variable_scope('conv2_3', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')

        res2c = tf.add(layers[-1], branch2b)
        layers.append(res2c)

    with tf.variable_scope('conv3_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 64, 128], 2, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 64, 128], 2, 'SAME', name='branch1')

        res3a = tf.add(branch2b, branch1)
        layers.append(res3a)

    with tf.variable_scope('conv3_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 128, 128], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        res3b = tf.add(layers[-1], branch2b)
        layers.append(res3b)

    with tf.variable_scope('conv3_3', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 128, 128], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        res3c = tf.add(layers[-1], branch2b)
        layers.append(res3c)

    with tf.variable_scope('conv3_4', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 128, 128], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        res3d = tf.add(layers[-1], branch2b)
        layers.append(res3d)

    with tf.variable_scope('conv4_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 128, 256], 2, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 128, 256], 2, 'SAME', name='branch1')

        res4a = tf.add(branch2b, branch1)
        layers.append(res4a)

    with tf.variable_scope('conv4_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 256], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        res4b = tf.add(layers[-1], branch2b)
        layers.append(res4b)

    with tf.variable_scope('conv4_3', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 256], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        res4c = tf.add(layers[-1], branch2b)
        layers.append(res4c)

    with tf.variable_scope('conv4_4', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 256], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        res4d = tf.add(layers[-1], branch2b)
        layers.append(res4d)

    with tf.variable_scope('conv4_5', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 256], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        res4e = tf.add(layers[-1], branch2b)
        layers.append(res4e)

    with tf.variable_scope('conv4_6', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 256], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        res4f = tf.add(layers[-1], branch2b)
        layers.append(res4f)

    with tf.variable_scope('conv5_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 512], 2, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 256, 512], 2, 'SAME', name='branch1')

        res5a = tf.add(branch2b, branch1)
        layers.append(res5a)

    with tf.variable_scope('conv5_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 512, 512], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')

        res5b = tf.add(layers[-1], branch2b)
        layers.append(res5b)

    with tf.variable_scope('conv5_3', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 512, 512], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')

        res5c = tf.add(layers[-1], branch2b)
        layers.append(res5c)
    '''

    ''' 18-layer pre-activation '''
    with tf.variable_scope('conv2_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch1')

        res2a = tf.add(branch2b, branch1)
        layers.append(res2a)

    with tf.variable_scope('conv2_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')

        res2b = tf.add(layers[-1], branch2b)
        layers.append(res2b)

    with tf.variable_scope('conv3_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 64, 128], 2, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 64, 128], 2, 'SAME', name='branch1')

        res3a = tf.add(branch2b, branch1)
        layers.append(res3a)

    with tf.variable_scope('conv3_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 128, 128], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        res3b = tf.add(layers[-1], branch2b)
        layers.append(res3b)

    with tf.variable_scope('conv4_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 128, 256], 2, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 128, 256], 2, 'SAME', name='branch1')

        res4a = tf.add(branch2b, branch1)
        layers.append(res4a)

    with tf.variable_scope('conv4_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 256], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        res4b = tf.add(layers[-1], branch2b)
        layers.append(res4b)

    with tf.variable_scope('conv5_1', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 256, 512], 2, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')

        branch1 = bn_relu_conv_layer(layers[-1], [3, 3, 256, 512], 2, 'SAME', name='branch1')

        res5a = tf.add(branch2b, branch1)
        layers.append(res5a)

    with tf.variable_scope('conv5_2', reuse=reuse):
        branch2a = bn_relu_conv_layer(layers[-1], [3, 3, 512, 512], 1, 'SAME', name='branch2a')
        branch2b = bn_relu_conv_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')

        res5b = tf.add(layers[-1], branch2b)
        layers.append(res5b)


    ''' 18-layer
    with tf.variable_scope('conv2_1', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')

        branch1 = conv_bn_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch1')

        res2a = tf.add(branch2b, branch1)
        res2a_relu = tf.nn.relu(res2a)
        layers.append(res2a_relu)

    with tf.variable_scope('conv2_2', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 64, 64], 1, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')

        res2b = tf.add(layers[-1], branch2b)
        res2b_relu = tf.nn.relu(res2b)
        layers.append(res2b_relu)

    with tf.variable_scope('conv3_1', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 64, 128], 2, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        branch1 = conv_bn_layer(layers[-1], [3, 3, 64, 128], 1, 'SAME', name='branch1')

        res3a = tf.add(branch2b, branch1)
        res3a_relu = tf.nn.relu(res3a)
        layers.append(res3a_relu)

    with tf.variable_scope('conv3_2', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 128, 128], 1, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')

        res3b = tf.add(layers[-1], branch2b)
        res3b_relu = tf.nn.relu(res3b)
        layers.append(res3b_relu)

    with tf.variable_scope('conv4_1', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 128, 256], 2, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        branch1 = conv_bn_layer(layers[-1], [3, 3, 128, 256], 1, 'SAME', name='branch1')

        res4a = tf.add(branch2b, branch1)
        res4a_relu = tf.nn.relu(res4a)
        layers.append(res4a_relu)

    with tf.variable_scope('conv4_2', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 256, 256], 1, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')

        res4b = tf.add(layers[-1], branch2b)
        res4b_relu = tf.nn.relu(res4b)
        layers.append(res4b_relu)

    with tf.variable_scope('conv5_1', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 256, 512], 2, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')

        branch1 = conv_bn_layer(layers[-1], [3, 3, 256, 512], 1, 'SAME', name='branch1')

        res5a = tf.add(branch2b, branch1)
        res5a_relu = tf.nn.relu(res5a)
        layers.append(res5a_relu)

    with tf.variable_scope('conv5_2', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [3, 3, 512, 512], 1, 'SAME', name='branch2a')
        branch2b = conv_bn_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')

        res5b = tf.add(layers[-1], branch2b)
        res5b_relu = tf.nn.relu(res5b)
        layers.append(res5b_relu)
    '''

    ''' 50-layer
    with tf.variable_scope('res2a', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 64, 64], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 64, 256], 1, 'VALID', name='branch2c')

        branch1 = conv_bn_layer(layers[-1], [1, 1, 64, 256], 1, 'VALID', name='branch1')

        res2a = tf.add(branch2c, branch1)
        res2a_relu = tf.nn.relu(res2a)
        layers.append(res2a_relu)

    with tf.variable_scope('res2b', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 256, 64], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 64, 256], 1, 'VALID', name='branch2c')

        res2b = tf.add(layers[-1], branch2c)
        res2b_relu = tf.nn.relu(res2b)
        layers.append(res2b_relu)

    with tf.variable_scope('res2c', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 256, 64], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 64, 64], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 64, 256], 1, 'VALID', name='branch2c')

        res2c = tf.add(layers[-1], branch2c)
        res2c_relu = tf.nn.relu(res2c)
        layers.append(res2c_relu)

    with tf.variable_scope('res3a', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 256, 128], 2, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 128, 512], 1, 'VALID', name='branch2c')

        branch1 = conv_bn_layer(layers[-1], [1, 1, 256, 512], 2, 'VALID', name='branch1')

        res3a = tf.add(branch2c, branch1)
        res3a_relu = tf.nn.relu(res3a)
        layers.append(res3a_relu)

    with tf.variable_scope('res3b', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 512, 128], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 128, 512], 1, 'VALID', name='branch2c')

        res3b = tf.add(layers[-1], branch2c)
        res3b_relu = tf.nn.relu(res3b)
        layers.append(res3b_relu)

    with tf.variable_scope('res3c', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 512, 128], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 128, 512], 1, 'VALID', name='branch2c')

        res3c = tf.add(layers[-1], branch2c)
        res3c_relu = tf.nn.relu(res3c)
        layers.append(res3c_relu)

    with tf.variable_scope('res3d', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 512, 128], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 128, 128], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 128, 512], 1, 'VALID', name='branch2c')

        res3d = tf.add(layers[-1], branch2c)
        res3d_relu = tf.nn.relu(res3d)
        layers.append(res3d_relu)

    with tf.variable_scope('res4a', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 512, 256], 2, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 256, 1024], 1, 'VALID', name='branch2c')

        branch1 = conv_bn_layer(layers[-1], [1, 1, 512, 1024], 2, 'VALID', name='branch1')

        res4a = tf.add(branch2c, branch1)
        res4a_relu = tf.nn.relu(res4a)
        layers.append(res4a_relu)

    with tf.variable_scope('res4b', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 1024, 256], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 256, 1024], 1, 'VALID', name='branch2c')

        res4b = tf.add(layers[-1], branch2c)
        res4b_relu = tf.nn.relu(res4b)
        layers.append(res4b_relu)

    with tf.variable_scope('res4c', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 1024, 256], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 256, 1024], 1, 'VALID', name='branch2c')

        res4c = tf.add(layers[-1], branch2c)
        res4c_relu = tf.nn.relu(res4c)
        layers.append(res4c_relu)

    with tf.variable_scope('res4d', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 1024, 256], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 256, 1024], 1, 'VALID', name='branch2c')

        res4d = tf.add(layers[-1], branch2c)
        res4d_relu = tf.nn.relu(res4d)
        layers.append(res4d_relu)

    with tf.variable_scope('res4e', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 1024, 256], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 256, 1024], 1, 'VALID', name='branch2c')

        res4e = tf.add(layers[-1], branch2c)
        res4e_relu = tf.nn.relu(res4e)
        layers.append(res4e_relu)

    with tf.variable_scope('res4f', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 1024, 256], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 256, 256], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 256, 1024], 1, 'VALID', name='branch2c')

        res4f = tf.add(layers[-1], branch2c)
        res4f_relu = tf.nn.relu(res4f)
        layers.append(res4f_relu)

    with tf.variable_scope('res5a', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 1024, 512], 2, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 512, 2048], 1, 'VALID', name='branch2c')

        branch1 = conv_bn_layer(layers[-1], [1, 1, 1024, 2048], 2, 'VALID', name='branch1')

        res5a = tf.add(branch2c, branch1)
        res5a_relu = tf.nn.relu(res5a)
        layers.append(res5a_relu)

    with tf.variable_scope('res5b', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 2048, 512], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 512, 2048], 1, 'VALID', name='branch2c')

        res5b = tf.add(layers[-1], branch2c)
        res5b_relu = tf.nn.relu(res5b)
        layers.append(res5b_relu)

    with tf.variable_scope('res5c', reuse=reuse):
        branch2a = conv_bn_relu_layer(layers[-1], [1, 1, 2048, 512], 1, 'VALID', name='branch2a')
        branch2b = conv_bn_relu_layer(branch2a, [3, 3, 512, 512], 1, 'SAME', name='branch2b')
        branch2c = conv_bn_layer(branch2b, [1, 1, 512, 2048], 1, 'VALID', name='branch2c')

        res5c = tf.add(layers[-1], branch2c)
        res5c_relu = tf.nn.relu(res5c)
        layers.append(res5c_relu)
    '''
    with tf.variable_scope('avg_pool', reuse=reuse):
        avg_pool = tf.nn.avg_pool(layers[-1],
                                  ksize=[1, 7, 7, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')
        layers.append(avg_pool)

    with tf.variable_scope('flat', reuse=reuse):
        flat = tf.reshape(layers[-1], [-1, 512], name='reshape')
        layers.append(flat)

    with tf.variable_scope('output', reuse=reuse):
        fc = output_layer(layers[-1], nlabels)
        layers.append(fc)

    return layers[-1]
