from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf
from data import distorted_inputs
from data import inputs
from model import select_model
import json

LAMBDA = 0.01
MOM = 0.9
tf.app.flags.DEFINE_string('pre_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

tf.app.flags.DEFINE_string('eval_data', 'valid',
                           'Data type (valid|train)')

tf.app.flags.DEFINE_string('train_dir', './Folds/tf/age_test_fold_is_0',
                           'Training directory')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            'Number of preprocessing threads')

tf.app.flags.DEFINE_string('optim', 'Momentum',
                           'Optimizer')

tf.app.flags.DEFINE_integer('image_size', 227,
                            'Image size')

tf.app.flags.DEFINE_float('eta', 0.002,
                          'Learning rate')

tf.app.flags.DEFINE_float('pdrop', 0.5,
                          'Dropout probability')

tf.app.flags.DEFINE_integer('max_steps', 10000,
                            'Number of iterations')

tf.app.flags.DEFINE_integer('epochs', -1,
                            'Number of epochs')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch size')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                           'Checkpoint name')

tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('pre_model',
                           '',  # './inception_v3.ckpt',
                           'checkpoint file')
FLAGS = tf.app.flags.FLAGS


# Every 5k steps cut learning rate in half
def exponential_staircase_decay(at_step=5000, decay_rate=0.5):
    def _decay(lr, global_step):
        return tf.train.exponential_decay(lr, global_step,
                                          at_step, decay_rate, staircase=True)

    return _decay


def optimizer(optim, eta, loss_fn):
    global_step = tf.Variable(0, trainable=False)
    optz = optim
    if optim == 'Adadelta':
        optz = lambda lr: tf.train.AdadeltaOptimizer(lr, 0.95, 1e-6)
        lr_decay_fn = None
    elif optim == 'Momentum':
        optz = lambda lr: tf.train.MomentumOptimizer(lr, MOM)
        lr_decay_fn = exponential_staircase_decay()

    return tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz, clip_gradients=4.,
                                           learning_rate_decay_fn=lr_decay_fn)


def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses)
    # total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def valid_loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='valid_cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='valid_cross_entropy')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses)
    return total_loss


def eval_once(sess, summary_op, summary_writer, logits, labels, num_eval, global_step):
    total_loss = valid_loss(logits, labels)
    top1 = tf.nn.in_top_k(logits, labels, 1)
    top2 = tf.nn.in_top_k(logits, labels, 2)

    print('Validation')
    num_steps = int(math.ceil(num_eval / FLAGS.batch_size))
    true_count1 = true_count2 = 0
    loss_count = 0.0
    total_sample_count = num_steps * FLAGS.batch_size
    for step in xrange(num_steps):
        v, predictions1, predictions2, loss_value = sess.run([logits, top1, top2, total_loss])
        true_count1 += np.sum(predictions1)
        true_count2 += np.sum(predictions2)
        loss_count += loss_value

    precision1 = true_count1 / total_sample_count
    precision2 = true_count2 / total_sample_count
    _loss = loss_count / float(num_steps)

    format_str = ('%s: step %d, loss = %.3f')
    print(format_str % (datetime.now(), global_step, _loss))
    print('step %d: precision @ 1 = %.3f (%d/%d)' % (global_step, precision1, true_count1, total_sample_count))
    print('step %d: precision @ 2 = %.3f (%d/%d)' % (global_step, precision2, true_count2, total_sample_count))

    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='cost', simple_value=_loss)
    summary.value.add(tag='Precision @ 1', simple_value=precision1)
    summary.value.add(tag='Precision @ 2', simple_value=precision2)
    summary_writer.add_summary(summary, global_step)


def main(argv=None):
    with tf.Graph().as_default():

        model_fn = select_model(FLAGS.model_type)
        # Open the metadata file and figure out nlabels, and size of epoch
        input_file = os.path.join(FLAGS.train_dir, 'md.json')
        print(input_file)
        with open(input_file, 'r') as f:
            md = json.load(f)

        # Validation
        eval_data = FLAGS.eval_data == 'valid'
        num_eval = md['%s_counts' % FLAGS.eval_data]
        valid_images, valid_labels, _ = inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size, train=not eval_data,
                                               num_preprocess_threads=FLAGS.num_preprocess_threads)
        valid_logits = model_fn(md['nlabels'], valid_images, 1, False)

        # Training
        images, labels, _ = distorted_inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size,
                                             FLAGS.num_preprocess_threads)
        logits = model_fn(md['nlabels'], images, 1 - FLAGS.pdrop, True)
        total_loss = loss(logits, labels)

        train_op = optimizer(FLAGS.optim, FLAGS.eta, total_loss)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        tf.global_variables_initializer().run(session=sess)

        # This is total hackland, it only works to fine-tune iv3
        if FLAGS.pre_model:
            inception_variables = tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="InceptionV3")
            restorer = tf.train.Saver(inception_variables)
            restorer.restore(sess, FLAGS.pre_model)

        if FLAGS.pre_checkpoint_path:
            if tf.gfile.Exists(FLAGS.pre_checkpoint_path) is True:
                print('Trying to restore checkpoint from %s' % FLAGS.pre_checkpoint_path)
                restorer = tf.train.Saver()
                tf.train.latest_checkpoint(FLAGS.pre_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %
                      (datetime.now(), FLAGS.pre_checkpoint_path))

        run_dir = '%s/run-%d/train' % (FLAGS.train_dir, os.getpid())
        valid_run_dir = '%s/run-%d/valid' % (FLAGS.train_dir, os.getpid())

        checkpoint_path = '%s/%s' % (run_dir, FLAGS.checkpoint)
        if tf.gfile.Exists(run_dir) is False:
            print('Creating %s' % run_dir)
            tf.gfile.MakeDirs(run_dir)

        tf.train.write_graph(sess.graph_def, run_dir, 'model.pb', as_text=True)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(run_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_run_dir, sess.graph)

        steps_per_train_epoch = int(md['train_counts'] / FLAGS.batch_size)
        num_steps = FLAGS.max_steps if FLAGS.epochs < 1 else FLAGS.epochs * steps_per_train_epoch
        print('Requested number of steps [%d]' % num_steps)

        for step in xrange(num_steps):
            if step == 0:
                eval_once(sess, summary_op, valid_writer, valid_logits, valid_labels, num_eval, step)

            start_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # Loss only actually evaluated every 100 steps?
            if step % 100 == 0:
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='cost', simple_value=loss_value)
                summary_writer.add_summary(summary, step)
                if step != 0:
                    eval_once(sess, summary_op, valid_writer, valid_logits, valid_labels, num_eval, step)

            if step % 1000 == 0 or (step + 1) == num_steps:
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()
