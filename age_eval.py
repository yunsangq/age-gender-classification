from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from data import inputs
import numpy as np
import tensorflow as tf
from model import inference, get_checkpoint
import os
import json

tf.app.flags.DEFINE_string('train_dir', './Folds/tf/age_test_fold_is_0',
                           'Training directory (where training data lives)')

tf.app.flags.DEFINE_integer('run_id', 10124,
                            'This is the run number (pid) for training proc')

tf.app.flags.DEFINE_string('device_id', '/gpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('eval_dir', './Folds/tf/age_eval_test_fold_is_0',
                           'Directory to put output to')

tf.app.flags.DEFINE_string('eval_data', 'test',
                           'Data type (valid|test)')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            'Number of preprocessing threads')

tf.app.flags.DEFINE_integer('image_size', 227,
                            'Image size')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch size')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                           'Checkpoint basename')

FLAGS = tf.app.flags.FLAGS

LAMBDA = 0.01


def eval_loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses)
    return total_loss


def eval_once(sess, saver, summary_writer, summary_op, logits, labels, num_eval):
    _loss = eval_loss(logits, labels)
    top1 = tf.nn.in_top_k(logits, labels, 1)
    top2 = tf.nn.in_top_k(logits, labels, 2)

    checkpoint_path = '%s/run-%d/train' % (FLAGS.train_dir, FLAGS.run_id)

    model_checkpoint_path, global_step = get_checkpoint(checkpoint_path)
    global_step = int(global_step)
    saver.restore(sess, model_checkpoint_path)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = []
    try:
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))
        num_steps = int(math.ceil(num_eval / FLAGS.batch_size))
        true_count1 = true_count2 = 0
        total_loss = 0.0
        total_sample_count = num_steps * FLAGS.batch_size
        step = 0
        print(FLAGS.batch_size, num_steps)

        while step < num_steps and not coord.should_stop():
            v, predictions1, predictions2, loss_value = sess.run([logits, top1, top2, _loss])
            true_count1 += np.sum(predictions1)
            true_count2 += np.sum(predictions2)
            total_loss += loss_value

            step += 1

        # Compute precision @ 1.

        precision1 = true_count1 / total_sample_count
        precision2 = true_count2 / total_sample_count
        total_loss /= num_steps
        print('step%d: loss = %.3f' % (global_step, total_loss))
        print('step%d: precision @ 1 = %.3f (%d/%d)' % (global_step, precision1, true_count1, total_sample_count))
        print('step%d: precision @ 2 = %.3f (%d/%d)' % (global_step, precision2, true_count2, total_sample_count))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision1)
        summary.value.add(tag='Precision @ 2', simple_value=precision2)
        summary.value.add(tag='cost', simple_value=total_loss)
        summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(run_dir):
    with tf.Session() as sess:
        input_file = os.path.join(FLAGS.train_dir, 'md.json')
        print(input_file)
        with open(input_file, 'r') as f:
            md = json.load(f)

        num_eval = md['%s_counts' % FLAGS.eval_data]

        images, labels, _ = inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size, mode='test',
                                   num_preprocess_threads=FLAGS.num_preprocess_threads)
        logits = inference(images, md['nlabels'], 1, reuse=False)
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(run_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess.run(init)

        eval_once(sess, saver, summary_writer, summary_op, logits, labels, num_eval)


def main(argv=None):  # pylint: disable=unused-argument
    run_dir = '%s/run-%d/eval' % (FLAGS.train_dir, FLAGS.run_id)
    if tf.gfile.Exists(run_dir):
        tf.gfile.DeleteRecursively(run_dir)
    tf.gfile.MakeDirs(run_dir)
    evaluate(run_dir)

if __name__ == '__main__':
    tf.app.run()

