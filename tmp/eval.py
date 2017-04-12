"""
At each tick, evaluate the latest checkpoint against some validation data.
Or, you can run once by passing --run_once.  OR, you can pass a --requested_step_seq of comma separated checkpoint #s that already exist that it can run in a row.
This program expects a training base directory with the data, and md.json file
There will be sub-directories for each run underneath with the name run-<PID>
where <PID> is the training program's process ID.  To run this program, you
will need to pass --train_dir <DIR> which is the base path name, --run_id <PID>
and if you are using a custom name for your checkpoint, you should
pass that as well (most times you probably wont).  This will yield a model path:
<DIR>/run-<PID>/checkpoint
Note: If you are training to use the same GPU you can supposedly
suspend the process.  I have not found this works reliably on my Linux machine.
Instead, I have found that, often times, the GPU will not reclaim the resources
and in that case, your eval may run out of GPU memory.
You can alternately run trainining for a number of steps, break the program
and run this, then restarting training from the old checkpoint.  I also
found this inconvenient.  In order to control this better, the program
requires that you explict placement of inference.  It defaults to the CPU
so that it can easily run side by side with training.  This does make it
much slower than if it was on the GPU, but for evaluation this may not be
a major problem.  To place on the gpu, just pass --device_id /gpu:<ID> where
<ID> is the GPU ID
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
import os
import json

tf.app.flags.DEFINE_string('train_dir', './Folds/tf/age_test_fold_is_0',
                           'Training directory (where training data lives)')

tf.app.flags.DEFINE_integer('run_id', 13537,
                            'This is the run number (pid) for training proc')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('eval_dir', './Folds/tf/eval_test_fold_is_0',
                           'Directory to put output to')

tf.app.flags.DEFINE_string('eval_data', 'valid',
                           'Data type (valid|train)')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            'Number of preprocessing threads')

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 2,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_integer('image_size', 227,
                            'Image size')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch size')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                           'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step_seq', '', 'Requested step to restore')
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


def eval_once(saver, summary_writer, summary_op, logits, labels, num_eval, run_id, requested_step=None):
    """Run Eval once.
    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    _loss = eval_loss(logits, labels)
    top1 = tf.nn.in_top_k(logits, labels, 1)
    top2 = tf.nn.in_top_k(logits, labels, 2)

    with tf.Session() as sess:
        checkpoint_path = '%s/run-%d/train' % (FLAGS.train_dir, run_id)

        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
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


def evaluate(run_dir, run_id):
    with tf.Graph().as_default() as g:
        input_file = os.path.join(FLAGS.train_dir, 'md.json')
        print(input_file)
        with open(input_file, 'r') as f:
            md = json.load(f)

        eval_data = FLAGS.eval_data == 'valid'
        num_eval = md['%s_counts' % FLAGS.eval_data]

        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id):
            print('Executing on %s' % FLAGS.device_id)
            images, labels, _ = inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size, train=not eval_data,
                                       num_preprocess_threads=FLAGS.num_preprocess_threads)
            logits = model_fn(md['nlabels'], images, 1, False)
            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(run_dir, g)
            saver = tf.train.Saver()

            print('Validation')
            eval_once(saver, summary_writer, summary_op, logits, labels, num_eval, run_id)
            '''
            for requested_step in range(6000, 10000, 1000):
                print('Running %s' % requested_step)
                eval_once(saver, summary_writer, summary_op, logits, labels, num_eval, requested_step)
            if FLAGS.requested_step_seq:
                sequence = FLAGS.requested_step_seq.split(',')
                for requested_step in sequence:
                    print('Running %s' % sequence)
                    eval_once(saver, summary_writer, summary_op, logits, labels, num_eval, requested_step)
            else:
                while True:
                    print('Running loop')
                    eval_once(saver, summary_writer, summary_op, logits, labels, num_eval)
                    if FLAGS.run_once:
                        break
                    time.sleep(FLAGS.eval_interval_secs)
            '''


def main(argv=None):  # pylint: disable=unused-argument
    run_dir = '%s/run-%d/valid' % (FLAGS.train_dir, FLAGS.run_id)
    if tf.gfile.Exists(run_dir):
        tf.gfile.DeleteRecursively(run_dir)
    tf.gfile.MakeDirs(run_dir)
    evaluate(run_dir, FLAGS.run_id)

if __name__ == '__main__':
    tf.app.run()

