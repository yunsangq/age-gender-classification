from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf
from data import distorted_inputs, inputs
from model import inference
import json
import math

LAMBDA = 0.01
MOM = 0.9
tf.app.flags.DEFINE_string('train_dir', './Folds/tf/age_test_fold_is_2',
                           'Training directory')

tf.app.flags.DEFINE_string('eval_data', 'valid',
                           'Data type (valid|test)')

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

tf.app.flags.DEFINE_integer('max_steps', 30001,
                            'Number of iterations')

tf.app.flags.DEFINE_integer('epochs', -1,
                            'Number of epochs')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch size')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                           'Checkpoint name')

FLAGS = tf.app.flags.FLAGS


class Train(object):
    def __init__(self):
        self.pdrop = tf.placeholder(dtype=tf.float32)

        input_file = os.path.join(FLAGS.train_dir, 'md.json')
        print(input_file)
        with open(input_file, 'r') as f:
            self.md = json.load(f)

        self.train_acc1 = []
        self.train_acc2 = []
        self.valid_acc1 = []
        self.valid_acc2 = []
        self.train_cost = []
        self.valid_cost = []

    def train(self):
        images, labels, _ = distorted_inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size,
                                             mode='train',
                                             num_preprocess_threads=FLAGS.num_preprocess_threads)
        self.num_eval = self.md['%s_counts' % FLAGS.eval_data]
        val_images, val_labels, _ = inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size,
                                           mode='valid',
                                           num_preprocess_threads=FLAGS.num_preprocess_threads)

        logits = inference(images, self.md['nlabels'], self.pdrop, reuse=False)
        val_logits = inference(val_images, self.md['nlabels'], self.pdrop, reuse=True)

        self.total_loss = self.loss(logits, labels)
        self.train_op = self.optimizer(FLAGS.optim, FLAGS.eta, self.total_loss)
        self.train_top1 = tf.nn.in_top_k(logits, labels, 1)
        self._train_labels = tf.cast(labels, tf.int32)

        self.val_total_loss = self.eval_loss(val_logits, val_labels)
        self.top1 = tf.nn.in_top_k(val_logits, val_labels, 1)
        self._val_labels = tf.cast(val_labels, tf.int32)
        self.val_logits = val_logits

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        tf.global_variables_initializer().run(session=sess)

        run_dir = '%s/run-%d/train' % (FLAGS.train_dir, os.getpid())
        val_run_dir = '%s/run-%d/valid' % (FLAGS.train_dir, os.getpid())

        checkpoint_path = '%s/%s' % (run_dir, FLAGS.checkpoint)
        if tf.gfile.Exists(run_dir) is False:
            print('Creating %s' % run_dir)
            tf.gfile.MakeDirs(run_dir)

        tf.train.write_graph(sess.graph_def, run_dir, 'model.pb', as_text=True)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(run_dir, sess.graph)
        val_writer = tf.summary.FileWriter(val_run_dir, sess.graph)

        steps_per_train_epoch = int(self.md['train_counts'] / FLAGS.batch_size)
        num_steps = FLAGS.max_steps if FLAGS.epochs < 1 else FLAGS.epochs * steps_per_train_epoch
        print('Requested number of steps [%d]' % num_steps)

        print('Start training...')
        print('----------------------------')

        for step in xrange(num_steps):
            if step == 0:
                self.eval_once(sess, step, val_writer, summary_op)

            cnt_top1 = cnt_top2 = 0.0
            start_time = time.time()
            _, loss_value, train_top1, batch_labels, v = sess.run([self.train_op, self.total_loss,
                                                                  self.train_top1, self._train_labels,
                                                                  logits],
                                                                  {self.pdrop: FLAGS.pdrop})
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
                v = np.argmax(v, axis=1)
                for i in range(len(v)):
                    if v[i] == batch_labels[i] or v[i] == batch_labels[i] - 1 or v[i] == batch_labels[i] + 1:
                        cnt_top2 += 1
                cnt_top1 += np.sum(train_top1)
                print('Train')
                print('step%d: loss = %.3f' % (step, loss_value))
                print('step%d: precision @ 1 = %.3f (%d/%d)' %
                      (step, cnt_top1/len(train_top1), cnt_top1, len(train_top1)))
                print('step%d: precision @ 2 = %.3f (%d/%d)' %
                      (step, cnt_top2/len(v), cnt_top2, len(v)))
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op, {self.pdrop: FLAGS.pdrop}))
                summary.value.add(tag='Precision @ 1', simple_value=cnt_top1/len(train_top1))
                summary.value.add(tag='Precision @ 2', simple_value=cnt_top2/len(v))
                summary.value.add(tag='cost', simple_value=loss_value)
                summary_writer.add_summary(summary, step)
                self.train_acc1.append(float(cnt_top1 / len(train_top1)))
                self.train_acc2.append(float(cnt_top2 / len(v)))
                self.train_cost.append(float(loss_value))
                if step != 0:
                    self.eval_once(sess, step, val_writer, summary_op)

            if step % 1000 == 0 or (step + 1) == num_steps:
                saver.save(sess, checkpoint_path, global_step=step)

        self.save(run_dir + "/age_test_fold_0.json")

    # Every 5k steps cut learning rate in half
    def exponential_staircase_decay(self, at_step=5000, decay_rate=0.5):
        def _decay(lr, global_step):
            return tf.train.exponential_decay(lr, global_step,
                                              at_step, decay_rate, staircase=True)

        return _decay

    def optimizer(self, optim, eta, loss_fn):
        global_step = tf.Variable(0, trainable=False)
        optz = optim
        if optim == 'Adadelta':
            optz = lambda lr: tf.train.AdadeltaOptimizer(lr, 0.95, 1e-6)
            lr_decay_fn = None
        elif optim == 'Momentum':
            optz = lambda lr: tf.train.MomentumOptimizer(lr, MOM)
            lr_decay_fn = self.exponential_staircase_decay()

        return tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz, clip_gradients=4.,
                                               learning_rate_decay_fn=lr_decay_fn)

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int32)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        losses = tf.get_collection('losses')
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses)
        tf.summary.scalar('tl (raw)', total_loss)
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss

    def eval_loss(self, logits, labels):
        labels = tf.cast(labels, tf.int32)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='val_cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='val_cross_entropy')
        return cross_entropy_mean

    def eval_once(self, sess, global_step, summary_writer, summary_op):
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_steps = int(math.ceil(self.num_eval / FLAGS.batch_size))
            true_count1 = true_count2 = 0
            total_loss = 0.0
            total_sample_count = num_steps * FLAGS.batch_size
            step = 0
            print('Validation')

            while step < num_steps and not coord.should_stop():
                predictions1, loss_value, v, batch_labels = sess.run([self.top1,
                                                                      self.val_total_loss,
                                                                      self.val_logits,
                                                                      self._val_labels],
                                                                      {self.pdrop: 1})
                v = np.argmax(v, axis=1)
                predictions2 = 0
                for i in range(len(v)):
                    if v[i] == batch_labels[i] or v[i] == batch_labels[i] - 1 or v[i] == batch_labels[i] + 1:
                        predictions2 += 1

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
            summary.ParseFromString(sess.run(summary_op, {self.pdrop: 1}))
            summary.value.add(tag='Precision @ 1', simple_value=precision1)
            summary.value.add(tag='Precision @ 2', simple_value=precision2)
            summary.value.add(tag='cost', simple_value=total_loss)
            summary_writer.add_summary(summary, global_step)
            self.valid_acc1.append(float(precision1))
            self.valid_acc2.append(float(precision2))
            self.valid_cost.append(float(total_loss))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    def save(self, filename):
        data = {"train_cost": self.train_cost,
                "valid_cost": self.valid_cost,
                "train_accuracy1": self.train_acc1,
                "train_accuracy2": self.train_acc2,
                "valid_accuracy1": self.valid_acc1,
                "valid_accuracy2": self.valid_acc2}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        print("save json data.")


if __name__ == '__main__':
    train = Train()
    train.train()
