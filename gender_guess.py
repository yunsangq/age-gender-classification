from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model import inference, get_checkpoint
from utils import ImageCoder, make_batch
from detect import face_detection_model
import os
import csv

RESIZE_FINAL = 227
GENDER_LIST = ['M', 'F']
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

tf.app.flags.DEFINE_string('model_dir', './Folds/tf/gender_test_fold_is_0/run-5508/train',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'gender',
                           'Classification type (age|gender)')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', './example/example_image14.jpg',
                           'File (Image) or File list (Text/No header TSV) to process')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')


FLAGS = tf.app.flags.FLAGS


def one_of(fname, types):
    for ty in types:
        if fname.endswith('.' + ty):
            return True
    return False


def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None


def classify(sess, label_list, softmax_output, coder, images, image_file):
    print('Running file %s' % image_file)
    image_batch = make_batch(image_file, coder, not FLAGS.single_look)
    batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})
    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]

    output /= batch_sz
    best = np.argmax(output)
    best_choice = (label_list[best], output[best])
    print('Guess @ 1 %s, prob = %.2f' % best_choice)

    nlabels = len(label_list)
    if nlabels > 2:
        output[best] = 0
        second_best = np.argmax(output)

        print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))
    return best_choice


def batchlist(srcfile):
    with open(srcfile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            reader.next()

        return [row[0] for row in reader]


def main(argv=None):  # pylint: disable=unused-argument
    files = []

    if FLAGS.face_detection_model:
        print('Using face detector (%s) %s' % (FLAGS.face_detection_type, FLAGS.face_detection_model))
        face_detect = face_detection_model(FLAGS.face_detection_type, FLAGS.face_detection_model)
        face_files, rectangles = face_detect.run(FLAGS.filename)
        print(face_files)
        files += face_files

    with tf.Session() as sess:
        label_list = AGE_LIST if FLAGS.class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)

        images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
        logits = inference(images, nlabels, 1, reuse=False)

        checkpoint_path = '%s' % (FLAGS.model_dir)
        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path)

        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)

        softmax_output = tf.nn.softmax(logits)

        coder = ImageCoder()

        # Support a batch mode if no face detection model
        if len(files) == 0:
            files.append(FLAGS.filename)
            # If it happens to be a list file, read the list and clobber the files
            if one_of(FLAGS.filename, ('csv', 'tsv', 'txt')):
                files = batchlist(FLAGS.filename)

        writer = None
        output = None
        if FLAGS.target:
            print('Creating output file %s' % FLAGS.target)
            output = open(FLAGS.target, 'w')
            writer = csv.writer(output)
            writer.writerow(('file', 'label', 'score'))

        for f in files:
            image_file = resolve_file(f)

            if image_file is None: continue

            try:
                best_choice = classify(sess, label_list, softmax_output, coder, images, image_file)
                if writer is not None:
                    writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
            except Exception as e:
                print(e)
                print('Failed to run image %s ' % image_file)

        if output is not None:
            output.close()


if __name__ == '__main__':
    tf.app.run()
