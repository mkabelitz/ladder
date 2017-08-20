"""
Imports.
"""

import os
import sys
import datetime
from shutil import copyfile

import pickle
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.training import saver as tf_saver

import utils
import models

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_labeled', None, 'Number of labeled samples to use for training.')
flags.DEFINE_integer('batch_size', 100, 'Number of samples used per batch.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for optimizer')


def load_mnist(pkl_file_path, num_labeled=None):
    with open(pkl_file_path, 'rb') as f:
        unpickler = pickle._Unpickler(f)
        unpickler.encoding = 'latin1'  # need this bc of some Python3 problem
        train_set, valid_set, test_set = unpickler.load()

    # Extract image data
    data_tr = np.array(train_set[0], dtype=np.float32).reshape((-1, 28, 28, 1))
    data_va = np.array(valid_set[0], dtype=np.float32).reshape((-1, 28, 28, 1))
    data_te = np.array(test_set[0], dtype=np.float32).reshape((-1, 28, 28, 1))

    # Get target data as one-hot encoded
    labels_tr = utils.labels_to_one_hot(train_set[1], 10)
    labels_va = utils.labels_to_one_hot(valid_set[1], 10)
    labels_te = utils.labels_to_one_hot(test_set[1], 10)

    # Combine training and validation data for final training
    data_tr = np.concatenate((data_tr, data_va))
    labels_tr = np.concatenate((labels_tr, labels_va))

    # Get all training samples as unlabeled samples
    unlabeled = np.concatenate((data_tr, data_va))

    # Make class balanced training set
    data_tr, labels_tr = utils.make_class_balanced_set(data_tr, labels_tr, num_labeled)

    return data_tr, labels_tr, data_te, labels_te, unlabeled


def main(_):

    data_tr, labels_tr, data_te, labels_te, unlabeled = load_mnist('./mnist/data/mnist.pkl', FLAGS.num_labeled)

    data_tr_batch, labels_tr_batch = utils.load_shuffle_batch(data_tr, labels_tr, FLAGS.batch_size, 5000, 1000)
    data_te_batch, labels_te_batch = utils.load_batch(data_te, labels_te, FLAGS.batch_size)

    with tf.variable_scope('bla') as scope:
        logits_tr = models.mnist_model(data_tr_batch, emb_size=10)
        scope.reuse_variables()
        logits_te = models.mnist_model(data_te_batch, emb_size=10)

    loss_tr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_tr, labels=labels_tr_batch))
    loss_te = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_te, labels=labels_tr_batch))

    acc_tr = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits_tr, 1), tf.arg_max(labels_tr_batch, 1)), tf.float32))
    acc_te = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits_te, 1), tf.arg_max(labels_te_batch, 1)), tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9)

    step = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_tr, global_step=step, name='train_op')

    with tf.Session() as sess:

        # initialize the variables
        sess.run(tf.initialize_all_variables())

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("from the train set:")
        for i in range(10000000):
            _, tr_batch, loss_tmp, acc_tmp = sess.run([train_op, data_tr_batch, loss_tr, acc_tr])

            if i % 100 == 0:
                print(i, ':')
                print('\ttrain loss: %.4f train acc: %.4f' % (loss_tmp, acc_tmp))
                l = 0.0
                a = 0.0
                for j in range(100):
                    te_batch, loss_tmp, acc_tmp = sess.run([data_te_batch, loss_te, acc_te])
                    l += loss_tmp
                    a += acc_tmp
                l /= 100.0
                a /= 100.0
                print('\ttest loss: %.4f test acc: %.4f' % (l, a))


        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
