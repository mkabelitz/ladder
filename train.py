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

import input_data
import models
import utils as u

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_labeled', None, 'Number of labeled samples to use for training. (None = all labeled samples)')
flags.DEFINE_integer('batch_size', 100, 'Number of samples used per batch.')
flags.DEFINE_integer('num_iters', 24000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 600, 'Number of steps between evaluations.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate for optimizer.')
flags.DEFINE_float('lr_decay_steps', 5000, 'Learning rate exponential decay factor.')
flags.DEFINE_float('lr_decay_factor', 0.33, 'Learning rate exponential decay factor.')
flags.DEFINE_string('dataset_name', 'mnist', 'Name of the dataset to be used.')
flags.DEFINE_string('model_name', 'mnist_supervised', 'Name of the model to be used.')
flags.DEFINE_string('optimizer_type', 'adam', 'Type of the optimizer to be used.')


def main(_):
    data_tr, labels_tr, data_te, labels_te, unlabeled = input_data.load_data(FLAGS.dataset_name, FLAGS.num_labeled)

    data_tr_batch, labels_tr_batch = u.load_shuffle_batch(data_tr,
                                                          labels_tr,
                                                          batch_size=FLAGS.batch_size,
                                                          capacity=FLAGS.batch_size * 100,
                                                          min_after_dequeue=FLAGS.batch_size * 10)
    data_te_batch, labels_te_batch = u.load_batch(data_te, labels_te, FLAGS.batch_size)

    with tf.variable_scope('model') as scope:
        model = models.get_model(FLAGS.model_name)
        logits_tr = model(data_tr_batch)
        scope.reuse_variables()
        logits_te = model(data_te_batch)

    l2_regularization_loss = u.get_l2_regularization_loss()

    loss_tr = l2_regularization_loss + u.get_batch_softmax_loss(logits=logits_tr, labels=labels_tr_batch)
    loss_te = l2_regularization_loss + u.get_batch_softmax_loss(logits=logits_te, labels=labels_te_batch)

    acc_tr = u.get_batch_accuracy(logits_tr, labels_tr_batch)
    acc_te = u.get_batch_accuracy(logits_te, labels_te_batch)

    step = tf.Variable(0, trainable=False, dtype=tf.int32)
    optimizer = u.get_optimizer(FLAGS.optimizer_type, FLAGS.learning_rate, step, FLAGS.lr_decay_steps,
                                FLAGS.lr_decay_factor)
    train_op = u.get_train_op(optimizer, loss_tr, step)

    with tf.Session() as sess:

        def eval_test():
            loss = 0.0
            acc = 0.0
            eval_iters = int(data_te.shape[0] / FLAGS.batch_size)
            for j in range(eval_iters):
                l, a = sess.run([loss_te, acc_te])
                loss += l
                acc += a
            loss /= eval_iters
            acc /= eval_iters
            return loss, acc

        # initialize the variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in tqdm(range(FLAGS.num_iters)):
            _, cur_loss_tr, cur_acc_tr = sess.run([train_op, loss_tr, acc_tr])

            if i % FLAGS.eval_interval == 0:
                print('train loss: %.4f train acc: %.4f' % (cur_loss_tr, cur_acc_tr))
                cur_loss_te, cur_acc_te = eval_test()
                print(' test loss: %.4f  test acc: %.4f' % (cur_loss_te, cur_acc_te))

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
