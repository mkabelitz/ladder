"""
Best: FINAL TEST LOSS: 0.5576  FINAL TEST ACC: 0.9067
Target: 0.9073
"""

import os

from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import input_data
import models
import utils as u

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_labeled', None, 'Number of labeled samples to use for training. (None = all labeled samples)')
flags.DEFINE_integer('batch_size', 100, 'Number of samples used per batch.')
flags.DEFINE_integer('num_iters', 10000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 10001, 'Number of steps between evaluations.')
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate for optimizer.')
flags.DEFINE_float('decay_first', 0.5, 'Percentage after when to start learning rate decay.')


def main(_):
    data_tr, labels_tr, data_te, labels_te, unlabeled = input_data.load_cifar10(num_labeled=FLAGS.num_labeled)
    print("    train shapes:", data_tr.shape, labels_tr.shape)
    print("     test shapes:", data_te.shape, labels_te.shape)
    print("unlabeled shapes:", unlabeled.shape)

    data_tr_batch, labels_tr_batch = u.load_shuffle_batch(data_tr,
                                                          labels_tr,
                                                          batch_size=FLAGS.batch_size,
                                                          capacity=FLAGS.batch_size * 100,
                                                          min_after_dequeue=FLAGS.batch_size * 20)
    data_te_batch, labels_te_batch = u.load_batch(data_te, labels_te, FLAGS.batch_size)

    with tf.variable_scope('model') as scope:
        logits_tr = models.cifar10_supervised_rasmus(data_tr_batch, is_training=True)
        scope.reuse_variables()
        logits_te = models.cifar10_supervised_rasmus(data_te_batch, is_training=False)

    loss_tr = u.get_supervised_loss(logits=logits_tr, labels=labels_tr_batch)
    loss_te = u.get_supervised_loss(logits=logits_te, labels=labels_te_batch)

    acc_tr = u.get_accuracy(logits_tr, labels_tr_batch)
    acc_te = u.get_accuracy(logits_te, labels_te_batch)

    step = tf.Variable(0, trainable=False, dtype=tf.int32)
    optimizer = u.get_adam_rasmus(step=step, learning_rate=FLAGS.learning_rate,
                                  num_total_iters=FLAGS.num_iters, decay_first=FLAGS.decay_first)
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

            if FLAGS.eval_interval is not None and i % FLAGS.eval_interval == 0:
                print('train loss: %.4f train acc: %.4f' % (cur_loss_tr, cur_acc_tr))
                cur_loss_te, cur_acc_te = eval_test()
                print(' test loss: %.4f  test acc: %.4f' % (cur_loss_te, cur_acc_te))

        print('\nOPTIMIZATION FINISHED!')
        final_loss_te, final_acc_te = eval_test()
        print('FINAL TEST LOSS: %.4f  FINAL TEST ACC: %.4f' % (final_loss_te, final_acc_te))

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
