"""
Accuracies: (0.8611, 0.8321, 0.8218, 0.8124)
Target: no target (Rasmus 0.9357)
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

flags.DEFINE_integer('num_labeled', 100, 'Number of labeled samples to use for training. (None = all labeled samples)')
flags.DEFINE_integer('batch_size', 100, 'Number of samples used per batch.')
flags.DEFINE_integer('num_iters', 1000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', None, 'Number of steps between evaluations.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for optimizer.')
flags.DEFINE_float('lr_decay_steps', 400, 'Interval of steps for learning rate decay.')
flags.DEFINE_float('lr_decay_factor', 0.33, 'Learning rate exponential decay factor.')


def main(_):
    data_tr, labels_tr, data_te, labels_te, unlabeled = input_data.load_mnist(num_labeled=FLAGS.num_labeled)
    print("    train shapes:", data_tr.shape, labels_tr.shape)
    print("     test shapes:", data_te.shape, labels_te.shape)
    print("unlabeled shapes:", unlabeled.shape)

    data_tr_batch, labels_tr_batch = u.load_shuffle_batch(data_tr,
                                                          labels_tr,
                                                          batch_size=FLAGS.batch_size,
                                                          capacity=FLAGS.batch_size * 100,
                                                          min_after_dequeue=FLAGS.batch_size * 20)
    data_te_batch, labels_te_batch = u.load_batch(data_te, labels_te, FLAGS.batch_size)
    unlabeled_batch, _ = u.load_shuffle_batch(unlabeled,
                                              unlabeled,
                                              batch_size=FLAGS.batch_size,
                                              capacity=FLAGS.batch_size * 100,
                                              min_after_dequeue=FLAGS.batch_size * 20)

    with tf.variable_scope('model') as scope:
        logits_tr, _ = models.mnist_assoc(data_tr_batch)
        # scope.reuse_variables()
        _, emb = models.mnist_assoc(unlabeled_batch)
        logits_te, _ = models.mnist_assoc(data_te_batch)

    loss_tr = u.get_supervised_loss(logits=logits_tr, labels=labels_tr_batch)
    loss_te = u.get_supervised_loss(logits=logits_te, labels=labels_te_batch)

    acc_tr = u.get_accuracy(logits_tr, labels_tr_batch)
    acc_te = u.get_accuracy(logits_te, labels_te_batch)

    step = tf.Variable(0, trainable=False, dtype=tf.int32)
    optimizer = u.get_adam_haeusser(learning_rate=FLAGS.learning_rate,
                                    step=step,
                                    decay_steps=FLAGS.lr_decay_steps,
                                    decay_factor=FLAGS.lr_decay_factor)
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


def add_semisup_loss(self, a, b, labels, walker_weight=1.0, visit_weight=1.0):
    """Add semi-supervised classification loss to the model.
    The loss constist of two terms: "walker" and "visit".
    Args:
      a: [N, emb_size] tensor with supervised embedding vectors.
      b: [M, emb_size] tensor with unsupervised embedding vectors.
      labels : [N] tensor with labels for supervised embeddings.
      walker_weight: Weight coefficient of the "walker" loss.
      visit_weight: Weight coefficient of the "visit" loss.
    """

    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(
        equality_matrix, [1], keep_dims=True))

    match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

    self.create_walk_statistics(p_aba, equality_matrix)

    loss_aba = tf.losses.softmax_cross_entropy(
        p_target,
        tf.log(1e-8 + p_aba),
        weights=walker_weight,
        scope='loss_aba')
    self.add_visit_loss(p_ab, visit_weight)


def add_visit_loss(self, p, weight=1.0):
    """Add the "visit" loss to the model.
    Args:
      p: [N, M] tensor. Each row must be a valid probability distribution
          (i.e. sum to 1.0)
      weight: Loss weight.
    """
    visit_probability = tf.reduce_mean(
        p, [0], keep_dims=True, name='visit_prob')
    t_nb = tf.shape(p)[1]
    visit_loss = tf.losses.softmax_cross_entropy(
        tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
        tf.log(1e-8 + visit_probability),
        weights=weight,
        scope='loss_visit')


def create_walk_statistics(self, p_aba, equality_matrix):
    """Adds "walker" loss statistics to the graph.
    Args:
      p_aba: [N, N] matrix, where element [i, j] corresponds to the
          probalility of the round-trip between supervised samples i and j.
          Sum of each row of 'p_aba' must be equal to one.
      equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
          i and j belong to the same class.
    """
    # Using the square root of the correct round trip probalilty as an estimate
    # of the current classifier accuracy.
    per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1)**0.5
    estimate_error = tf.reduce_mean(
        1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')
    self.add_average(estimate_error)
    self.add_average(p_aba)


def add_average(self, variable):
    """Add moving average variable to the model."""
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.ema.apply([variable]))
    average_variable = tf.identity(
        self.ema.average(variable), name=variable.name[:-2] + '_avg')
    return average_variable


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
