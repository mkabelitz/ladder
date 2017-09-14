import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops


# Function to transform single digit class labels to one-hot
def labels_to_one_hot(labels, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    for i, l in enumerate(labels):
        one_hot[i][l] = 1.0
    return one_hot


# Function for shuffling data (and possibly labels in unison)
def shuffle_data(data, labels=None):
    perm = np.random.permutation(data.shape[0])
    shuffled_data = data[perm]
    shuffled_labels = None if labels is None else labels[perm]
    return shuffled_data, shuffled_labels


# Function for creating a class balanced subset of a labeled data set (expects 1-hot labels)
def make_class_balanced_set(data, labels, num_labeled=None):
    if num_labeled is None or num_labeled == data.shape[0]:
        return data, labels

    num_classes = labels.shape[1]
    assert num_labeled % num_classes == 0.0, "Can't make balanced data set of size %d!" % num_labeled

    data_balanced = np.zeros(((num_labeled,) + data.shape[1:]))
    labels_balanced = np.zeros((num_labeled, num_classes))

    per_class = int(num_labeled / num_classes)
    for i, l in enumerate(range(num_classes)):
        w = np.where(np.argmax(labels, axis=1) == l)[0]
        idx = np.random.choice(w, size=per_class)
        data_balanced[i * per_class:(i + 1) * per_class] = data[idx]
        labels_balanced[i * per_class:(i + 1) * per_class] = labels[idx]

    return data_balanced, labels_balanced


def load_batch(data, labels, batch_size):
    images, labels = tf.train.batch(
        [data, labels],
        batch_size=batch_size,
        enqueue_many=True,
        allow_smaller_final_batch=False
    )
    return images, labels


def load_shuffle_batch(data, labels, batch_size, capacity, min_after_dequeue):
    images, labels = tf.train.shuffle_batch(
        [data, labels],
        batch_size=batch_size,
        enqueue_many=True,
        allow_smaller_final_batch=False,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return images, labels


def load_unlabeled_batch(data, batch_size):
    images, labels = tf.train.batch(
        [data],
        batch_size=batch_size,
        enqueue_many=True,
        allow_smaller_final_batch=False
    )
    return images


def load_unlabeled_shuffle_batch(data, batch_size, capacity, min_after_dequeue):
    images, labels = tf.train.shuffle_batch(
        [data],
        batch_size=batch_size,
        enqueue_many=True,
        allow_smaller_final_batch=False,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )
    return images


def get_softmax_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def get_l2_regularization_loss():
    return tf.reduce_sum(tf.losses.get_regularization_losses())


def get_supervised_loss(logits, labels):
    softmax_loss = get_softmax_loss(logits, labels)
    l2_regularization_loss = get_l2_regularization_loss()
    total_loss = softmax_loss + l2_regularization_loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        total_loss = control_flow_ops.with_dependencies([updates], total_loss)
    return total_loss


def get_denoising_loss(crt, cln, denoising_cost):
    return tf.reduce_mean(tf.reduce_sum(tf.square(crt - cln), 1)) / crt.get_shape().as_list()[-1] * denoising_cost


def get_accuracy(logits, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1)), tf.float32))


def get_adam_haeusser(learning_rate, step, decay_steps, decay_factor, staircase=True):
    exp_decay = tf.train.exponential_decay(learning_rate, step, decay_steps, decay_factor, staircase=staircase)
    return tf.train.AdamOptimizer(learning_rate=exp_decay)


def get_adam_rasmus(learning_rate, step, num_total_iters, decay_first):
    lr_const = tf.constant(learning_rate, dtype=tf.float32)
    decay_start_lr = (1.0 / (1.0 - decay_first)) * learning_rate
    pol_decay_rate = tf.train.polynomial_decay(learning_rate=decay_start_lr,
                                               global_step=step,
                                               decay_steps=num_total_iters,
                                               end_learning_rate=0.000,
                                               power=1.0)
    step_thresh = int(num_total_iters * decay_first)
    cond = tf.less(step, step_thresh)
    adam_lr = tf.cond(cond, lambda: lr_const, lambda: pol_decay_rate)
    return tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=0.9)


def get_train_op(optimizer, loss, step, bn_assigns=None):
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)
    if bn_assigns:
        bn_updates = tf.group(*bn_assigns)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(bn_updates)
    return train_op
