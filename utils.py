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
    if num_labeled == None or num_labeled == data.shape[0]:
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


def get_softmax_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def get_l2_regularization_loss():
    return tf.reduce_sum(tf.losses.get_regularization_losses())


def get_total_loss(logits, labels):
    softmax_loss = get_softmax_loss(logits, labels)
    l2_regularization_loss = get_l2_regularization_loss()
    total_loss = softmax_loss + l2_regularization_loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        total_loss = control_flow_ops.with_dependencies([updates], total_loss)
    return total_loss


def get_accuracy(logits, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1)), tf.float32))


def get_optimizer(optimizer_type, learning_rate, step, decay_steps, decay_factor, staircase=True):
    exp_decay = tf.train.exponential_decay(learning_rate, step, decay_steps, decay_factor, staircase=staircase)
    if optimizer_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate=exp_decay, beta1=0.9)
    else:
        print('Optimizer unknown!')
        exit(0)


def get_train_op(optimizer, loss, step):
    return slim.learning.create_train_op(loss, optimizer, global_step=step)
