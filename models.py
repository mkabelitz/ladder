import tensorflow as tf
import tensorflow.contrib.slim as slim


def _cifar10_supervised_rasmus(inputs, is_training, emb_size=10, l2_weight_decay=0.0, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(l2_weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
        net = slim.conv2d(net, 96, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 96, [3, 3], scope='conv1_2')
        net = slim.conv2d(net, 96, [3, 3], scope='conv1_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 192, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 192, [3, 3], scope='conv2_2')
        net = slim.conv2d(net, 192, [3, 3], scope='conv2_3')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.conv2d(net, 192, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 192, [1, 1], scope='conv3_2')
        net = slim.conv2d(net, 10, [1, 1], scope='conv3_3')
        net = slim.avg_pool2d(net, [7, 7], scope='pool3')

        emb = slim.flatten(net, scope='flatten')
    return emb


def _mnist_supervised_haeusser(inputs, is_training, emb_size=10, l2_weight_decay=1e-4, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.elu,
                        weights_regularizer=slim.l2_regularizer(l2_weight_decay)):
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14

        net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7

        net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

        net = slim.flatten(net, scope='flatten')
        emb = slim.fully_connected(net, emb_size, scope='fc1')
    return emb


def _mnist_supervised_rasmus(inputs, is_training, emb_size=10, l2_weight_decay=0.0, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(l2_weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
        net = slim.conv2d(net, 32, [5, 5], scope='conv1_1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
        net = slim.conv2d(net, 10, [1, 1], scope='conv3_2')
        net = slim.avg_pool2d(net, [7, 7], scope='pool3')

        net = slim.flatten(net, scope='flatten')
        emb = slim.fully_connected(net, emb_size, scope='fc1')
    return emb


def get_model(model_name):
    if model_name == 'cifar10_supervised_rasmus':
        return _cifar10_supervised_rasmus
    elif model_name == 'mnist_supervised_haeusser':
        return _mnist_supervised_haeusser
    elif model_name == 'mnist_supervised_rasmus':
        return _mnist_supervised_rasmus
    else:
        print('Model unknown!')
        exit(0)
