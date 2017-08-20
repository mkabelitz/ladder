import tensorflow as tf
import tensorflow.contrib.slim as slim


def _mnist_supervised(inputs, emb_size=10, l2_weight_decay=1e-4, batch_norm_decay=0.99):
    inputs = tf.cast(inputs, tf.float32)

    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.elu,
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


def get_model(model_name):
    if model_name == 'mnist_supervised':
        return _mnist_supervised
    else:
        print('Model unknown!')
        exit(0)
