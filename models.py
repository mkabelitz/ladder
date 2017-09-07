import tensorflow as tf
import tensorflow.contrib.slim as slim


def leaky_relu(features, name=None):
    alpha = 0.1
    return tf.maximum(features, alpha * features)


def _g_m(u):
    a1 = tf.get_variable('a1', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a2 = tf.get_variable('a2', shape=u.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    a3 = tf.get_variable('a3', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a4 = tf.get_variable('a4', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a5 = tf.get_variable('a5', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5


def _g_v(u):
    a6 = tf.get_variable('a6', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a7 = tf.get_variable('a7', shape=u.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    a8 = tf.get_variable('a8', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a9 = tf.get_variable('a9', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a10 = tf.get_variable('a10', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10


# The combinator function described in the paper, initial values from https://github.com/CuriousAI/ladder/
def _g(z_crt, u):
    m = _g_m(u)
    return (z_crt - m) * _g_v(u) + m


def _noise(data, noise_std):
    new_noise = tf.random_normal(shape=tf.shape(data), mean=0.0, stddev=noise_std, dtype=tf.float32)
    result = tf.add(data, new_noise)
    result.set_shape(data.get_shape())
    return result


def cifar10_gamma(inputs, is_training, emb_size=10, l2_weight_decay=0.0, batch_norm_decay=0.9, noise_std=0.3):
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

        net_noisy = _noise(net, noise_std=noise_std)
        mean, var = tf.nn.moments(net_noisy, axes=[0, 1, 2])
        net_noisy_norm = (net_noisy - mean) / tf.sqrt(var + tf.constant(1e-10))

        comb = _g(net_noisy, net_noisy_norm)

    return emb, net, comb


def cifar10_supervised_rasmus(inputs, is_training, l2_weight_decay=0.0, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=leaky_relu,
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


def mnist_supervised_haeusser(inputs, is_training, emb_size=10, l2_weight_decay=1e-4, batch_norm_decay=0.9):
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


def mnist_supervised_rasmus(inputs, is_training, l2_weight_decay=0.0, batch_norm_decay=0.9):
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
        emb = slim.fully_connected(net, 10, scope='fc1')
    return emb
