import tensorflow as tf
import tensorflow.contrib.slim as slim


# Function for adding batch normalization beta parameter
def _add_bias(data):
    own_beta = tf.get_variable('own_beta', shape=data.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return data + own_beta


# Function for scaling by batch normalization gamma parameter
def _apply_scale(data):
    own_gamma = tf.get_variable('own_gamma', shape=data.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    return data * own_gamma


def _gamma_layer(data, activation_fn, is_training, is_unlabeled, noise_std, ema, bn_assigns):

    running_mean_enc = tf.get_variable('running_mean_enc', shape=[data.get_shape()[-1]], trainable=False,
                                       initializer=tf.constant_initializer(0.0))
    running_var_enc = tf.get_variable('running_var_enc', shape=[data.get_shape()[-1]], trainable=False,
                                      initializer=tf.constant_initializer(1.0))
    mean_enc, var_enc = tf.nn.moments(data, axes=[0])
    print("1.1")
    if is_unlabeled:
        print("1.2")
        assign_mean_enc = running_mean_enc.assign(mean_enc)
        assign_var_enc = running_var_enc.assign(var_enc)
        bn_assigns.append(ema.apply([running_mean_enc, running_var_enc]))
        with tf.control_dependencies([assign_mean_enc, assign_var_enc]):
            normalized_enc = (data - mean_enc) / tf.sqrt(var_enc + 1e-10)
    elif is_training:
        print("1.3")
        normalized_enc = (data - mean_enc) / tf.sqrt(var_enc + 1e-10)
    else:
        print("1.4")
        normalized_enc = (data - ema.average(running_mean_enc)) / tf.sqrt(ema.average(running_var_enc) + 1e-10)

    z_tilde = _noise(normalized_enc, noise_std)
    with tf.variable_scope('bn_correct'):
        bn_corrected_tilde = _apply_scale(_add_bias(z_tilde))
    h_tilde = activation_fn(bn_corrected_tilde)

    z = normalized_enc
    with tf.variable_scope('bn_correct', reuse=True):
        bn_corrected = _apply_scale(_add_bias(z))
    h = activation_fn(bn_corrected)

    running_mean_dec = tf.get_variable('running_mean_dec', shape=[data.get_shape()[-1]], trainable=False,
                                       initializer=tf.constant_initializer(0.0))
    running_var_dec = tf.get_variable('running_var_dec', shape=[data.get_shape()[-1]], trainable=False,
                                      initializer=tf.constant_initializer(1.0))
    mean_dec, var_dec = tf.nn.moments(h_tilde, axes=[0])
    print("2.1")
    if is_unlabeled:
        print("2.2")
        assign_mean_dec = running_mean_dec.assign(mean_dec)
        assign_var_dec = running_var_dec.assign(var_dec)
        bn_assigns.append(ema.apply([running_mean_dec, running_var_dec]))
        with tf.control_dependencies([assign_mean_dec, assign_var_dec]):
            normalized_dec = (h_tilde - mean_dec) / tf.sqrt(var_dec + 1e-10)
    elif is_training:
        print("2.3")
        normalized_dec = (h_tilde - mean_dec) / tf.sqrt(var_dec + 1e-10)
    else:
        print("2.4")
        normalized_dec = (h_tilde - ema.average(running_mean_dec)) / tf.sqrt(ema.average(running_var_dec) + 1e-10)

    z_est = _g(z_tilde, normalized_dec)

    return h, z_est, z


def _leaky_relu(features, name=None):
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


def cifar10_gamma(inputs, is_training, batch_norm_decay=0.9, noise_std=0.3):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
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


def cifar10_supervised_rasmus(inputs, is_training, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=_leaky_relu,
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

        logits = slim.flatten(net, scope='flatten')
    return logits


def mnist_gamma(inputs, is_training, is_unlabeled, ema, bn_assigns, batch_norm_decay=0.9, noise_std=0.3):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
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

    net = tf.layers.dense(net, 10, use_bias=False, name='dense')
    logits, z_crt, z_cln = _gamma_layer(net, lambda x: x, is_training=is_training, is_unlabeled=is_unlabeled,
                                        noise_std=noise_std, ema=ema, bn_assigns=bn_assigns)
    return logits, z_crt, z_cln


def mnist_supervised_haeusser(inputs, emb_size=128, l2_weight_decay=1e-3):
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
        logits = slim.fully_connected(emb, 10, scope='fc2')
    return logits


def mnist_supervised_rasmus(inputs, is_training, batch_norm_decay=0.9):
    inputs = tf.cast(inputs, tf.float32)
    net = inputs
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
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
        logits = slim.fully_connected(net, 10, scope='fc1')
    return logits
