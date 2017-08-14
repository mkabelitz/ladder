"""
Convolutional Ladder Network for CIFAR-10
Author: Marco Kabelitz (marco.kabelitz@rwth-aachen.de)

Based on 'Semi-Supervised Learning with Ladder Networks' by Rasmus, Valpola, et al.
    - Paper: https://arxiv.org/abs/1507.02672
    -  Code: https://github.com/CuriousAI/ladder

The code for batch normalization, encoder, decoder, and combinator function is adapted from Rinu Boney's code found at
https://github.com/rinuboney/ladder/. By 'adapted' I mean 'copied and pasted as far as possible', but since his code
only works for fully connected MLPs, quite some work had to be put into enabling the use of convolutional networks.
"""

# ----------------------------------------------------------------------------------------------------------------------
"""
Imports.
"""

import numpy as np
import tensorflow as tf

import os
import sys
import datetime
from shutil import copyfile

import pickle

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ----------------------------------------------------------------------------------------------------------------------
"""
Hyperparameters.
"""

# MNIST data information
NUM_CLASSES = 10  # Number of classes
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3

# Labeled and unlabeled examples
NUM_LABELED = 4000  # Number of labeled examples
BATCH_SIZE = 200  # Batch size, including labeled and unlabeled examples
NUM_EPOCHS = 70  # Number of epochs
NUM_LABELED_IN_BATCH = 100  # Number of labeled examples per batch
NUM_LABELED_IN_EPOCH = 50000  # Number of labeled examples per epoch
ITERS_PER_EPOCHE = 500  # Number of training steps/weight updates per epoch
NUM_TOTAL_ITERS = int(ITERS_PER_EPOCHE * NUM_EPOCHS)  # Total number of learning steps

# Model hyperparameters
INITIAL_LEARNING_RATE = 0.002  # Initial learning rate
LR_DECAY_FIRST = 0.86  # Percentage of epochs after which to start learning rate decay
NOISE_STD = 0.0  # Noise standard deviation for Gaussian noise applied to layers
RELU_TYPE = 'prelu'  # Type of ReLU activation function to use: 'relu', 'prelu'
WEIGHT_INITIALIZER_TYPE = 'he'  # Type of weight initializer: 'default', 'xavier', 'he'
WEIGHT_DECAY = 0.0  # Weight decay loss multiplication factor
OPTIMIZER = 'adam'  # Optimizer to use: 'adam', 'adadelta'

# Stuff for logging and saving stuff
EPOCHS_BEFORE_SAVING = NUM_EPOCHS + 1  # Number of epochs before starting to save the TF model
SAVING_INTERVAL_IN_EPOCHS = 100  # Interval in epochs for saving the TF model
TRAIN_SUMMARY_INTERVAL_IN_STEPS = 1  # Interval in learning steps for writing TF train summaries
EVAL_SUMMARY_INTERVAL_IN_STEPS = ITERS_PER_EPOCHE  # Interval in learning steps for evaluating test set and writing TF test summaries
FILE_NAME = '4k_white_gamma.py'
MODEL_NAME = 'cifar10_ladder' + \
             '_lr=' + str(INITIAL_LEARNING_RATE) + \
             '_lrdf=' + str(LR_DECAY_FIRST) + \
             '_std=' + str(NOISE_STD) + \
             '_nl=' + str(NUM_LABELED) + \
             '_relu=' + str(RELU_TYPE) + \
             '_wi=' + str(WEIGHT_INITIALIZER_TYPE) + \
             '_wd=' + str(WEIGHT_DECAY) + \
             '_opt=' + str(OPTIMIZER) + \
             '/date=' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')).replace(' ', '/') + '/'
SESSION_PATH = "./logs/"
VERSION_PATH = SESSION_PATH + '4k_white_gamma/'
SUMMARY_PATH = VERSION_PATH + MODEL_NAME + 'summaries/'
SAVE_PATH = VERSION_PATH + MODEL_NAME + 'saves/'
CODE_COPY_PATH = VERSION_PATH + MODEL_NAME + 'code/'

# Seed initialization
# np.random.seed(10)

# Some sanity checks
assert NUM_LABELED % NUM_CLASSES == 0  # Make sure we can make balanced labeled set
assert NUM_LABELED_IN_BATCH > 0  # Otherwise TensorFlow might freak out when splitting batches
assert NUM_LABELED_IN_BATCH <= NUM_LABELED_IN_EPOCH
assert NUM_LABELED_IN_BATCH <= BATCH_SIZE

# ----------------------------------------------------------------------------------------------------------------------
"""
Network architecture.
"""


# Function we use to apply the decoder namespaces for encoder values
def enhance_for_decoder(dict_encoder):
    dict_complete = {}
    for key, val in dict_encoder.items():
        dict_complete[key] = val
        decoder_key = key[:-5] + 'u'
        dict_complete[decoder_key] = val
    return dict_complete


# The types of our layers, has to include 'l0': 'input'
# - layer types: 'input', 'conv', 'max_pool', 'avg_pool', 'dense'
layer_types = {
     'l0': 'input',
     'l1': 'conv',
     'l2': 'conv',
     'l3': 'conv',
     'l4': 'max_pool',
     'l5': 'conv',
     'l6': 'conv',
     'l7': 'conv',
     'l8': 'max_pool',
     'l9': 'conv',
    'l10': 'conv',
    'l11': 'conv',
    'l12': 'avg_pool',
}

# Number of encoder layers (input does not count)
L = len(layer_types) - 1

# Kernel shapes for conv and deconv layers in format (height, width, channels_in, channels_out)
kernel_shapes = enhance_for_decoder({
    'model/l1/z_pre': [3, 3, 3, 96],
    'model/l2/z_pre': [3, 3, 96, 96],
    'model/l3/z_pre': [3, 3, 96, 96],
    'model/l5/z_pre': [3, 3, 96, 192],
    'model/l6/z_pre': [3, 3, 192, 192],
    'model/l7/z_pre': [3, 3, 192, 192],
    'model/l9/z_pre': [3, 3, 192, 192],
    'model/l10/z_pre': [3, 3, 192, 192],
    'model/l11/z_pre': [3, 3, 192, 10],
})

# Paddings for conv and deconv layers
paddings = enhance_for_decoder({
    'model/l1/z_pre': 'SAME',
    'model/l2/z_pre': 'VALID',
    'model/l3/z_pre': 'VALID',
    'model/l5/z_pre': 'SAME',
    'model/l6/z_pre': 'VALID',
    'model/l7/z_pre': 'SAME',
    'model/l9/z_pre': 'SAME',
    'model/l10/z_pre': 'SAME',
    'model/l11/z_pre': 'SAME',
})

# Output sizes of the layers (we need this for batchnorm initialization)
S = (96, 96, 96, 96, 192, 192, 192, 192, 192, 192, 10, 10)  # TODO Should be automatically calculated at some point

# Factor by which to multiply the reconstruction cost for each layer
# - all 0.0 for supervised-only; last 1.0 and all others 0.0 for Gamma model
# - we have L+1 entries since entry 0 is for denoising input
denoising_costs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]

# ----------------------------------------------------------------------------------------------------------------------
"""
Load data.
"""

# Downloaded MNIST from http://deeplearning.net/tutorial/gettingstarted.html
with open('./data/50k_labels_white.pkl', 'rb') as f:
    unpickler = pickle._Unpickler(f)
    unpickler.encoding = 'latin1'  # need this bc of some Python3 problem
    d = unpickler.load()
    data_tr = d['train_data']
    labels_tr = d['train_labels']
    data_te = d['test_data']
    labels_te = d['test_labels']
    del d


# Function for shuffling data (and possibly labels in unison)
def shuffle_data(data, labels=None):
    perm = np.random.permutation(data.shape[0])
    shuffled_data = data[perm]
    shuffled_labels = None if labels is None else labels[perm]
    return shuffled_data, shuffled_labels


# Initial shuffle of all data
data_tr, labels_tr = shuffle_data(data_tr, labels_tr)
data_te, labels_te = shuffle_data(data_te, labels_te)

# Get all training samples as unlabeled samples and shuffle (bc y not?)
unlabeled = data_tr
unlabeled, _ = shuffle_data(unlabeled)


# Function for creating a class balanced subset of a labeled data set
def make_balanced_labeled_set(data, labels):
    data_balanced = np.zeros((NUM_LABELED, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    labels_balanced = np.zeros((NUM_LABELED, NUM_CLASSES))
    per_class = int(NUM_LABELED / NUM_CLASSES)
    for i, l in enumerate(range(NUM_CLASSES)):
        w = np.where(np.argmax(labels, axis=1) == l)[0]
        idx = np.random.choice(w, size=per_class)
        data_balanced[i*per_class:(i+1)*per_class] = data[idx]
        labels_balanced[i * per_class:(i + 1) * per_class] = labels[idx]
    return shuffle_data(data_balanced, labels_balanced)

# Make class balanced training set
data_tr, labels_tr = make_balanced_labeled_set(data_tr, labels_tr)
# counts = [0] * 10 TODO Delete

# Check if everything's formatted nicely
print("     Train data:", data_tr.shape, labels_tr.shape)
print("      Test data:", data_te.shape, labels_te.shape)
print(" Unlabeled data:", unlabeled.shape)

# Functions for splitting and joining TF batches via labeled/unlabeled (extra functions for flat data from FC layers)
join = lambda l, u: tf.concat([l, u], 0)
get_labeled = lambda x: tf.slice(x, [0, 0, 0, 0], [NUM_LABELED_IN_BATCH, -1, -1, -1],
                                 name='slice_labeled') if x is not None else x
get_unlabeled = lambda x: tf.slice(x, [NUM_LABELED_IN_BATCH, 0, 0, 0], [-1, -1, -1, -1],
                                   name='slice_unlabeled') if x is not None else x
split_lu = lambda x: (get_labeled(x), get_unlabeled(x))
get_labeled_flat = lambda x: tf.slice(x, [0, 0], [NUM_LABELED_IN_BATCH, -1],
                                      name='slice_labeled_flat') if x is not None else x
get_unlabeled_flat = lambda x: tf.slice(x, [NUM_LABELED_IN_BATCH, 0], [-1, -1],
                                        name='slice_unlabeled_flat') if x is not None else x
split_lu_flat = lambda x: (get_labeled_flat(x), get_unlabeled_flat(x))

# ----------------------------------------------------------------------------------------------------------------------
"""
Placeholder variables.
"""

with tf.variable_scope('ph'):
    # Holds the input data to the TF model, first the labeled examples, then the unlabeled
    data_ph = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], name='data_ph')
    # Holds the labels to the labeled examples in data_ph
    labels_ph = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='labels_ph')
    # Flag checked by batch normalization
    is_training_ph = tf.placeholder(tf.bool, name='is_training_ph')

# ----------------------------------------------------------------------------------------------------------------------
"""
Layer functions.
"""

# To calculate the moving averages of mean and variance
ewma = tf.train.ExponentialMovingAverage(decay=0.99)
# Stores the updates to be made to average mean and variance
bn_assigns = []


# Function for batch normalization with either passed mean, variance or on single batch statistics
def batch_normalization(batch, layer_type, mean=None, var=None):
    if mean is None or var is None:
        if layer_type == 'dense':
            mean, var = tf.nn.moments(batch, axes=[0])
        else:
            mean, var = tf.nn.moments(batch, axes=[0, 1, 2])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


# Running means and variances of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[s]), trainable=False) for s in S]
running_var = [tf.Variable(tf.constant(1.0, shape=[s]), trainable=False) for s in S]


# Function to apply batch normalization and update statistics, only used for clean labeled data
def update_batch_normalization(batch, l, layer_type):
    if layer_type == 'dense':
        mean, var = tf.nn.moments(batch, axes=[0])
    else:
        mean, var = tf.nn.moments(batch, axes=[0, 1, 2])
    assign_mean = running_mean[l - 1].assign(mean)
    assign_var = running_var[l - 1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l - 1], running_var[l - 1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)


# Function for adding batch normalization beta parameter
def add_beta(data):
    own_beta = tf.get_variable('own_beta', shape=data.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return data + own_beta


# Function for scaling by batch normalization gamma parameter
def apply_gamma(data):
    own_gamma = tf.get_variable('own_gamma', shape=data.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    return data * own_gamma


# Function adding Gaussian noise in encoder
def noise(data):
    new_noise = tf.random_normal(shape=tf.shape(data), mean=0.0, stddev=NOISE_STD, dtype=tf.float32)
    result = tf.add(data, new_noise)
    result.set_shape(data.get_shape())
    return result


# Function for convolutional layer in encoder
def conv(data, strides=(1, 1, 1, 1)):
    if WEIGHT_INITIALIZER_TYPE == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer()
    elif WEIGHT_INITIALIZER_TYPE == 'he':
        initializer = tf.contrib.layers.variance_scaling_initializer()
    elif WEIGHT_INITIALIZER_TYPE == 'default':
        initializer = None
    else:
        raise Exception("Unknown weight initializer type!")
    weights = tf.get_variable("weights", kernel_shapes[tf.get_variable_scope().name],
                              initializer=initializer,
                              regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))
    return tf.nn.conv2d(data, weights, strides=strides, padding=paddings[tf.get_variable_scope().name])


# Function for 2x2-max pooling in encoder
def max_pool(data, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID'):
    return tf.nn.max_pool(data, ksize=ksize, strides=strides, padding=padding)


# Function for global(!) average pooling in encoder
def avg_pool(data):
    return tf.nn.avg_pool(data, ksize=(1, data.get_shape().as_list()[1], data.get_shape().as_list()[2], 1),
                          strides=(1, 1, 1, 1), padding='VALID')


# Function for deconvolution in decoder
def deconv(data, output_shape, strides=(1, 1, 1, 1)):
    if WEIGHT_INITIALIZER_TYPE == 'xavier':
        intializer = tf.contrib.layers.xavier_initializer()
    elif WEIGHT_INITIALIZER_TYPE == 'he':
        intializer = tf.contrib.layers.variance_scaling_initializer()
    elif WEIGHT_INITIALIZER_TYPE == 'default':
        initializer = None
    else:
        raise Exception("Unknown weight initializer type!")
    weights = tf.get_variable("weights", kernel_shapes[tf.get_variable_scope().name],
                              initializer=initializer,
                              regularizer=tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY))
    return tf.nn.conv2d_transpose(data, weights, output_shape=output_shape, strides=strides,
                                  padding=paddings[tf.get_variable_scope().name])


# Function for reversing pooling in decoder (upsampling with bilinear interpolation)
def unpool(data, scale):
    new_height = int(round(data.get_shape().as_list()[1] * scale))
    new_width = int(round(data.get_shape().as_list()[2] * scale))
    return tf.image.resize_images(data, [new_height, new_width])


# Function for applying different kinds of ReLU activations
def relu(data):
    if RELU_TYPE == 'prelu':
        alpha = tf.get_variable("alpha", data.get_shape()[-1], initializer=tf.constant_initializer(0.1))
        pos = tf.nn.relu(data)
        neg = alpha * (data - abs(data)) * 0.5
        return pos + neg
    elif RELU_TYPE == 'relu':
        return tf.nn.relu(data)
    else:
        raise Exception("Unknown ReLU type!")


# ----------------------------------------------------------------------------------------------------------------------
"""
Ladder network combinator function.
"""


def g_m(u):
    a1 = tf.get_variable('a1', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a2 = tf.get_variable('a2', shape=u.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    a3 = tf.get_variable('a3', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a4 = tf.get_variable('a4', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a5 = tf.get_variable('a5', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5


def g_v(u):
    a6 = tf.get_variable('a6', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a7 = tf.get_variable('a7', shape=u.get_shape()[-1], initializer=tf.constant_initializer(1.0))
    a8 = tf.get_variable('a8', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a9 = tf.get_variable('a9', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    a10 = tf.get_variable('a10', shape=u.get_shape()[-1], initializer=tf.constant_initializer(0.0))
    return a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10


# The combinator function described in the paper, initial values from https://github.com/CuriousAI/ladder/
def g(z_crt, u):
    m = g_m(u)
    return (z_crt - m) * g_v(u) + m


# ----------------------------------------------------------------------------------------------------------------------
"""
Ladder network encoder.
"""

# There are lot of layer intermediate results to be saved
encoder_vals = {}


# The implementation of the encoder path
# - 'corrupted' induces whether to compute the noisy path or the clean path (for training well always need both)
# - all the funny variable names are from Algorithm 1 in the paper
def encoder(data, corrupted):
    # This helps with TF variable scope stuff
    make_key = lambda key_without_suffix, l: key_without_suffix + str(l) + ('_crt' if corrupted else '_cln')

    print("\nCorrupted encoder:") if corrupted else print("\nClean encoder:")
    h = noise(data) if corrupted else data  # Take (noisy) input data as first h
    h.set_shape([BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])  # We don't have to set this, it's just a hint to TF
    m, v = tf.nn.moments(h, [0, 1, 2])  # Here we just assume input data to be an image (not float)
    encoder_vals.update({make_key('h', 0): h, make_key('z', 0): h,
                         make_key('m', 0): m, make_key('v', 0): v})
    print("\t", make_key('h', 0), "shape:", h.shape)

    # Iterate over all layers
    for l in range(1, L + 1):

        scope_name = 'l' + str(l)
        h_prev = encoder_vals[make_key('h', l - 1)]  # Get output from previous layer

        with tf.variable_scope(scope_name) as scope:

            layer_type = layer_types[scope_name]

            with tf.variable_scope('z_pre', reuse=not corrupted):  # Forces us to compute corrupted encoder first!
                if layer_type == 'conv':
                    z_pre = conv(h_prev)
                elif layer_type == 'max_pool':
                    z_pre = max_pool(h_prev)
                elif layer_type == 'avg_pool':
                    z_pre = avg_pool(h_prev)
                elif layer_type == 'dense':
                    flat = tf.reshape(h_prev, (-1, 10))  # TODO This 10 is hardcoded now, should be inferred
                    z_pre = tf.layers.dense(flat, 10, use_bias=False, name='dense')
                else:
                    raise Exception("Unknown layer type!")

            # Here we split the batch into labeled and unlabeled examples
            z_pre_l, z_pre_u = split_lu_flat(z_pre) if layer_type == 'dense' else split_lu(z_pre)
            # These moments could as well be calculated by the batch normalization, but we want to save them later on
            m, v = tf.nn.moments(z_pre_u, [0]) if layer_type == 'dense' else tf.nn.moments(z_pre_u, [0, 1, 2])

            # Batch normalization to be used during training
            # - updates statistics only for clean labeled data!
            # - does not use statistics for unlabeled data
            def training_batch_norm():
                if corrupted:
                    with tf.variable_scope('z_crt'):
                        z = noise(join(batch_normalization(z_pre_l, layer_type),
                                       batch_normalization(z_pre_u, layer_type, m, v)))
                else:
                    with tf.variable_scope('z_cln'):
                        z = join(update_batch_normalization(z_pre_l, l, layer_type),
                                 batch_normalization(z_pre_u, layer_type, m, v))
                return z

            # Batch normalization to be used during evaluation, uses statistics collected during training
            def eval_batch_norm():
                mean = ewma.average(running_mean[l - 1])
                var = ewma.average(running_var[l - 1])
                if layer_type == 'dense':
                    m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
                else:
                    m_l, v_l = tf.nn.moments(z_pre_l, axes=[0, 1, 2])
                z = join(batch_normalization(z_pre_l, layer_type, m_l, v_l),
                         batch_normalization(z_pre_u, layer_type, mean, var))
                return z

            z = tf.cond(is_training_ph, training_batch_norm, eval_batch_norm)

            # Apply ReLU, beta, gamma depending on layer type
            # - no ReLU for dense, since is last layer before softmax (applied outside of encoder function)
            # - no beta for dense, because the paper says this helps with overfitting
            if layer_type == 'conv':
                with tf.variable_scope('h', reuse=not corrupted):
                    h = relu(add_beta(z))
            elif layer_type == 'dense':  # TODO this is also model specific hardcoded
                with tf.variable_scope('h', reuse=not corrupted):
                    h = apply_gamma(z)
            else:
                with tf.variable_scope('h', reuse=not corrupted):
                    h = apply_gamma(add_beta(z))

            # Save some stuff
            encoder_vals.update({make_key('z', l): z, make_key('h', l): h,
                                 make_key('m', l): m, make_key('v', l): v})
            print("\t", make_key('h', l), "shape:", h.shape)

    # Return logits, softmax applied afterwards
    return tf.reshape(encoder_vals[make_key('h', L)], (BATCH_SIZE, NUM_CLASSES))


# ----------------------------------------------------------------------------------------------------------------------
"""
Ladder network decoder.
"""


# The implementation of the decoder path
def decoder():
    z_est = {}  # Stores the estimates for denoised z
    d_cost = []  # Stores the denoising costs of all layers

    # Check how many layers have to be denoised according to denoising_costs (premature optimization?!)
    end_idx = L
    for i in range(L, -1, -1):
        if denoising_costs[i] != 0.0:
            end_idx = i - 1
    if end_idx == L:
        print("\nDecoder will not be used, training is supervised-only!")
    else:
        print("\nDenoising down to layer " + str(end_idx + 1) + "!")
        print("\nDecoder:")

    # Iterate over the specified decoder layers
    for l in range(L, end_idx, -1):

        # Get previous corrupted z value
        z_crt = get_unlabeled_flat(encoder_vals['z' + str(l) + '_crt']) if layer_types['l' + str(l)] == 'dense' \
            else get_unlabeled(encoder_vals['z' + str(l) + '_crt'])

        # Get corresponding clean z value (denoising target)
        z_cln = get_unlabeled_flat(encoder_vals['z' + str(l) + '_cln']) if layer_types['l' + str(l)] == 'dense' \
            else get_unlabeled(encoder_vals['z' + str(l) + '_cln'])

        m, v = encoder_vals['m' + str(l) + '_crt'], encoder_vals['v' + str(l) + '_crt']

        with tf.variable_scope('l' + str(l + 1)) as scope:

            # If uppermost layer, just get corresponding output
            if l == L:
                u = get_unlabeled_flat(encoder_vals['h' + str(l) + '_crt']) if layer_types['l' + str(
                    l)] == 'dense' else get_unlabeled(encoder_vals['h' + str(l) + '_crt'])
            # Else, revert layer effect
            else:
                with tf.variable_scope('u'):
                    if layer_types['l' + str(l + 1)] == 'conv':
                        shape = get_unlabeled(encoder_vals['h' + str(l) + '_crt']).get_shape().as_list()
                        shape = [BATCH_SIZE - NUM_LABELED_IN_BATCH] + shape[1:4]
                        u = deconv(z_est[l + 1], output_shape=shape)
                    elif layer_types['l' + str(l + 1)] == 'max_pool':
                        u = unpool(z_est[l + 1], 2.0)
                    elif layer_types['l' + str(l + 1)] == 'avg_pool':
                        u = unpool(z_est[l + 1], 6.0)
                    elif layer_types['l' + str(l + 1)] == 'dense':
                        flat = tf.layers.dense(z_est[l + 1], 10, use_bias=False, name='dense')
                        shape = get_unlabeled(encoder_vals['h' + str(l) + '_crt']).get_shape().as_list()
                        shape = [-1] + shape[1:4]
                        u = tf.reshape(flat, shape)  # weird hack happening here
                    else:
                        raise Exception("Unknown layer type!")

            u = batch_normalization(u, layer_types['l' + str(l)])  # again, nothing updated here
            z_est[l] = g(z_crt, u)
            print("\t", 'z' + str(l) + '_est shape:', z_est[l].get_shape())
            z_est_bn = (z_est[l] - m) / tf.sqrt(v + 1e-10)
            # Append the weighted cost of this layer to d_cost
            d_cost.append(
                (tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z_cln), 1)) / S[l - 1]) * denoising_costs[l])

    # Return the array of all weighted layer costs
    return tf.reduce_sum(d_cost)


# ----------------------------------------------------------------------------------------------------------------------
"""
Prediction and loss.
"""

with tf.variable_scope('model'):
    # Corrupted and clean model output
    encoder_output_crt = tf.identity(encoder(data_ph, corrupted=True), name='encoder_output_crt')
    encoder_output_cln = tf.identity(encoder(data_ph, corrupted=False), name='encoder_output_cln')

    # Corrupted and clean logits from model output (just slicing)
    logits_crt = tf.identity(encoder_output_crt[:NUM_LABELED_IN_BATCH, :], name='logits_crt')
    logits_cln = tf.identity(encoder_output_cln[:NUM_LABELED_IN_BATCH, :], name='logits_cln')

    # Get array of decoder costs
    decoder_costs = tf.identity(decoder(), name='decoder_costs')

with tf.variable_scope('loss'):
    # Softmax for corrupted and clean (this is just for logging, do not use these to compute cross entropy error!)
    softmax_crt = tf.nn.softmax(logits_crt, name='softmax_crt')
    softmax_cln = tf.nn.softmax(logits_cln, name='softmax_cln')

    # Softmax cross entropy losses, corrupted and clean
    softmax_loss_crt = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_crt, labels=tf.cast(labels_ph, dtype=tf.int32)),
        name='softmax_loss_crt')
    softmax_loss_cln = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_cln, labels=tf.cast(labels_ph, dtype=tf.int32)),
        name='softmax_loss_cln')

    # Decoder loss
    decoder_loss = tf.reduce_sum(decoder_costs, name='decoder_loss')

    # Weight decay loss
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    weight_decay_loss = tf.reduce_sum(reg_losses, name='weight_decay_loss')

    # Total loss (Softmax, weight decay, and decoder loss weighted equally)
    total_loss_crt = tf.add(softmax_loss_crt, weight_decay_loss + decoder_loss, name='total_loss_crt')
    total_loss_cln = tf.add(softmax_loss_cln, weight_decay_loss + decoder_loss, name='total_loss_cln')

# ----------------------------------------------------------------------------------------------------------------------
"""
Optimizer.
"""

step = tf.Variable(0.0, trainable=False, dtype=tf.float32)

if OPTIMIZER == 'adam':  # Adam optimizer with linear learning rate decay after fixed time
    lr_const = tf.constant(INITIAL_LEARNING_RATE, dtype=tf.float32)
    pol_decay_rate = tf.train.polynomial_decay(learning_rate=(1.0 / (1.0 - LR_DECAY_FIRST)) * INITIAL_LEARNING_RATE,
                                               global_step=step,
                                               decay_steps=NUM_TOTAL_ITERS,
                                               end_learning_rate=0.000,
                                               power=1.0)
    cond = tf.less(step, NUM_TOTAL_ITERS * LR_DECAY_FIRST)
    adam_lr = tf.cond(cond, lambda: lr_const, lambda: pol_decay_rate)
    tf.summary.scalar('adam_lr', adam_lr)
    tf.summary.scalar('pol_decay_rate', pol_decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=0.9)
else:  # AdaDelta as default
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=INITIAL_LEARNING_RATE, rho=0.95)

# This is needed for updating mean/variance batchnorm statistics
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(total_loss_crt, global_step=step, name='train_op')

# ----------------------------------------------------------------------------------------------------------------------
"""
Accuracy.
"""

with tf.variable_scope('acc'):
    accuracy_crt = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits_crt, 1), tf.argmax(labels_ph, 1), name='correct_pred_crt'), tf.float32),
        name='accuracy_crt')
    accuracy_cln = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits_cln, 1), tf.argmax(labels_ph, 1), name='correct_pred_cln'), tf.float32),
        name='accuracy_cln')

# ----------------------------------------------------------------------------------------------------------------------
"""
Summaries.
"""

tf.summary.scalar("loss__softmax_loss_crt", softmax_loss_crt)
tf.summary.scalar("loss__softmax_loss_cln", softmax_loss_cln)
tf.summary.scalar("loss__weight_decay_loss", weight_decay_loss)
tf.summary.scalar("loss__decoder_loss", decoder_loss)
tf.summary.scalar("loss__total_loss_crt", total_loss_crt)
tf.summary.scalar("loss__total_loss_cln", total_loss_cln)

tf.summary.scalar("acc__accuracy_crt", accuracy_crt)
tf.summary.scalar("acc__accuracy_cln", accuracy_cln)

summary_op = tf.summary.merge_all()

# ----------------------------------------------------------------------------------------------------------------------
"""
Run training and evaluation.
"""


# Function for evaluation of a whole dataset
def eval_set(data, labels):
    acc_crt = 0.0
    acc_cln = 0.0
    lss_crt = 0.0
    lss_cln = 0.0
    eval_batch_size = NUM_LABELED_IN_BATCH
    assert data.shape[0] % eval_batch_size == 0
    offsets = [i * eval_batch_size for i in range(int(data.shape[0] / eval_batch_size))]
    for i in offsets:
        data_batch = data[i:i + eval_batch_size, :, :, :]
        data_batch = np.concatenate((data_batch, np.zeros(data_batch.shape)))  # TODO Not exactly efficient
        labels_batch = labels[i:i + eval_batch_size, :]
        acc_tmp_crt, acc_tmp_cln, lss_tmp_crt, lss_tmp_cln = sess.run(
            [accuracy_crt, accuracy_cln, total_loss_crt, total_loss_cln],
            feed_dict={data_ph: data_batch,
                       labels_ph: labels_batch,
                       is_training_ph: False})
        acc_crt += acc_tmp_crt
        acc_cln += acc_tmp_cln
        lss_crt += lss_tmp_crt
        lss_cln += lss_tmp_cln
    acc_crt /= len(offsets)
    acc_cln /= len(offsets)
    lss_crt /= len(offsets)
    lss_cln /= len(offsets)
    return acc_crt, acc_cln, lss_crt, lss_cln


# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # Writers for logging progress on train and test set
    train_writer = tf.summary.FileWriter(SUMMARY_PATH + "train", graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(SUMMARY_PATH + "test", graph=tf.get_default_graph())

    # Saver for saving TF model
    saver = tf.train.Saver(max_to_keep=50)

    # Variable initialization
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # When everything is just about to start running, save a copy of the running code, just in case
    if not os.path.exists(CODE_COPY_PATH):
        os.makedirs(CODE_COPY_PATH)
    copyfile("./" + FILE_NAME, CODE_COPY_PATH + FILE_NAME)
    print("Saved copy of this code!")

    # Training loop
    offset_labeled, offset_unlabeled = 0, 0
    for step in tqdm(range(NUM_TOTAL_ITERS)):

        # Manage offsets for train and test sets, so we can shuffle data when we're done with one epoch
        num_labeled = NUM_LABELED_IN_BATCH
        num_unlabeled = BATCH_SIZE - num_labeled
        offset_labeled = (offset_labeled + num_labeled) % (data_tr.shape[0])
        if offset_labeled < num_labeled:
            data_tr, labels_tr = shuffle_data(data_tr, labels_tr)
            offset_labeled = 0
        offset_unlabeled = (offset_unlabeled + num_unlabeled) % (unlabeled.shape[0])
        if offset_unlabeled < num_unlabeled:
            unlabeled, _ = shuffle_data(unlabeled)
            offset_unlabeled = 0

        # Get batch with both labeled and unlabeled examples
        data_tr_batch = np.concatenate((data_tr[offset_labeled:(offset_labeled + num_labeled), :, :, :],
                                        unlabeled[offset_unlabeled:(offset_unlabeled + num_unlabeled), :, :, :]))
        labels_tr_batch = labels_tr[offset_labeled:(offset_labeled + num_labeled), :]

        # Run optimization op (backprop)
        _, batch_summary, acc_tr_crt, acc_tr_cln, lss_tr_crt, lss_tr_cln = sess.run(
            [train_op, summary_op, accuracy_crt, accuracy_cln, total_loss_crt, total_loss_cln],
            feed_dict={data_ph: data_tr_batch,
                       labels_ph: labels_tr_batch,
                       is_training_ph: True})

        # Log training summary
        if step % TRAIN_SUMMARY_INTERVAL_IN_STEPS == 0:
            train_writer.add_summary(batch_summary, step)

        # Evaluate test set and log test summary
        if step % EVAL_SUMMARY_INTERVAL_IN_STEPS == 0:
            acc_te_crt, acc_te_cln, lss_te_crt, lss_te_cln = eval_set(data_te, labels_te)
            print('\nTrain: acc_crt:', "%.4f" % acc_tr_crt, 'acc_cln:', "%.4f" % acc_tr_cln, ' lss_crt:',
                  "%.4f" % lss_tr_crt, ' lss_cln:', "%.4f" % lss_tr_cln)
            print(' Test: acc_crt:', "%.4f" % acc_te_crt, 'acc_cln:', "%.4f" % acc_te_cln, ' lss_crt:',
                  "%.4f" % lss_te_crt, ' lss_cln:', "%.4f" % lss_te_cln)
            test_summary = tf.Summary(value=[tf.Summary.Value(tag="acc__accuracy", simple_value=acc_te_cln),
                                             ])
            test_writer.add_summary(test_summary, step)

        # Save model
        if step > ITERS_PER_EPOCHE * EPOCHS_BEFORE_SAVING and step % (
                    ITERS_PER_EPOCHE * SAVING_INTERVAL_IN_EPOCHS) == 0:
            print("Saving model...")
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            time_str = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            saver.save(sess, save_path=SAVE_PATH + time_str + ".ckpt")

    print("\nOptimization finished!")

    print("Saving final model...")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    saver.save(sess, save_path=SAVE_PATH + "final.ckpt")

    print("\nFinal evaluation:")
    acc_te_crt, acc_te_cln, lss_te_crt, lss_te_cln = eval_set(data_te, labels_te)
    print("Model:", MODEL_NAME)
    print('\nTrain: acc_crt:', "%.4f" % acc_tr_crt, 'acc_cln:', "%.4f" % acc_tr_cln, ' lss_crt:',
          "%.4f" % lss_tr_crt, ' lss_cln:', "%.4f" % lss_tr_cln)
    print(' Test: acc_crt:', "%.4f" % acc_te_crt, 'acc_cln:', "%.4f" % acc_te_cln, ' lss_crt:',
          "%.4f" % lss_te_crt, ' lss_cln:', "%.4f" % lss_te_cln)
