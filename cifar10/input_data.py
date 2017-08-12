# import tensorflow as tf
# from tensorflow.contrib.layers.python.layers import batch_norm
import pickle
import numpy as np
import scipy
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
"""
CIFAR-10 parameters.
"""

image_size = 32
num_labels = 10  # labels 0-9
num_channels = 3  # RGB
channel_depth = 256  # RGB depth 8-bit, [0-255]

# ----------------------------------------------------------------------------------------------------------------------
"""
Reading CIFAR-10 input data and setting up training/test set.
"""

data_pickle_files = [
    './data/cifar-10-batches-py/data_batch_1',
    './data/cifar-10-batches-py/data_batch_2',
    './data/cifar-10-batches-py/data_batch_3',
    './data/cifar-10-batches-py/data_batch_4',
    './data/cifar-10-batches-py/data_batch_5'
]
test_pickle_file = './data/cifar-10-batches-py/test_batch'

print("\nReading data...")

train_data = np.empty([0, image_size * image_size * num_channels], int)
train_labels = np.empty([0], int)
for data_pickle_file in data_pickle_files:
    with open(data_pickle_file, 'rb') as f:
        unpickler = pickle._Unpickler(f)
        unpickler.encoding = 'latin1'  # need this bc of some Python3 problem
        save = unpickler.load()
        train_data = np.append(train_data, save['data'], axis=0)
        train_labels = np.append(train_labels, save['labels'])
        del save  # hint to help gc free up memory

print("\tTrain data   :", train_data.shape)
print("\tTrain labels :", train_labels.shape)

with open(test_pickle_file, 'rb') as f:
    unpickler = pickle._Unpickler(f)
    unpickler.encoding = 'latin1'  # need this bc of some Python3 problem
    save = unpickler.load()
    test_data = save['data']
    test_labels = np.asanyarray(save['labels'])
    del save  # hint to help gc free up memory
print("\tTest data    :", test_data.shape)
print("\tTest labels  :", test_labels.shape)

train_data_unpre = train_data
train_labels_unpre = train_labels
test_data_unpre = test_data
test_labels_unpre = test_labels

# ----------------------------------------------------------------------------------------------------------------------
"""
Use global contrast normalization and ZCA whitening.
"""


def global_contrast_normalize(data, scale, min_divisor=1e-8):
    data = data - data.mean(axis=1)[:, np.newaxis]

    normalizers = np.sqrt(np.sum(data ** 2, axis=1)) / scale
    normalizers[normalizers < min_divisor] = np.float32(1.)

    data /= normalizers[:, np.newaxis]

    return data

print("\nApplying global contrast normalization...")
norm_scale = 55.0
train_data = global_contrast_normalize(train_data, scale=norm_scale)
test_data = global_contrast_normalize(test_data, scale=norm_scale)
print("\tTrain data   :", train_data.shape)
print("\tTest data    :", test_data.shape)


def compute_zca_transform(data, filter_bias=0.1):
    # n, m = data.shape
    # meanX = np.mean(data, axis=0)
    #
    # bias = filter_bias * scipy.sparse.identity(m, 'float32')
    # cov = np.cov(data, rowvar=0, bias=1) + bias
    # eigs, eigv = scipy.linalg.eigh(cov)
    #
    # assert not np.isnan(eigs).any()
    # assert not np.isnan(eigv).any()
    # assert eigs.min() > 0
    #
    # sqrt_eigs = np.sqrt(eigs)
    # P = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)
    # assert not np.isnan(P).any()
    # P = np.float32(P)
    #
    # return meanX, P
    meanX = np.mean(data, 0)

    covX = np.cov(data.T)

    D, E = np.linalg.eigh(covX + filter_bias * np.eye(covX.shape[0], covX.shape[1]))

    assert not np.isnan(D).any()
    assert not np.isnan(E).any()
    assert D.min() > 0

    D = D ** -0.5

    W = np.dot(E, np.dot(np.diag(D), E.T))
    return meanX, W


def zca_whiten(train_data, test_data):
    meanX, W = compute_zca_transform(train_data)
    train_w = np.dot(train_data - meanX, W)
    test_w = np.dot(test_data - meanX, W)

    return train_w, test_w

print("\nApplying ZCA whitening...")
train_data, test_data = zca_whiten(train_data, test_data)
print("\tTrain data   :", train_data.shape)
print("\tTest data    :", test_data.shape)

# ----------------------------------------------------------------------------------------------------------------------
"""
Reformat into a TensorFlow-friendly shape:
    - convolutions need the image data formatted as a cube (width by height by #channels)
    - labels as float 1-hot encodings.
"""


def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, num_channels, image_size, image_size)).transpose((0, 2, 3, 1)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


print('\nFormatting data...')
train_data, train_labels = reformat(train_data, train_labels)
test_data, test_labels = reformat(test_data, test_labels)
print('\tTraining set :', train_data.shape, train_labels.shape)
print('\tTest set     :', test_data.shape, test_labels.shape)

train_data_unpre, train_labels_unpre = reformat(train_data_unpre, train_labels_unpre)
test_data_unpre, test_labels_unpre = reformat(test_data_unpre, test_labels_unpre)

# ----------------------------------------------------------------------------------------------------------------------
"""
Make complete training set and balanced 4k training set.
"""

print('\nPickling data...')
print("\tPickling complete data set")
pickle_dict = {'train_data': train_data, 'train_labels': train_labels,
               'test_data': test_data, 'test_labels': test_labels}
pickle.dump(pickle_dict, open("50k_labels_white.pkl", "wb"))

print("\tBalancing 4k data set")
train_data_4k = []
train_labels_4k = []
class_counts = [0] * 10
overall_count = 0
i = 0
while overall_count < 4000 and i < len(train_data):
    cur_class = int(np.argmax(train_labels[i]))
    if class_counts[cur_class] < 400:
        class_counts[cur_class] += 1
        train_data_4k += [train_data[i]]
        train_labels_4k += [train_labels[i]]
        overall_count += 1
    i += 1

train_data_4k = np.array(train_data_4k, dtype=np.float32)
train_labels_4k = np.array(train_labels_4k, dtype=np.float32)


def alternative_shuffle(data, labels):
    perm = np.random.permutation(data.shape[0])
    shuffled_data = data[perm]
    shuffled_labels = labels[perm]
    return shuffled_data, shuffled_labels

train_data_4k, train_labels_4k = alternative_shuffle(train_data_4k, train_labels_4k)

print("\tPickling 4k data set")
pickle_dict = {'train_data': train_data_4k, 'train_labels': train_labels_4k,
               'test_data': test_data, 'test_labels': test_labels}
pickle.dump(pickle_dict, open("4k_labels_white.pickle", "wb"))

print("\tPickling unpreprocessed data set")
pickle_dict = {'train_data': train_data_unpre, 'train_labels': train_labels_unpre,
               'test_data': test_data_unpre, 'test_labels': test_labels_unpre}
pickle.dump(pickle_dict, open("50k_labels.pickle", "wb"))
