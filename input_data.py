import pickle
import numpy as np
import utils as u


def _load_cifar10(pkl_file_path, num_labeled=None):
    with open(pkl_file_path, 'rb') as f:
        unpickler = pickle._Unpickler(f)
        unpickler.encoding = 'latin1'  # need this bc of some Python3 problem
        d = unpickler.load()
        data_tr = d['train_data']
        labels_tr = d['train_labels']
        data_te = d['test_data']
        labels_te = d['test_labels']
        del d

    unlabeled = data_tr
    data_tr, labels_tr = u.make_class_balanced_set(data_tr, labels_tr, num_labeled)

    return data_tr, labels_tr, data_te, labels_te, unlabeled


def _load_mnist(pkl_file_path, num_labeled=None):
    with open(pkl_file_path, 'rb') as f:
        unpickler = pickle._Unpickler(f)
        unpickler.encoding = 'latin1'  # need this bc of some Python3 problem
        train_set, valid_set, test_set = unpickler.load()

    # Extract image data
    data_tr = np.array(train_set[0], dtype=np.float32).reshape((-1, 28, 28, 1))
    data_va = np.array(valid_set[0], dtype=np.float32).reshape((-1, 28, 28, 1))
    data_te = np.array(test_set[0], dtype=np.float32).reshape((-1, 28, 28, 1))

    # Get target data as one-hot encoded
    labels_tr = u.labels_to_one_hot(train_set[1], 10)
    labels_va = u.labels_to_one_hot(valid_set[1], 10)
    labels_te = u.labels_to_one_hot(test_set[1], 10)

    # Combine training and validation data for final training
    data_tr = np.concatenate((data_tr, data_va))
    labels_tr = np.concatenate((labels_tr, labels_va))

    # Get all training samples as unlabeled samples
    unlabeled = data_tr

    # Make class balanced training set
    data_tr, labels_tr = u.make_class_balanced_set(data_tr, labels_tr, num_labeled)

    return data_tr, labels_tr, data_te, labels_te, unlabeled


def load_data(dataset_name, num_labeled):
    if dataset_name == 'cifar10':
        return _load_cifar10('./cifar10/data/50k_labels.pkl', num_labeled)
    elif dataset_name == 'mnist':
        return _load_mnist('./mnist/data/mnist.pkl', num_labeled)
    else:
        print('Unknown dataset!')
        exit(0)