""" Models definition for convolutional nets architecture. Defined fitting requirements of repo"""
import haiku as hk
from jax.tree_util import Partial
from jax.nn import relu

from models.bn_base_unit import Conv_BN, Linear_BN


def conv_3_2(sizes, number_classes, dim=2, activation_fn=relu):
    """ Convnet with 3 convolutional layers followed by 2 fully-connected
    """

    def act():
        return activation_fn

    if dim == 1:
        conv_fn = hk.Conv1D
    elif dim == 2:
        conv_fn = hk.Conv2D
    else:
        raise Exception("Convnet dimension restricted to 1 or 2")

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 3), act]
    layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3), act]
    layer_4 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[3]), act]
    layer_5 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5],


def conv_3_2_bn(sizes, number_classes, activation_fn=relu):
    """ Convnet with 3 convolutional layers followed by 2 fully-connected, with BN added after
    every layer apart from the final one.
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    train_layer_1 = [Partial(Conv_BN, True, sizes[0], 3), act]
    train_layer_2 = [max_pool, Partial(Conv_BN, True, sizes[1], 3), act]
    train_layer_3 = [max_pool, Partial(Conv_BN, True, sizes[2], 3), act]
    train_layer_4 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, True, sizes[3]), act]
    layer_5 = [Partial(hk.Linear, number_classes)]

    test_layer_1 = [Partial(Conv_BN, False, sizes[0], 3), act]
    test_layer_2 = [max_pool, Partial(Conv_BN, False, sizes[1], 3), act]
    test_layer_3 = [max_pool, Partial(Conv_BN, False, sizes[2], 3), act]
    test_layer_4 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, False, sizes[3]), act]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, layer_5], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, layer_5]


def conv_4_2(sizes, number_classes, dim=2, activation_fn=relu):
    """ Convnet with 4 convolutional layers followed by 2 fully-connected
    """

    def act():
        return activation_fn

    if dim == 1:
        conv_fn = hk.Conv1D
    elif dim == 2:
        conv_fn = hk.Conv2D
    else:
        raise Exception("Convnet dimension restricted to 1 or 2")

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 3), act]
    layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3), act]
    layer_4 = [max_pool, Partial(conv_fn, sizes[3], 3), act]
    layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4]), act]
    layer_6 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6],


def conv_4_2_bn(sizes, number_classes, activation_fn=relu):
    """ Convnet with 4 convolutional layers followed by 2 fully-connected, with BN added after
    every layer apart from the final one.
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    train_layer_1 = [Partial(Conv_BN, True, sizes[0], 3), act]
    train_layer_2 = [max_pool, Partial(Conv_BN, True, sizes[1], 3), act]
    train_layer_3 = [max_pool, Partial(Conv_BN, True, sizes[2], 3), act]
    train_layer_4 = [max_pool, Partial(Conv_BN, True, sizes[3], 3), act]
    train_layer_5 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, True, sizes[4]), act]
    layer_6 = [Partial(hk.Linear, number_classes)]

    test_layer_1 = [Partial(Conv_BN, False, sizes[0], 3), act]
    test_layer_2 = [max_pool, Partial(Conv_BN, False, sizes[1], 3), act]
    test_layer_3 = [max_pool, Partial(Conv_BN, False, sizes[2], 3), act]
    test_layer_4 = [max_pool, Partial(Conv_BN, False, sizes[3], 3), act]
    test_layer_5 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, False, sizes[4]), act]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, train_layer_5, layer_6], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, test_layer_5, layer_6]


def conv_6_2(sizes, number_classes, dim=2, activation_fn=relu):
    """ Convnet with 6 convolutional layers followed by 2 fully-connected
    """

    def act():
        return activation_fn

    if dim == 1:
        conv_fn = hk.Conv1D
    elif dim == 2:
        conv_fn = hk.Conv2D
    else:
        raise Exception("Convnet dimension restricted to 1 or 2")

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 3), act]
    layer_2 = [Partial(conv_fn, sizes[1], 3), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3), act]
    layer_4 = [Partial(conv_fn, sizes[3], 3), act]
    layer_5 = [max_pool, Partial(conv_fn, sizes[4], 3), act]
    layer_6 = [max_pool, Partial(conv_fn, sizes[5], 3), act]
    layer_7 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[6]), act]
    layer_8 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8],


def conv_6_2_bn(sizes, number_classes, activation_fn=relu):
    """ Convnet with 6 convolutional layers followed by 2 fully-connected, with BN added after
    every layer apart from the final one.
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    train_layer_1 = [Partial(Conv_BN, True, sizes[0], 3), act]
    train_layer_2 = [Partial(Conv_BN, True, sizes[1], 3), act]
    train_layer_3 = [max_pool, Partial(Conv_BN, True, sizes[2], 3), act]
    train_layer_4 = [Partial(Conv_BN, True, sizes[3], 3), act]
    train_layer_5 = [max_pool, Partial(Conv_BN, True, sizes[4], 3), act]
    train_layer_6 = [max_pool, Partial(Conv_BN, True, sizes[5], 3), act]
    train_layer_7 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, True, sizes[6]), act]
    layer_8 = [Partial(hk.Linear, number_classes)]

    test_layer_1 = [Partial(Conv_BN, False, sizes[0], 3), act]
    test_layer_2 = [Partial(Conv_BN, False, sizes[1], 3), act]
    test_layer_3 = [max_pool, Partial(Conv_BN, False, sizes[2], 3), act]
    test_layer_4 = [Partial(Conv_BN, False, sizes[3], 3), act]
    test_layer_5 = [max_pool, Partial(Conv_BN, False, sizes[4], 3), act]
    test_layer_6 = [max_pool, Partial(Conv_BN, False, sizes[5], 3), act]
    test_layer_7 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, False, sizes[6]), act]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, train_layer_5, train_layer_6, train_layer_7, layer_8], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, test_layer_5, test_layer_6, test_layer_7, layer_8],
