""" Models definition for convolutional nets architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax
from jax.tree_util import Partial


def conv_3_2(sizes, number_classes, dim=2):
    """ Convnet with 3 convolutional layers followed by 2 fully-connected
    """
    def act():
        return jax.nn.relu
    if dim==1:
        conv_fn = hk.Conv1D
    elif dim==2:
        conv_fn = hk.Conv2D
    else:
        raise Exception("Convnet dimension restricted to 1 or 2")

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2*sizes[0], 4*sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=(2, 2), strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=(4, 4), strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 3), act]
    layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3), act]
    layer_4 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[3]), act]
    layer_5 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5]


def conv_4_2(sizes, number_classes, dim=2):
    """ Convnet with 4 convolutional layers followed by 2 fully-connected
    """
    def act():
        return jax.nn.relu
    if dim==1:
        conv_fn = hk.Conv1D
    elif dim==2:
        conv_fn = hk.Conv2D
    else:
        raise Exception("Convnet dimension restricted to 1 or 2")

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2*sizes[0], 4*sizes[0], 4*sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=(2, 2), strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=(4, 4), strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 3), act]
    layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3), act]
    layer_4 = [max_pool, Partial(conv_fn, sizes[3], 3), act]
    layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4]), act]
    layer_6 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6]


def conv_6_2(sizes, number_classes, dim=2):
    """ Convnet with 6 convolutional layers followed by 2 fully-connected
    """
    def act():
        return jax.nn.relu
    if dim==1:
        conv_fn = hk.Conv1D
    elif dim==2:
        conv_fn = hk.Conv2D
    else:
        raise Exception("Convnet dimension restricted to 1 or 2")

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], sizes[0], 2*sizes[0], 2*sizes[0], 4*sizes[0], 4*sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=(2, 2), strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=(4, 4), strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 3), act]
    layer_2 = [Partial(conv_fn, sizes[1], 3), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3), act]
    layer_4 = [Partial(conv_fn, sizes[3], 3), act]
    layer_5 = [max_pool, Partial(conv_fn, sizes[4], 3), act]
    layer_6 = [max_pool, Partial(conv_fn, sizes[5], 3), act]
    layer_7 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[6]), act]
    layer_8 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8]
