""" Models definition for convolutional nets architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax.nn
from jax.tree_util import Partial
from jax.nn import relu, tanh

from models.bn_base_unit import Base_BN, Conv_BN, Linear_BN
from models.dropout_units import Base_Dropout


def _tanh():
    return lambda x: tanh(2 * x)


def conv_3_2(sizes, number_classes, dim=2, activation_fn=relu, with_bias=True, tanh_head=False):
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

    layer_1 = [Partial(conv_fn, sizes[0], 3, with_bias=with_bias), act]
    layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3, with_bias=with_bias), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3, with_bias=with_bias), act]
    layer_4 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[3], with_bias=with_bias), act]
    if tanh_head:
        layer_5 = [Partial(hk.Linear, number_classes, with_bias=with_bias), _tanh]
    else:
        layer_5 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5],


def conv_3_2_bn(sizes, number_classes, activation_fn=relu, with_bias=True):
    """ Convnet with 3 convolutional layers followed by 2 fully-connected, with BN added after
    every layer apart from the final one.
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    train_layer_1 = [Partial(Conv_BN, True, sizes[0], 3, with_bias=with_bias), act]
    train_layer_2 = [max_pool, Partial(Conv_BN, True, sizes[1], 3, with_bias=with_bias), act]
    train_layer_3 = [max_pool, Partial(Conv_BN, True, sizes[2], 3, with_bias=with_bias), act]
    train_layer_4 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, True, sizes[3], with_bias=with_bias), act]
    layer_5 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [Partial(Conv_BN, False, sizes[0], 3, with_bias=with_bias), act]
    test_layer_2 = [max_pool, Partial(Conv_BN, False, sizes[1], 3, with_bias=with_bias), act]
    test_layer_3 = [max_pool, Partial(Conv_BN, False, sizes[2], 3, with_bias=with_bias), act]
    test_layer_4 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, False, sizes[3], with_bias=with_bias), act]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, layer_5], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, layer_5]


def conv_4_2(sizes, number_classes, dim=2, activation_fn=relu, with_bias=True, tanh_head=False):
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

    layer_1 = [Partial(conv_fn, sizes[0], 3, with_bias=with_bias), act]
    layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3, with_bias=with_bias), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3, with_bias=with_bias), act]
    layer_4 = [max_pool, Partial(conv_fn, sizes[3], 3, with_bias=with_bias), act]
    layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4], with_bias=with_bias), act]
    if tanh_head:
        layer_6 = [Partial(hk.Linear, number_classes, with_bias=with_bias), _tanh]
    else:
        layer_6 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6],


def conv_4_2_dropout(sizes, number_classes, dim=2, activation_fn=relu, dropout_rate=0, with_bias=True):
    """ Dropout version of conv_4_2
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

    train_layer_1 = [Partial(conv_fn, sizes[0], 3, with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    train_layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3, with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    train_layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3, with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    train_layer_4 = [max_pool, Partial(conv_fn, sizes[3], 3, with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    train_layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4], with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    layer_6 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [Partial(conv_fn, sizes[0], 3, with_bias=with_bias), act, Partial(Base_Dropout, 0)]
    test_layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3, with_bias=with_bias), act, Partial(Base_Dropout, 0)]
    test_layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3, with_bias=with_bias), act, Partial(Base_Dropout, 0)]
    test_layer_4 = [max_pool, Partial(conv_fn, sizes[3], 3, with_bias=with_bias), act, Partial(Base_Dropout, 0)]
    test_layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4], with_bias=with_bias), act, Partial(Base_Dropout, 0)]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, train_layer_5, layer_6], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, test_layer_5, layer_6]


def conv_4_2_bn(sizes, number_classes, activation_fn=relu, with_bias=True):
    """ Convnet with 4 convolutional layers followed by 2 fully-connected, with BN added after
    every layer apart from the final one.
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    train_layer_1 = [Partial(hk.Conv2D, sizes[0], 3, with_bias=with_bias), Partial(Base_BN, True), act]
    train_layer_2 = [max_pool, Partial(hk.Conv2D, sizes[1], 3, with_bias=with_bias), Partial(Base_BN, True), act]
    train_layer_3 = [max_pool, Partial(hk.Conv2D, sizes[2], 3, with_bias=with_bias), Partial(Base_BN, True), act]
    train_layer_4 = [max_pool, Partial(hk.Conv2D, sizes[3], 3, with_bias=with_bias), Partial(Base_BN, True), act]
    train_layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4], with_bias=with_bias), Partial(Base_BN, True), act]
    layer_6 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [Partial(hk.Conv2D, sizes[0], 3, with_bias=with_bias), Partial(Base_BN, False), act]
    test_layer_2 = [max_pool, Partial(hk.Conv2D, sizes[1], 3, with_bias=with_bias), Partial(Base_BN, False), act]
    test_layer_3 = [max_pool, Partial(hk.Conv2D, sizes[2], 3, with_bias=with_bias), Partial(Base_BN, False), act]
    test_layer_4 = [max_pool, Partial(hk.Conv2D, sizes[3], 3, with_bias=with_bias), Partial(Base_BN, False), act]
    test_layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4], with_bias=with_bias), Partial(Base_BN, False), act]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, train_layer_5, layer_6], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, test_layer_5, layer_6]


def conv_4_2_ln(sizes, number_classes, activation_fn=relu, with_bias=True):
    """ Convnet with 4 convolutional layers followed by 2 fully-connected, with LayerNorm layers
    added after activation functions
    """

    def act():
        return activation_fn

    def tanh():
        return jax.nn.tanh

    def hard_offset():
        def _apply_hard_offset(x):
            return x-1.5
        return _apply_hard_offset

    conv_fn = hk.Conv2D

    conv_ln = Partial(hk.LayerNorm, axis=[-1, -2, -3], create_scale=False, create_offset=False, param_axis=-1)
    mlp_ln = Partial(hk.LayerNorm, axis=-1, create_scale=False, create_offset=False, param_axis=-1)

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 3, with_bias=with_bias), conv_ln, hard_offset, act]
    layer_2 = [max_pool, Partial(conv_fn, sizes[1], 3, with_bias=with_bias), conv_ln, hard_offset, act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3, with_bias=with_bias), conv_ln, hard_offset, act]
    layer_4 = [max_pool, Partial(conv_fn, sizes[3], 3, with_bias=with_bias), conv_ln, hard_offset, act]
    layer_5 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4], with_bias=with_bias), mlp_ln, hard_offset, act]
    layer_6 = [Partial(hk.Linear, number_classes, with_bias=with_bias), mlp_ln]  #, act]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6],


def conv_6_2(sizes, number_classes, dim=2, activation_fn=relu, with_bias=True):
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

    layer_1 = [Partial(conv_fn, sizes[0], 3, with_bias=with_bias), act]
    layer_2 = [Partial(conv_fn, sizes[1], 3, with_bias=with_bias), act]
    layer_3 = [max_pool, Partial(conv_fn, sizes[2], 3, with_bias=with_bias), act]
    layer_4 = [Partial(conv_fn, sizes[3], 3, with_bias=with_bias), act]
    layer_5 = [max_pool, Partial(conv_fn, sizes[4], 3, with_bias=with_bias), act]
    layer_6 = [max_pool, Partial(conv_fn, sizes[5], 3, with_bias=with_bias), act]
    layer_7 = [bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[6], with_bias=with_bias), act]
    layer_8 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8],


def conv_6_2_bn(sizes, number_classes, activation_fn=relu, with_bias=True):
    """ Convnet with 6 convolutional layers followed by 2 fully-connected, with BN added after
    every layer apart from the final one.
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    train_layer_1 = [Partial(Conv_BN, True, sizes[0], 3, with_bias=with_bias), act]
    train_layer_2 = [Partial(Conv_BN, True, sizes[1], 3, with_bias=with_bias), act]
    train_layer_3 = [max_pool, Partial(Conv_BN, True, sizes[2], 3, with_bias=with_bias), act]
    train_layer_4 = [Partial(Conv_BN, True, sizes[3], 3, with_bias=with_bias), act]
    train_layer_5 = [max_pool, Partial(Conv_BN, True, sizes[4], 3, with_bias=with_bias), act]
    train_layer_6 = [max_pool, Partial(Conv_BN, True, sizes[5], 3, with_bias=with_bias), act]
    train_layer_7 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, True, sizes[6], with_bias=with_bias), act]
    layer_8 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [Partial(Conv_BN, False, sizes[0], 3, with_bias=with_bias), act]
    test_layer_2 = [Partial(Conv_BN, False, sizes[1], 3, with_bias=with_bias), act]
    test_layer_3 = [max_pool, Partial(Conv_BN, False, sizes[2], 3, with_bias=with_bias), act]
    test_layer_4 = [Partial(Conv_BN, False, sizes[3], 3, with_bias=with_bias), act]
    test_layer_5 = [max_pool, Partial(Conv_BN, False, sizes[4], 3, with_bias=with_bias), act]
    test_layer_6 = [max_pool, Partial(Conv_BN, False, sizes[5], 3, with_bias=with_bias), act]
    test_layer_7 = [bigger_max_pool, hk.Flatten, Partial(Linear_BN, False, sizes[6], with_bias=with_bias), act]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, train_layer_5, train_layer_6, train_layer_7, layer_8], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, test_layer_5, test_layer_6, test_layer_7, layer_8],


##############################
# Batchnorm activations architecture  # TODO: Poor solution; should redesign fully how models are built

# The architecture below are to retrieve activations value pre-relu and pre-bn for visualization purposes;
# to build histograms of those at some given time step
##############################
def conv_4_2_act_pre_relu(sizes, number_classes, activation_fn=relu):
    """ conv_4_2 slightly modified to retrieve the activation value before the relu
    """
    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    layer_1 = [Partial(hk.Conv2D, sizes[0], 3)]
    layer_2 = [act, max_pool, Partial(hk.Conv2D, sizes[1], 3)]
    layer_3 = [act, max_pool, Partial(hk.Conv2D, sizes[2], 3)]
    layer_4 = [act, max_pool, Partial(hk.Conv2D, sizes[3], 3)]
    layer_5 = [act, bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4])]
    layer_6 = [act, Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6],


def conv_4_2_act_pre_bn(sizes, number_classes, activation_fn=relu):
    """ conv_4_2 slightly modified to retrieve the activation value before the bn layer
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    layer_1 = [Partial(hk.Conv2D, sizes[0], 3)]
    train_layer_2 = [Partial(Base_BN, True), act, max_pool, Partial(hk.Conv2D, sizes[1], 3)]
    train_layer_3 = [Partial(Base_BN, True), act, max_pool, Partial(hk.Conv2D, sizes[2], 3)]
    train_layer_4 = [Partial(Base_BN, True), act, max_pool, Partial(hk.Conv2D, sizes[3], 3)]
    train_layer_5 = [Partial(Base_BN, True), act, bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4])]
    train_layer_6 = [Partial(Base_BN, True), act, Partial(hk.Linear, number_classes)]

    test_layer_2 = [Partial(Base_BN, False), act, max_pool, Partial(hk.Conv2D, sizes[1], 3)]
    test_layer_3 = [Partial(Base_BN, False), act, max_pool, Partial(hk.Conv2D, sizes[2], 3)]
    test_layer_4 = [Partial(Base_BN, False), act, max_pool, Partial(hk.Conv2D, sizes[3], 3)]
    test_layer_5 = [Partial(Base_BN, False), act, bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4])]
    test_layer_6 = [Partial(Base_BN, False), act, Partial(hk.Linear, number_classes)]

    return [layer_1, train_layer_2, train_layer_3, train_layer_4, train_layer_5, train_layer_6], \
           [layer_1, test_layer_2, test_layer_3, test_layer_4, test_layer_5, test_layer_6]


def conv_4_2_act_post_bn(sizes, number_classes, activation_fn=relu):
    """ conv_4_2 slightly modified to retrieve the activation value after the bn layer but before the activation
    """

    def act():
        return activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    bigger_max_pool = Partial(hk.MaxPool, window_shape=4, strides=4, padding="VALID")

    train_layer_1 = [Partial(hk.Conv2D, sizes[0], 3), Partial(Base_BN, True)]
    train_layer_2 = [act, max_pool, Partial(hk.Conv2D, sizes[1], 3), Partial(Base_BN, True)]
    train_layer_3 = [act, max_pool, Partial(hk.Conv2D, sizes[2], 3), Partial(Base_BN, True)]
    train_layer_4 = [act, max_pool, Partial(hk.Conv2D, sizes[3], 3), Partial(Base_BN, True)]
    train_layer_5 = [act, bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4]), Partial(Base_BN, True)]
    layer_6 = [act, Partial(hk.Linear, number_classes)]

    test_layer_1 = [Partial(hk.Conv2D, sizes[0], 3), Partial(Base_BN, False)]
    test_layer_2 = [act, max_pool, Partial(hk.Conv2D, sizes[1], 3), Partial(Base_BN, False)]
    test_layer_3 = [act, max_pool, Partial(hk.Conv2D, sizes[2], 3), Partial(Base_BN, False)]
    test_layer_4 = [act, max_pool, Partial(hk.Conv2D, sizes[3], 3), Partial(Base_BN, False)]
    test_layer_5 = [act, bigger_max_pool, hk.Flatten, Partial(hk.Linear, sizes[4]), Partial(Base_BN, False)]

    return [train_layer_1, train_layer_2, train_layer_3, train_layer_4, train_layer_5, layer_6], \
           [test_layer_1, test_layer_2, test_layer_3, test_layer_4, test_layer_5, layer_6]
