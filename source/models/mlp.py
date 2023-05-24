""" Models definition for MLP lie architecture. Defined fitting requirements of repo"""
import haiku as hk
from jax.tree_util import Partial
from jax.nn import relu, tanh
from utils.utils import ReluMod

from models.bn_base_unit import Linear_BN, Base_BN
from models.dropout_units import Base_Dropout


def mlp_3(sizes, number_classes, activation_fn=ReluMod, with_bias=True, tanh_head=False):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units"""
    act = activation_fn

    def _tanh():
        return lambda x: tanh(2*x)

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act]
    layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act]
    if tanh_head:
        layer_3 = [Partial(hk.Linear, number_classes, with_bias=with_bias), _tanh]
    else:
        layer_3 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    return [layer_1, layer_2, layer_3],


def mlp_3_dropout(sizes, number_classes, activation_fn=ReluMod, dropout_rate=0, with_bias=True):
    """ Dropout version of mlp_3 model"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    train_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    train_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    layer_3 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act, Partial(Base_Dropout, 0)]  # dropout is zero at eval
    test_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act, Partial(Base_Dropout, 0)]

    return [train_layer_1, train_layer_2, layer_3], [test_layer_1, test_layer_2, layer_3]


def mlp_3_bn(sizes, number_classes, bn_config, activation_fn=ReluMod, with_bias=True):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units,
    , with BN added after every layer apart from the final one"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    train_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), Partial(Base_BN, True, bn_config), act]
    train_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), Partial(Base_BN, True, bn_config), act]
    layer_3 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), Partial(Base_BN, False, bn_config), act]
    test_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), Partial(Base_BN, False, bn_config), act]

    return [train_layer_1, train_layer_2, layer_3], [test_layer_1, test_layer_2, layer_3]

##############################
# Batchnorm activations architecture  # TODO: Poor solution; should redesign fully how models are built

# The architecture below are to retrieve activations value pre-relu and pre-bn for visualization purposes;
# to build histograms of those at some given time step
##############################


def mlp_3_act_pre_relu(sizes, number_classes, activation_fn=ReluMod):
    """ MLP with 2 hidden units used only to retrieve activations value pre-relu"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0])]
    layer_2 = [act, Partial(hk.Linear, sizes[1])]
    layer_3 = [act, Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3],


def mlp_3_act_pre_bn(sizes, number_classes, bn_config, activation_fn=ReluMod):
    """ MLP with 2 hidden units used only to retrieve activations value pre-bacthnorm"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0])]
    train_layer_2 = [Partial(Base_BN, True, bn_config), act, Partial(hk.Linear, sizes[1])]
    train_layer_3 = [Partial(Base_BN, True, bn_config), act, Partial(hk.Linear, number_classes)]

    test_layer_2 = [Partial(Base_BN, False, bn_config), act, Partial(hk.Linear, sizes[1])]
    test_layer_3 = [Partial(Base_BN, False, bn_config), act, Partial(hk.Linear, number_classes)]

    return [layer_1, train_layer_2, train_layer_3], [layer_1, test_layer_2, test_layer_3]


def mlp_3_act_post_bn(sizes, number_classes, bn_config, activation_fn=ReluMod):
    """ MLP with 2 hidden units used only to retrieve activations value post-bacthnorm"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    train_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0]), Partial(Base_BN, True, bn_config)]
    train_layer_2 = [act, Partial(hk.Linear, sizes[1]), Partial(Base_BN, True, bn_config)]
    layer_3 = [act, Partial(hk.Linear, number_classes)]

    test_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0]), Partial(Base_BN, False, bn_config)]
    test_layer_2 = [act, Partial(hk.Linear, sizes[1]), Partial(Base_BN, False, bn_config)]

    return [train_layer_1, train_layer_2, layer_3], [test_layer_1, test_layer_2, layer_3]


##############################
# Architecture for regression (scalar outputs)
##############################
def mlp_3_reg(sizes, activation_fn=ReluMod, with_bias=True):
    """ Build a MLP with 2 hidden layers similar to popular LeNet
    Designed for regression tasks: 1 output"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act]
    layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act]
    layer_3 = [Partial(hk.Linear, 1, with_bias=with_bias)]

    return [layer_1, layer_2, layer_3],
