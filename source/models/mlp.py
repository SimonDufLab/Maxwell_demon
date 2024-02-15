""" Models definition for MLP lie architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.nn import relu, tanh
from utils.utils import ReluActivationModule, IdentityActivationModule
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from models.bn_base_unit import Linear_BN, Base_BN
from models.dropout_units import Base_Dropout

FloatStrOrBool = Union[str, float, bool]


class LinearLayer(hk.Module):
    def __init__(
            self,
            output_size: int,
            activation_fn: Optional[hk.Module],
            with_bias: bool,
            with_bn: bool = False,
            is_training: bool = True,
            fc_config: Optional[Mapping[str, FloatStrOrBool]] = {},
            bn_config: Optional[Mapping[str, FloatStrOrBool]] = {},
            name: Optional[str] = None,
            parent: Optional[hk.Module] = None):
        super().__init__(name=name)
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.with_bn = with_bn
        self.fc_layer = hk.Linear(output_size, with_bias=with_bias, **fc_config)
        if with_bn:
            self.bn_layer = Base_BN(is_training=is_training, bn_config=bn_config)
        if activation_fn:
            self.activation_layer = activation_fn()
        else:
            self.activation_layer = None

    def __call__(self, x):
        block_name = self.name + "/~/"
        x = jax.vmap(jnp.ravel, in_axes=0)(x)  # flatten
        x = self.fc_layer(x)
        if self.with_bn:
            x = self.bn_layer(x)
        if self.activation_layer:
            x = self.activation_layer(x)
            activation_name = block_name + self.activation_layer.name
            self.activation_mapping[activation_name] = {"preceding": None,
                                                        "following": activation_name}
        else:
            activation_name = None

        fc_name = block_name + self.fc_layer.name
        self.activation_mapping[fc_name] = {"preceding": self.preceding_activation_name,
                                            "following": activation_name}

        if self.with_bn:
            bn_name = block_name + self.bn_layer.name
            self.activation_mapping[bn_name] = {"preceding": None,
                                                "following": activation_name}

        self.last_act_name = activation_name

        return x

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name


def mlp_3(sizes, number_classes, activation_fn=ReluActivationModule, with_bias=True, with_bn=False, bn_config={}, tanh_head=False):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units"""
    act = activation_fn

    def _tanh():
        return lambda x: tanh(2*x)

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [Partial(LinearLayer, sizes[0], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act)]  # hk.Flatten, in LinearLayer
    layer_2 = [Partial(LinearLayer, sizes[1], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act)]
    if tanh_head:
        layer_3 = [Partial(LinearLayer, number_classes, with_bias=with_bias, activation_fn=_tanh)]
    else:
        layer_3 = [Partial(LinearLayer, number_classes, with_bias=with_bias, activation_fn=None)]

    return [layer_1, layer_2, layer_3],


def mlp_3_dropout(sizes, number_classes, activation_fn=ReluActivationModule, dropout_rate=0, with_bias=True):
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


def mlp_3_bn(sizes, number_classes, bn_config, activation_fn=ReluActivationModule, with_bias=True):
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


def mlp_3_act_pre_relu(sizes, number_classes, activation_fn=ReluActivationModule):
    """ MLP with 2 hidden units used only to retrieve activations value pre-relu"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0])]
    layer_2 = [act, Partial(hk.Linear, sizes[1])]
    layer_3 = [act, Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3],


def mlp_3_act_pre_bn(sizes, number_classes, bn_config, activation_fn=ReluActivationModule):
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


def mlp_3_act_post_bn(sizes, number_classes, bn_config, activation_fn=ReluActivationModule):
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
def mlp_3_reg(sizes, activation_fn=ReluActivationModule, with_bias=True):
    """ Build a MLP with 2 hidden layers similar to popular LeNet
    Designed for regression tasks: 1 output"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act]
    layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act]
    layer_3 = [Partial(hk.Linear, 1, with_bias=with_bias)]

    return [layer_1, layer_2, layer_3],
