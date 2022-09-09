""" Models definition for MLP lie architecture. Defined fitting requirements of repo"""
import haiku as hk
from jax.tree_util import Partial
from jax.nn import relu

from models.bn_base_unit import Linear_BN


def mlp_3(sizes, number_classes, activation_fn=relu):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units"""
    def act():
        return activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0]), act]
    layer_2 = [Partial(hk.Linear, sizes[1]), act]
    layer_3 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3],


def mlp_3_bn(sizes, number_classes, activation_fn=relu):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units,
    , with BN added after every layer apart from the final one"""
    def act():
        return activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    train_layer_1 = [hk.Flatten, Partial(Linear_BN, True, sizes[0]), act]
    train_layer_2 = [Partial(Linear_BN, True, sizes[1]), act]
    layer_3 = [Partial(hk.Linear, number_classes)]

    test_layer_1 = [hk.Flatten, Partial(Linear_BN, False, sizes[0]), act]
    test_layer_2 = [Partial(Linear_BN, False, sizes[1]), act]

    return [train_layer_1, train_layer_2, layer_3], [test_layer_1, test_layer_2, layer_3]
