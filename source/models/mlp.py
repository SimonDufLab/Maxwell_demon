""" Models definition for MLP lie architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax
from jax.tree_util import Partial


def mlp_3(size, number_classes):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units"""
    def act():
        return jax.nn.relu

    layer_1 = [hk.Flatten, Partial(hk.Linear, size), act]
    layer_2 = [Partial(hk.Linear, size*3), act]
    layer_3 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3]
