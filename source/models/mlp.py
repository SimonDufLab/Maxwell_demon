""" Models definition for MLP lie architecture. Defined fitting requirements of repo"""
import haiku as hk
from jax.tree_util import Partial
from jax.nn import relu


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
