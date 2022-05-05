""" Models definition for MLP lie architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax
import jax.numpy as jnp


def lenet_var_size(size, number_classes):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units"""
    act = jax.nn.relu

    layer_1 = hk.Sequential([hk.Flatten(), hk.Linear(size), act])
    layer_2 = hk.Sequential([hk.Linear(size*3), act])
    layer_3 = hk.Linear(number_classes)

    return [layer_1, layer_2, layer_3]
