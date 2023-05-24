""" Models definition for vgg nets architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax.nn
from jax.tree_util import Partial
from utils.utils import ReluMod

from models.bn_base_unit import Base_BN, Conv_BN, Linear_BN
from models.dropout_units import Base_Dropout


def vgg16(sizes, number_classes, bn_config, activation_fn=ReluMod, with_bias=True):
    """Prunable VGG16 implementation.
       default sizes=(64, 4096)
    """

    act = activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], 4 * sizes[0],
                 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 16 * sizes[0], sizes[1], sizes[1], sizes[1]]

    max_pool = Partial(hk.MaxPool, window_shape=2, strides=2, padding="VALID")
    avg_pool = Partial(hk.AvgPool, window_shape=2, strides=2, padding="VALID")

    def make_layers(training):
        layers = []
        for i in range(16):
            if i < 13:
                if i in [2, 4, 7, 10]:
                    layers.append([max_pool, Partial(Conv_BN, training, sizes[i], 3, bn_config, with_bias=with_bias), act])
                else:
                    layers.append([Partial(Conv_BN, training, sizes[i], 3, bn_config, with_bias=with_bias), act])
            elif i == 13:
                layers.append([avg_pool, hk.Flatten, Partial(Linear_BN, training, sizes[i], bn_config, with_bias=with_bias), act])
            elif i < 15:
                layers.append([Partial(Linear_BN, training, sizes[i], bn_config, with_bias=with_bias), act])
            else:
                layers.append([Partial(hk.Linear, number_classes, with_bias=with_bias)])
        return layers

    return make_layers(True), make_layers(False)
