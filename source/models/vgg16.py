""" Models definition for vgg nets architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax.nn
import jax.numpy as jnp
from jax.tree_util import Partial
from utils.utils import ReluActivationModule, MaxPool, AvgPool
from typing import Optional, Mapping, Union

from haiku._src.base import current_name
from models.bn_base_unit import Base_BN, Conv_BN, Linear_BN
from models.dropout_units import Base_Dropout

FloatStrOrBool = Union[str, float, bool]


class LogitsVGG(hk.Module):
    """Create the final linear layers for vgg models. Typical implementation, with only one fc layer"""

    def __init__(
            self,
            num_classes: int,
            logits_config: Optional[Mapping[str, FloatStrOrBool]] = {},
            with_bias: bool = True,
            name: Optional[str] = None,
            parent: Optional[hk.Module] = None):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.bundle_name = current_name()
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.logits_layer = hk.Linear(num_classes, with_bias=with_bias, **logits_config)

    def __call__(self, inputs):
        activations = []
        block_name = self.bundle_name + "/~/"
        x = jax.vmap(jnp.ravel, in_axes=0)(inputs)  # flatten
        x = self.logits_layer(x)

        logits_name = block_name + self.logits_layer.name
        self.activation_mapping[logits_name] = {"preceding": self.preceding_activation_name,
                                                "following": None}
        return x, activations

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return None


def vgg16(sizes, num_classes, bn_config, activation_fn=ReluActivationModule, with_bias=True):
    """Prunable VGG16 implementation.
       default sizes=(64, 4096)
    """

    act = activation_fn

    if len(sizes) == 2:  # Size can be specified with 2 args
        sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], 4 * sizes[0],
                 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 16 * sizes[0], sizes[1], sizes[1], sizes[1]]

    max_pool = Partial(MaxPool, window_shape=2, strides=2, padding="VALID")
    avg_pool = Partial(AvgPool, window_shape=2, strides=2, padding="VALID")

    def make_layers(training):
        layers = []
        for i in range(16):
            if i < 13:
                if i in [2, 4, 7, 10]:
                    layers.append([max_pool, Partial(Conv_BN, training, sizes[i], 3, bn_config=bn_config,
                                                     with_bias=with_bias, activation_fn=act)])
                else:
                    layers.append([Partial(Conv_BN, training, sizes[i], 3, bn_config=bn_config,
                                           with_bias=with_bias, activation_fn=act)])
            elif i == 13:
                layers.append([avg_pool, Partial(Linear_BN, training, sizes[i], bn_config=bn_config,
                                                             with_bias=with_bias, activation_fn=act)])
            elif i < 15:
                layers.append([Partial(Linear_BN, training, sizes[i], bn_config=bn_config,
                                       with_bias=with_bias, activation_fn=act)])
            else:
                layers.append([Partial(LogitsVGG, num_classes, with_bias=with_bias)])
        return layers

    return make_layers(True), make_layers(False)
