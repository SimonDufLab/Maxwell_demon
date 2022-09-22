""" Models definition for the differrent resnet architecture"""
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from typing import Any, Mapping, Optional, Sequence, Union

from haiku.nets import ResNet
from haiku._src.nets.resnet import check_length

FloatStrOrBool = Union[str, float, bool]

# def resnet_block(channels: int, stride: Union[int, Sequence[int]], use_projection: bool, bottleneck: bool,
#                  is_training: bool, name: Optional[str] = None):
#     """ Resnet block with optional bottleneck;
#      adapted from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py"""
#
#     create_scale = True
#     create_offset = True
#     decay_rate = 0.999
#     bn_config = (create_scale, create_offset, decay_rate)
#
#     if use_projection:
#         proj_conv = hk.Conv2D(
#             output_channels=channels,
#             kernel_shape=1,
#             stride=stride,
#             with_bias=False,
#             padding="SAME",
#             name="shortcut_conv")
#
#         proj_batchnorm = hk.BatchNorm(name="shortcut_batchnorm", *bn_config)
#
#     channel_div = 4 if bottleneck else 1
#     conv_0 = hk.Conv2D(
#         output_channels=channels // channel_div,
#         kernel_shape=1 if bottleneck else 3,
#         stride=1 if bottleneck else stride,
#         with_bias=False,
#         padding="SAME",
#         name="conv_0")
#     bn_0 = hk.BatchNorm(name="batchnorm_0", *bn_config)
#
#     conv_1 = hk.Conv2D(
#         output_channels=channels // channel_div,
#         kernel_shape=3,
#         stride=stride if bottleneck else 1,
#         with_bias=False,
#         padding="SAME",
#         name="conv_1")
#
#     bn_1 = hk.BatchNorm(name="batchnorm_1", *bn_config)
#     layers = ((conv_0, bn_0), (conv_1, bn_1))

# global batch normalization configuration
bn_config = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}


class ResnetBlock(ResNet.BlockV1):
    """Resnet block that also outputs the activations"""

    def __init__(
            self,
            channels: int,
            stride: Union[int, Sequence[int]],
            use_projection: bool,
            bottleneck: bool,
            is_training: bool,
            name: Optional[str] = None,
    ):
        self.is_training = is_training
        super().__init__(channels=channels, stride=stride, use_projection=use_projection, bn_config=bn_config,
                         bottleneck=bottleneck, name=name)

    def __call__(self, inputs):
        out = shortcut = inputs
        activations = []

        if self.use_projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_batchnorm(shortcut, self.is_training)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            out = bn_i(out, self.is_training)
            if i < len(self.layers) - 1:  # Don't apply relu on last layer
                out = jax.nn.relu(out)
                activations.append(out)

        out = jax.nn.relu(out + shortcut)
        activations.append(out)

        return out, activations


class ResnetInit(hk.Module):
    """Create the initial layer for resnet models"""
    def __init__(
            self,
            is_training: bool,
            conv_config: Optional[Mapping[str, FloatStrOrBool]],
            bn_config: Optional[Mapping[str, FloatStrOrBool]],
            name: Optional[str] = None):
        super().__init__(name=name)
        self.is_training = is_training
        self.bn = hk.BatchNorm(name="init_bn", **bn_config)
        self.conv = hk.Conv2D(**conv_config)

    def __call__(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x, is_training=self.is_training)
        return jax.nn.relu(x)


def block_group(channels: int, num_blocks: int, stride: Union[int, Sequence[int]], bottleneck: bool,
                use_projection: bool):
    """Adapted from: https://github.com/deepmind/dm-haiku/blob/d6e3c2085253735c3179018be495ebabf1e6b17c/
    haiku/_src/nets/resnet.py#L200"""

    train_layers = []
    test_layers = []

    for i in range(num_blocks):
        train_layers.append(
            [Partial(ResnetBlock, channels=channels,
                     stride=(1 if i else stride),
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     is_training=True,)])
                     # name=f"block_{i}")])
        test_layers.append(
            [Partial(ResnetBlock, channels=channels,
                     stride=(1 if i else stride),
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     is_training=False,)])
                     # name=f"block_{i}")])

    return train_layers, test_layers


def resnet_model(blocks_per_group: Sequence[int],
                 num_classes: int,
                 bottleneck: bool = True,
                 channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
                 use_projection: Sequence[bool] = (True, True, True, True),
                 logits_config: Optional[Mapping[str, Any]] = None,
                 name: Optional[str] = None,
                 initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
                 strides: Sequence[int] = (1, 2, 2, 2),):

    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    check_length(4, strides, "strides")

    # def act():
    #     return jax.nn.relu

    train_layers = [[Partial(ResnetInit, is_training=True, conv_config=initial_conv_config, bn_config=bn_config)]]
    test_layers = [[Partial(ResnetInit, is_training=False, conv_config=initial_conv_config, bn_config=bn_config)]]

    # train_layers.append([Partial(hk.Conv2D, **initial_conv_config),
    #                      Partial(Partial(hk.BatchNorm, name="initial_batchnorm", **bn_config), is_training=True), act])
    # test_layers.append([Partial(hk.Conv2D, **initial_conv_config),
    #                     Partial(Partial(hk.BatchNorm, name="initial_batchnorm", **bn_config), is_training=False), act])

    for i, stride in enumerate(strides):
        block_train_layers, block_test_layers = block_group(channels=channels_per_group[i],
                                                            num_blocks=blocks_per_group[i],
                                                            stride=stride,
                                                            bottleneck=bottleneck,
                                                            use_projection=use_projection[i])
        if i == 0:
            max_pool = Partial(hk.MaxPool, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")
            block_train_layers[0] = [max_pool] + block_train_layers[0]
            block_test_layers[0] = [max_pool] + block_test_layers[0]

        train_layers += block_train_layers
        test_layers += block_test_layers

    def layer_mean():
        return Partial(jnp.mean, axis=(1, 2))

    final_layer = [layer_mean, Partial(hk.Linear, num_classes, **logits_config)]

    train_layers.append(final_layer)
    test_layers.append(final_layer)

    return train_layers, test_layers


default_logits_config = {"w_init": jnp.zeros, "name": "logits"}
default_initial_conv_config = {"kernel_shape": 7,
                               "stride": 2,
                               "with_bias": False,
                               "padding": "SAME",
                               "name": "initial_conv"}


def resnet18(size: Union[int, Sequence[int]],
             num_classes: int,
             logits_config: Optional[Mapping[str, Any]] = default_logits_config,
             initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = default_initial_conv_config,
             strides: Sequence[int] = (1, 2, 2, 2)):

    resnet_config = {
                    "blocks_per_group": (2, 2, 2, 2),
                    "bottleneck": False,
                    "channels_per_group": (size, size*2, size*4, size*8),  # typical resnet18 size = 64
                    "use_projection": (False, True, True, True),
                    }
    default_initial_conv_config["output_channels"] = size

    return resnet_model(num_classes=num_classes,
                        initial_conv_config=initial_conv_config,
                        strides=strides,
                        logits_config=logits_config,
                        **resnet_config)
