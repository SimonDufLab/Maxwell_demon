""" Models definition for the differrent resnet architecture"""
import haiku as hk
import jax
from jax.tree_util import Partial
from typing import Optional, Sequence, Union

from haiku.nets import ResNet


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
        bn_config = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}
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

        return jax.nn.relu(out + shortcut), activations


def block_group(channels: int, num_blocks: int, stride: Union[int, Sequence[int]], resnet_v2: bool, bottleneck: bool,
                use_projection: bool, name: Optional[str] = None):
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
                     is_training=True,
                     name="block_%d" % (i))])
        test_layers.append(
            [Partial(ResnetBlock, channels=channels,
                     stride=(1 if i else stride),
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     is_training=False,
                     name="block_%d" % (i))])

    return train_layers, test_layers
