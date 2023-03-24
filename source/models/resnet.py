""" Models definition for the differrent resnet architecture"""
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.nn import relu
from typing import Any, Callable, Mapping, Optional, Sequence, Union

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
base_bn_config = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}


class ResnetBlock(hk.Module):
    """Resnet block that also outputs the activations"""

    def __init__(
            self,
            channels: Sequence[int],
            stride: Union[int, Sequence[int]],
            activation_fn: Callable,
            use_projection: bool,
            bottleneck: bool,
            is_training: bool,
            with_bn: bool,
            bn_config: dict = base_bn_config,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.use_projection = use_projection

        bn_config = dict(bn_config)

        if self.use_projection:
            self.proj_conv = hk.Conv2D(
                output_channels=channels[1],
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv",
                **default_block_conv_config)

            self.proj_batchnorm = hk.BatchNorm(name="shortcut_batchnorm", **bn_config)

        channel_div = 4 if bottleneck else 1
        conv_0 = hk.Conv2D(
            output_channels=channels[0],
            kernel_shape=1 if bottleneck else 3,
            stride=1 if bottleneck else stride,
            with_bias=False,
            padding="SAME",
            name="conv_0",
            **default_block_conv_config)
        bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels[1],
            kernel_shape=3,
            stride=stride if bottleneck else 1,
            with_bias=False,
            padding="SAME",
            name="conv_1",
            **default_block_conv_config)

        bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
        layers = ((conv_0, bn_0), (conv_1, bn_1))

        if bottleneck:
            conv_2 = hk.Conv2D(
                output_channels=channels[2],
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding="SAME",
                name="conv_2",
                **default_block_conv_config)

            bn_2 = hk.BatchNorm(name="batchnorm_2", scale_init=jnp.zeros, **bn_config)
            layers = layers + ((conv_2, bn_2),)

        self.layers = layers
        self.is_training = is_training
        self.activation_fn = activation_fn
        self.with_bn = with_bn

    def __call__(self, inputs):
        out = shortcut = inputs
        activations = []

        if self.use_projection:
            shortcut = self.proj_conv(shortcut)
            if self.with_bn:
                shortcut = self.proj_batchnorm(shortcut, self.is_training)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            if self.with_bn:
                out = bn_i(out, self.is_training)
            if i < len(self.layers) - 1:  # Don't apply act right away on last layer
                out = self.activation_fn(out)
                activations.append(out)

        try:
            out = self.activation_fn(out + shortcut)
        except:
            print(out.shape)
            print(shortcut.shape)
            print(self.layers[-1][0].name)
            print(self.proj_conv.name)
            raise SystemExit
        activations.append(out)

        return out, activations


class ResnetInit(hk.Module):
    """Create the initial layer for resnet models"""
    def __init__(
            self,
            is_training: bool,
            activation_fn: Callable,
            conv_config: Optional[Mapping[str, FloatStrOrBool]],
            bn_config: Optional[Mapping[str, FloatStrOrBool]],
            with_bn: bool,
            name: Optional[str] = None):
        super().__init__(name=name)
        self.is_training = is_training
        self.bn = hk.BatchNorm(name="init_bn", **bn_config)
        self.conv = hk.Conv2D(**conv_config)
        self.activation_fn = activation_fn
        self.with_bn = with_bn

    def __call__(self, inputs):
        x = self.conv(inputs)
        if self.with_bn:
            x = self.bn(x, is_training=self.is_training)
        return self.activation_fn(x)


def block_group(channels: Sequence[int], num_blocks: int, stride: Union[int, Sequence[int]], activation_fn: Callable, bottleneck: bool,
                use_projection: bool, with_bn: bool, bn_config: dict):
    """Adapted from: https://github.com/deepmind/dm-haiku/blob/d6e3c2085253735c3179018be495ebabf1e6b17c/
    haiku/_src/nets/resnet.py#L200"""

    train_layers = []
    test_layers = []
    layer_per_block = len(channels)//num_blocks

    for i in range(num_blocks):
        train_layers.append(
            [Partial(ResnetBlock, channels=channels[i*layer_per_block:(i+1)*layer_per_block],
                     stride=(1 if i else stride),
                     activation_fn=activation_fn,
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     with_bn=with_bn,
                     bn_config=bn_config,
                     is_training=True,)])
                     # name=f"block_{i}")])
        test_layers.append(
            [Partial(ResnetBlock, channels=channels[i*layer_per_block:(i+1)*layer_per_block],
                     stride=(1 if i else stride),
                     activation_fn=activation_fn,
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     with_bn=with_bn,
                     bn_config=bn_config,
                     is_training=False,)])
                     # name=f"block_{i}")])

    return train_layers, test_layers


def resnet_model(blocks_per_group: Sequence[int],
                 num_classes: int,
                 activation_fn: Callable,
                 bottleneck: bool = True,
                 channels_per_group: Sequence[Sequence[int]] = tuple([[64*i]*4 for i in [1, 2, 4, 8]]),
                 use_projection: Sequence[bool] = (True, True, True, True),
                 logits_config: Optional[Mapping[str, Any]] = None,
                 with_bn: bool = True,
                 bn_config: dict = base_bn_config,
                 name: Optional[str] = None,
                 initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
                 strides: Sequence[int] = (1, 2, 2, 2),):

    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    check_length(4, strides, "strides")

    # def act():
    #     return jax.nn.relu

    train_layers = [[Partial(ResnetInit, is_training=True, activation_fn=activation_fn, conv_config=initial_conv_config, bn_config=bn_config, with_bn=with_bn)]]
    test_layers = [[Partial(ResnetInit, is_training=False, activation_fn=activation_fn, conv_config=initial_conv_config, bn_config=bn_config, with_bn=with_bn)]]

    # train_layers.append([Partial(hk.Conv2D, **initial_conv_config),
    #                      Partial(Partial(hk.BatchNorm, name="initial_batchnorm", **bn_config), is_training=True), act])
    # test_layers.append([Partial(hk.Conv2D, **initial_conv_config),
    #                     Partial(Partial(hk.BatchNorm, name="initial_batchnorm", **bn_config), is_training=False), act])

    for i, stride in enumerate(strides):
        block_train_layers, block_test_layers = block_group(channels=channels_per_group[i],
                                                            num_blocks=blocks_per_group[i],
                                                            stride=stride,
                                                            activation_fn=activation_fn,
                                                            bottleneck=bottleneck,
                                                            use_projection=use_projection[i],
                                                            with_bn=with_bn,
                                                            bn_config=bn_config)
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


kaiming_normal = hk.initializers.VarianceScaling(2.0, 'fan_in', "truncated_normal")
# default_logits_config = {"w_init": jnp.zeros, "name": "logits"}
default_logits_config = {"name": "logits"}
default_initial_conv_config = {"kernel_shape": 7,
                               "stride": 2,
                               "with_bias": False,
                               "padding": "SAME",
                               "name": "initial_conv",
                               "w_init": kaiming_normal}
default_block_conv_config = {"w_init": kaiming_normal}


def resnet18(size: Union[int, Sequence[int]],
             num_classes: int,
             activation_fn: Callable = relu,
             logits_config: Optional[Mapping[str, Any]] = default_logits_config,
             initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = default_initial_conv_config,
             strides: Sequence[int] = (1, 2, 2, 2),
             with_bn: bool = True,
             bn_config: dict = base_bn_config):

    if type(size) == int:
        init_conv_size = size
        sizes = [[size*i]*4 for i in [1, 2, 4, 8]]
    else:
        init_conv_size = size[0]
        sizes = size[1:]
        sizes = [sizes[i:i+4] for i in range(0, 16, 4)]

    resnet_config = {
                    "blocks_per_group": (2, 2, 2, 2),
                    "bottleneck": False,
                    "channels_per_group": sizes,  # typical resnet18 size = 64
                    "use_projection": (False, True, True, True),
                    "bn_config": bn_config
                    }
    default_initial_conv_config["output_channels"] = init_conv_size

    return resnet_model(num_classes=num_classes,
                        activation_fn=activation_fn,
                        initial_conv_config=initial_conv_config,
                        strides=strides,
                        logits_config=logits_config,
                        with_bn=with_bn,
                        **resnet_config)
