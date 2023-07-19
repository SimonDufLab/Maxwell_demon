""" Models definition for the differrent resnet architecture"""
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.nn import relu
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from models.bn_base_unit import Base_BN
from utils.utils import ReluActivationModule, MaxPool

from haiku.nets import ResNet
from haiku._src.nets.resnet import check_length
from haiku._src.utils import replicate
from haiku._src.conv import to_dimension_numbers

FloatStrOrBool = Union[str, float, bool]


def init_identity_conv(shape, dtype):
    eye = jnp.eye(shape[-1])
    return eye.reshape(shape).astype(dtype)


class IdentityConv2D(hk.Module):

    def __init__(
        self,
        out_channels: int,
        data_format: str = "NHWC",
        name: Optional[str] = None,
                ):
        """ Utility layer that performs a convolution identity transformation initially. Maintain a state but no parameter.
            Used to prune skip connections
        """
        super().__init__(name=name)
        self.out_channels = out_channels
        self.data_format = data_format
        self.channel_index = hk.get_channel_index(data_format)
        self.dimension_numbers = to_dimension_numbers(
            2, channels_last=(self.channel_index == -1),
            transpose=False)

    def __call__(self,
                 inputs: Any,  # Used to be jax.Array; but cluster jax version < 0.4.1 (not compatible)
                 precision: Optional[jax.lax.Precision] = None,
                 ) -> Any:  # Again; switch to jax.Array when version updated on cluster
        w = hk.get_state("w", (1, 1, inputs.shape[-1], self.out_channels), inputs.dtype, init=init_identity_conv)

        # out = jax.lax.conv(inputs, w, window_strides=replicate(1, 2, "strides"), padding="SAME")
        out = jax.lax.conv_general_dilated(inputs,
                                           w,
                                           window_strides=replicate(1, 2, "strides"),
                                           padding="SAME",
                                           dimension_numbers=self.dimension_numbers,
                                           precision=precision)

        return out

    @property
    def w(self):
        return hk.get_state("w")

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


class ResnetBlockV1(hk.Module):
    """Resnet block that also outputs the activations"""

    def __init__(
            self,
            channels: Sequence[int],
            stride: Union[int, Sequence[int]],
            activation_fn: hk.Module,
            use_projection: bool,
            bottleneck: bool,
            is_training: bool,
            with_bn: bool,
            bn_config: dict = base_bn_config,
            name: Optional[str] = None,
            preceding_activation_name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name
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
        else:
            self.identity_skip = IdentityConv2D(out_channels=channels[1], name="identity_skip")

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
        layers = ((conv_0, bn_0, activation_fn()), (conv_1, bn_1, activation_fn()))

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
            layers = layers + ((conv_2, bn_2, activation_fn()),)

        self.layers = layers
        self.is_training = is_training
        # self.activation_fn = activation_fn
        self.with_bn = with_bn

    def __call__(self, inputs):
        out = shortcut = inputs
        activations = []
        block_name = self.name + "/~/"

        if self.use_projection:
            shortcut = self.proj_conv(shortcut)
            skip_layer_name = block_name+self.proj_conv.name
            if self.with_bn:
                shortcut = self.proj_batchnorm(shortcut, self.is_training)
                skip_bn_name = block_name+self.proj_batchnorm.name
        else:
            shortcut = self.identity_skip(shortcut)
            skip_layer_name = block_name+self.identity_skip.name

        prev_act_name = self.preceding_activation_name
        for i, (conv_i, bn_i, act_i) in enumerate(self.layers):
            conv_name = block_name+conv_i.name
            self.activation_mapping[conv_name] = {"preceding": prev_act_name,
                                                  "following": block_name + act_i.name}
            self.activation_mapping[block_name + act_i.name] = {"preceding": None,
                                                                "following": block_name + act_i.name}
            out = conv_i(out)
            if self.with_bn:
                out = bn_i(out, self.is_training)
                bn_name = block_name+bn_i.name
                self.activation_mapping[bn_name] = {"preceding": None,
                                                    "following": block_name + act_i.name}
            prev_act_name = block_name + act_i.name
            if i < len(self.layers) - 1:  # Don't apply act right away on last layer
                out = act_i(out)
                activations.append(out)

        # try:
        self.last_act_name = block_name + act_i.name
        out = act_i(out + shortcut)
        self.activation_mapping[skip_layer_name] = {"preceding": self.preceding_activation_name,
                                                    "following": self.last_act_name}
        if self.with_bn and self.use_projection:
            self.activation_mapping[skip_bn_name] = {"preceding": None,
                                                     "following": self.last_act_name}
        # except:
        #     print(out.shape)
        #     print(shortcut.shape)
        #     print(self.layers[-1][0].name)
        #     print(self.proj_conv.name)
        #     raise SystemExit
        activations.append(out)

        return out, activations

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name


class ResnetBlockV2(ResnetBlockV1):
    """ Resnet block that also outputs the activations
        Difference with V1: bn is after concatenation with shortcut"""

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
        super().__init__(
            channels=channels,
            stride=stride,
            activation_fn=activation_fn,
            use_projection=use_projection,
            bottleneck=bottleneck,
            is_training=is_training,
            with_bn=with_bn,
            bn_config=bn_config,
            name=name)

    def __call__(self, inputs):
        out = shortcut = inputs
        activations = []

        if self.use_projection:
            shortcut = self.proj_conv(shortcut)
        else:
            shortcut = self.identity_skip(shortcut)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            if i < len(self.layers) - 1:  # Don't apply activation and bn right away on last block layer
                if self.with_bn:
                    out = bn_i(out, self.is_training)
                out = self.activation_fn(out)
                activations.append(out)

        out = self.activation_fn(out + shortcut)
        if self.with_bn:
            out = bn_i(out, self.is_training)
        activations.append(out)

        return out, activations


class ResnetInit(hk.Module):
    """Create the initial layer for resnet models"""
    def __init__(
            self,
            is_training: bool,
            activation_fn: hk.Module,
            conv_config: Optional[Mapping[str, FloatStrOrBool]],
            bn_config: Optional[Mapping[str, FloatStrOrBool]],
            with_bn: bool,
            name: Optional[str] = None,
            preceding_activation_name: Optional[str] = None):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name
        self.is_training = is_training
        self.bn = hk.BatchNorm(name="init_bn", **bn_config)
        self.conv = hk.Conv2D(**conv_config)
        self.activation_fn = activation_fn()
        self.with_bn = with_bn

    def __call__(self, inputs):
        block_name = self.name + "/~/"
        x = self.conv(inputs)
        if self.with_bn:
            x = self.bn(x, is_training=self.is_training)

        x = self.activation_fn(x)

        conv_name = block_name + self.conv.name
        self.activation_mapping[conv_name] = {"preceding": self.preceding_activation_name,
                                              "following": block_name + self.activation_fn.name}
        self.activation_mapping[block_name + self.activation_fn.name] = {"preceding": None,
                                                                         "following": block_name + self.activation_fn.name}
        if self.with_bn:
            bn_name = block_name + self.bn.name
            self.activation_mapping[bn_name] = {"preceding": None,
                                                "following": block_name + self.activation_fn.name}
        self.last_act_name = block_name + self.activation_fn.name

        return x

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name


def block_group(channels: Sequence[int], num_blocks: int, stride: Union[int, Sequence[int]], activation_fn: hk.Module, bottleneck: bool,
                use_projection: bool, with_bn: bool, bn_config: dict, resnet_block: hk.Module = ResnetBlockV1):
    """Adapted from: https://github.com/deepmind/dm-haiku/blob/d6e3c2085253735c3179018be495ebabf1e6b17c/
    haiku/_src/nets/resnet.py#L200"""

    train_layers = []
    test_layers = []
    layer_per_block = len(channels)//num_blocks

    for i in range(num_blocks):
        train_layers.append(
            [Partial(resnet_block, channels=channels[i*layer_per_block:(i+1)*layer_per_block],
                     stride=(1 if i else stride),
                     activation_fn=activation_fn,
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     with_bn=with_bn,
                     bn_config=bn_config,
                     is_training=True,)])
                     # name=f"block_{i}")])
        test_layers.append(
            [Partial(resnet_block, channels=channels[i*layer_per_block:(i+1)*layer_per_block],
                     stride=(1 if i else stride),
                     activation_fn=activation_fn,
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     with_bn=with_bn,
                     bn_config=bn_config,
                     is_training=False,)])
                     # name=f"block_{i}")])

    return train_layers, test_layers


class LinearBlock(hk.Module):
    """Create the final linear layers for resnet models. More than one since based on EarlyCrop implementation"""

    def __init__(
            self,
            is_training: bool,
            num_classes: int,
            activation_fn: hk.Module,
            fc_config: Optional[Mapping[str, FloatStrOrBool]],
            logits_config: Optional[Mapping[str, FloatStrOrBool]],
            bn_config: Optional[Mapping[str, FloatStrOrBool]],
            with_bn: bool,
            name: Optional[str] = None,
            preceding_activation_name: Optional[str] = None):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name
        self.with_bn = with_bn
        self.fc_layer = hk.Linear(**fc_config[0])
        # self.fc_layer2 = hk.Linear(**fc_config[1])
        self.bn_layer = Base_BN(is_training=is_training, bn_config=bn_config, name="lin_bn")
        # self.bn_layer2 = Base_BN(is_training=is_training, bn_config=bn_config, name="lin_bn2")
        self.logits_layer = hk.Linear(num_classes, **logits_config)  # TODO: de-hardencode the outputs_dim
        self.activation_layer = activation_fn()
        # self.activation_layer2 = activation_fn()

    def __call__(self, inputs):
        activations = []
        block_name = self.name + "/~/"
        x = jnp.mean(inputs, axis=(1, 2))  # Kind of average pooling layer
        # x = jax.vmap(jnp.ravel, in_axes=0)(inputs)  # flatten
        x = self.fc_layer(x)
        if self.with_bn:
            x = self.bn_layer(x)
        x = self.activation_layer(x)
        activations.append(x)
        # x = self.fc_layer2(x)
        # if self.with_bn:
        #     x = self.bn_layer2(x)
        # x = self.activation_layer2(x)
        # activations.append(x)
        x = self.logits_layer(x)

        fc_name = block_name + self.fc_layer.name
        self.activation_mapping[fc_name] = {"preceding": self.preceding_activation_name,
                                            "following": block_name + self.activation_layer.name}
        self.activation_mapping[block_name + self.activation_layer.name] = {"preceding": None,
                                                                            "following": block_name + self.activation_layer.name}
        # self.activation_mapping[block_name+self.fc_layer2.name] = {"preceding": block_name + self.activation_layer.name,
        #                                                          "following": block_name + self.activation_layer2.name}
        # self.activation_mapping[block_name + self.activation_layer2.name] = {"preceding": None,
        #                                                                     "following": block_name + self.activation_layer2.name}
        if self.with_bn:
            bn_name = block_name + self.bn_layer.name
            self.activation_mapping[bn_name] = {"preceding": None,
                                                "following": block_name + self.activation_layer.name}
        #     self.activation_mapping[block_name + self.bn_layer2.name] = {"preceding": None,
        #                                                                  "following": block_name + self.activation_layer2.name}
        logits_name = block_name + self.logits_layer.name
        self.activation_mapping[logits_name] = {"preceding": block_name + self.activation_layer.name,
                                                "following": None}
        # self.activation_mapping[logits_name] = {"preceding": self.preceding_activation_name,
        #                                         "following": None}
        return x, activations

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return None


def resnet_model(blocks_per_group: Sequence[int],
                 num_classes: int,
                 activation_fn: hk.Module,
                 bottleneck: bool = True,
                 channels_per_group: Sequence[Sequence[int]] = tuple([[64*i]*4 for i in [1, 2, 4, 8]]),
                 use_projection: Sequence[bool] = (True, True, True, True),
                 logits_config: Optional[Mapping[str, Any]] = None,
                 with_bn: bool = True,
                 bn_config: dict = base_bn_config,
                 resnet_block: hk.Module = ResnetBlockV1,  # Either V1 or V2
                 name: Optional[str] = None,
                 initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
                 strides: Sequence[int] = (1, 2, 2, 2),):

    # act = activation_fn

    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    check_length(4, strides, "strides")

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
                                                            bn_config=bn_config,
                                                            resnet_block=resnet_block)
        if i == 0:
            max_pool = Partial(MaxPool, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")
            block_train_layers[0] = [max_pool] + block_train_layers[0]
            block_test_layers[0] = [max_pool] + block_test_layers[0]

        train_layers += block_train_layers
        test_layers += block_test_layers

    # def layer_mean():
    #     return Partial(jnp.mean, axis=(1, 2))

    # if with_bn:
    #     train_final_layers = [
    #         [layer_mean, Partial(hk.Linear, **default_fc_layer_config), Partial(Base_BN, is_training=True, name="lin_bn", bn_config=bn_config), act],
    #         [Partial(hk.Linear, num_classes, **logits_config)]]  # TODO: de-hardencode the outputs_dim
    #     test_final_layers = [
    #         [layer_mean, Partial(hk.Linear, **default_fc_layer_config), Partial(Base_BN, is_training=False, name="lin_bn", bn_config=bn_config), act],
    #         [Partial(hk.Linear, num_classes, **logits_config)]]  # TODO: de-hardencode the outputs_dim
    # else:
    #     train_final_layers = [
    #         [layer_mean, Partial(hk.Linear, **default_fc_layer_config), act],
    #         [Partial(hk.Linear, num_classes, **logits_config)]]  # TODO: de-hardencode the outputs_dim
    #     test_final_layers = [
    #         [layer_mean, Partial(hk.Linear, **default_fc_layer_config), act],
    #         [Partial(hk.Linear, num_classes, **logits_config)]]  # TODO: de-hardencode the outputs_dim

    train_layers.append([Partial(LinearBlock, is_training=True, num_classes=num_classes, activation_fn=activation_fn,
                                 fc_config=(default_fc_layer_config, default_fc2_layer_config), logits_config=logits_config, bn_config=bn_config,
                                 with_bn=with_bn)])
    test_layers.append([Partial(LinearBlock, is_training=False, num_classes=num_classes, activation_fn=activation_fn,
                                fc_config=(default_fc_layer_config, default_fc2_layer_config), logits_config=logits_config, bn_config=bn_config,
                                with_bn=with_bn)])

    # train_layers += train_final_layers
    # test_layers += test_final_layers

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
default_fc_layer_config = {"with_bias": True, "w_init": kaiming_normal}
default_fc2_layer_config = {"with_bias": True, "w_init": kaiming_normal}


def resnet18(size: Union[int, Sequence[int]],
             num_classes: int,
             activation_fn: hk.Module = ReluActivationModule,
             logits_config: Optional[Mapping[str, Any]] = default_logits_config,
             initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = default_initial_conv_config,
             strides: Sequence[int] = (1, 2, 2, 2),
             with_bn: bool = True,
             bn_config: dict = base_bn_config,
             version: str = 'V1'):

    assert version in ["V1", "V2"], "version must be either V1 or V2"

    if type(size) == int:
        init_conv_size = size
        sizes = [[size*i]*4 for i in [1, 2, 4, 8]]
        fc_size = 4 * size
        # fc2_size = 2*size
    else:
        init_conv_size = size[0]
        fc_size = size[-1]
        # fc2_size = size[-1]
        sizes = size[1:-1]
        # sizes = size[1:-2]
        # sizes = size[1:]
        sizes = [sizes[i:i+4] for i in range(0, 16, 4)]

    resnet_config = {
                    "blocks_per_group": (2, 2, 2, 2),
                    "bottleneck": False,
                    "channels_per_group": sizes,  # typical resnet18 size = 64
                    "use_projection": (False, True, True, True),
                    "bn_config": bn_config
                    }
    default_initial_conv_config["output_channels"] = init_conv_size
    default_fc_layer_config["output_size"] = fc_size
    # default_fc2_layer_config["output_size"] = fc2_size

    if version == "V1":
        resnet_block_type = ResnetBlockV1
    elif version == "V2":
        resnet_block_type = ResnetBlockV2

    return resnet_model(num_classes=num_classes,
                        activation_fn=activation_fn,
                        initial_conv_config=initial_conv_config,
                        strides=strides,
                        logits_config=logits_config,
                        with_bn=with_bn,
                        resnet_block=resnet_block_type,
                        **resnet_config)
