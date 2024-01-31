""" Models definition for the differrent resnet architecture"""
import haiku as hk
import numpy as np
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


class CustomBatchNorm(hk.BatchNorm):
    """ Slightly modify version of BatchNorm layer that allows to fix the scale value to a given constant"""
    def __init__(
            self,
            create_scale: bool,
            create_offset: bool,
            decay_rate: float,
            eps: float = 1e-5,
            constant_scale: Optional[float] = None,
            scale_init: Optional[hk.initializers.Initializer] = None,
            offset_init: Optional[hk.initializers.Initializer] = None,
            axis: Optional[Sequence[int]] = None,
            cross_replica_axis: Optional[Union[str, Sequence[str]]] = None,
            cross_replica_axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
            data_format: str = "channels_last",
            deactivate_small_units: bool = False,  # New argument that deactivate units with magnitude smaller than mean
            name: Optional[str] = None,
    ):
        super().__init__(create_scale=create_scale, create_offset=create_offset, decay_rate=decay_rate, eps=eps,
                         scale_init=scale_init, offset_init=offset_init, axis=axis,
                         cross_replica_axis=cross_replica_axis,
                         cross_replica_axis_index_groups=cross_replica_axis_index_groups, data_format=data_format,
                         name=name)
        self.constant_scale = constant_scale
        self.deactivate_small_units = deactivate_small_units
        if self.create_scale and self.constant_scale is not None:
            raise ValueError(
                "Cannot set `constant_scale` if `create_scale=True`.")

    def __call__(  # Sole modification is that scale can be fixed
            self,
            inputs: jax.Array,
            is_training: bool,
            test_local_stats: bool = False,
            scale: Optional[jax.Array] = None,
            offset: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Computes the normalized version of the input.

        Args:
          inputs: An array, where the data format is ``[..., C]``.
          is_training: Whether this is during training.
          test_local_stats: Whether local stats are used when is_training=False.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.

        Returns:
          The array, normalized across all but the last dimension.
        """
        if self.create_scale and scale is not None:
            raise ValueError(
                "Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`.")

        channel_index = self.channel_index
        if channel_index < 0:
            channel_index += inputs.ndim

        if self.axis is not None:
            axis = self.axis
        else:
            axis = [i for i in range(inputs.ndim) if i != channel_index]

        if is_training or test_local_stats:
            mean = jnp.mean(inputs, axis, keepdims=True)
            mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)
            if self.cross_replica_axis:
                mean = jax.lax.pmean(
                    mean,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups)
                mean_of_squares = jax.lax.pmean(
                    mean_of_squares,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups)
            var = mean_of_squares - jnp.square(mean)
        else:
            mean = self.mean_ema.average.astype(inputs.dtype)
            var = self.var_ema.average.astype(inputs.dtype)

        if is_training:
            self.mean_ema(mean)
            self.var_ema(var)

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
        elif self.constant_scale:
            scale = np.ones([], dtype=w_dtype) * self.constant_scale
        elif scale is None:
            scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)

        # If deactivation is enabled, apply the deactivation logic
        if self.deactivate_small_units:
            magnitude = jnp.abs(inputs)
            mask = magnitude > jnp.abs(mean)
            inputs = inputs * mask

        eps = jax.lax.convert_element_type(self.eps, var.dtype)
        inv = scale * jax.lax.rsqrt(var + eps)
        return (inputs - mean) * inv + offset


def init_identity_conv(shape, dtype):
    eye_shape = max(shape[-2], shape[-1])
    eye = jnp.eye(eye_shape)
    eye = eye[:shape[-2], :shape[-1]]  # Do nothing when truly initialized, but allows init_fn() of rnadam to work
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
            parent: Optional[hk.Module] = None,
            final_block: bool = False,
    ):
        super().__init__(name=name)
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.use_projection = use_projection

        bn_config = dict(bn_config)

        proj_channels = channels[2] if bottleneck else channels[1]
        if self.use_projection:
            self.proj_conv = hk.Conv2D(
                output_channels=proj_channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv",
                **default_block_conv_config)

            self.proj_batchnorm = CustomBatchNorm(name="shortcut_batchnorm", **bn_config)
        else:
            self.identity_skip = IdentityConv2D(out_channels=proj_channels, name="identity_skip")

        # channel_div = 4 if bottleneck else 1
        conv_0 = hk.Conv2D(
            output_channels=channels[0],
            kernel_shape=1 if bottleneck else 3,
            stride=1 if bottleneck else stride,
            with_bias=False,
            padding="SAME",
            name="conv_0",
            **default_block_conv_config)
        bn_0 = CustomBatchNorm(name="batchnorm_0", **bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels[1],
            kernel_shape=3,
            stride=stride if bottleneck else 1,
            with_bias=False,
            padding="SAME",
            name="conv_1",
            **default_block_conv_config)

        bn_1 = CustomBatchNorm(name="batchnorm_1", **bn_config)
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

            bn_2 = CustomBatchNorm(name="batchnorm_2", scale_init=jnp.zeros, **bn_config)
            layers = layers + ((conv_2, bn_2, activation_fn()),)

        self.layers = layers
        self.is_training = is_training
        self.with_bn = with_bn

        self.final_block = final_block

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
            if self.final_block:
                out = jnp.mean(out, axis=(1, 2), keepdims=True)
                shortcut = jnp.mean(shortcut, axis=(1, 2), keepdims=True)
            if self.with_bn:
                out = bn_i(out, self.is_training)
                bn_name = block_name+bn_i.name
                self.activation_mapping[bn_name] = {"preceding": None,
                                                    "following": block_name + act_i.name}
            prev_act_name = block_name + act_i.name
            if i < len(self.layers) - 1:  # Don't apply act right away on last layer
                out = act_i(out)
                activations.append(out)

        self.last_act_name = block_name + act_i.name
        out = act_i(out + shortcut)
        self.activation_mapping[skip_layer_name] = {"preceding": self.preceding_activation_name,
                                                    "following": self.last_act_name}
        if self.with_bn and self.use_projection:
            self.activation_mapping[skip_bn_name] = {"preceding": None,
                                                     "following": self.last_act_name}
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
            activation_fn: hk.Module,
            use_projection: bool,
            bottleneck: bool,
            is_training: bool,
            with_bn: bool,
            bn_config: dict = base_bn_config,
            name: Optional[str] = None,
            parent: Optional[hk.Module] = None,
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
            name=name,
            parent=parent)
        self.delayed_activation = parent.get_delayed_activations()
        self.delayed_norm = parent.get_delayed_norm()

    def __call__(self, inputs):
        out = shortcut = inputs
        activations = []
        block_name = self.name + "/~/"

        if self.with_bn:
            out = self.delayed_norm(out, self.is_training)
        out = self.delayed_activation(out)
        activations.append(out)
        if self.use_projection:
            shortcut = self.proj_conv(out)
            skip_layer_name = block_name + self.proj_conv.name
        else:
            shortcut *= jnp.where(out > 0, 1, 0)  # Trick so that pruning is consistent even is there is a skip
            shortcut = self.identity_skip(shortcut)
            skip_layer_name = block_name + self.identity_skip.name
        prev_act_name = self.preceding_activation_name
        for i, (conv_i, bn_i, act_i) in enumerate(self.layers):
            conv_name = block_name + conv_i.name
            self.activation_mapping[conv_name] = {"preceding": prev_act_name,
                                                  "following": block_name + act_i.name}
            self.activation_mapping[block_name + act_i.name] = {"preceding": None,
                                                                "following": block_name + act_i.name}
            out = conv_i(out)
            if self.with_bn:
                bn_name = block_name + bn_i.name
                self.activation_mapping[bn_name] = {"preceding": None,
                                                    "following": block_name + act_i.name}
            prev_act_name = block_name + act_i.name
            if i < len(self.layers) - 1:  # Don't apply act right away on last layer
                if self.with_bn:
                    out = bn_i(out, self.is_training)
                out = act_i(out)
                activations.append(out)

        self.last_act_name = block_name + act_i.name
        out = out + shortcut
        self.activation_mapping[skip_layer_name] = {"preceding": self.preceding_activation_name,
                                                    "following": self.last_act_name}

        # activations.append(out)

        return out, activations

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name

    def get_delayed_activations(self):
        return self.layers[-1][-1]

    def get_delayed_norm(self):
        return self.layers[-1][-2]


class ResnetBlockV3(hk.Module):
    """BN after out+shortcut"""

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
            parent: Optional[str] = hk.Module,
    ):
        super().__init__(name=name)
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.use_projection = use_projection

        bn_config = dict(bn_config)

        proj_channels = channels[2] if bottleneck else channels[1]
        if self.use_projection:
            self.proj_conv = hk.Conv2D(
                output_channels=proj_channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv",
                **default_block_conv_config)

            self.proj_batchnorm = CustomBatchNorm(name="shortcut_batchnorm", **bn_config)
        else:
            self.identity_skip = IdentityConv2D(out_channels=proj_channels, name="identity_skip")

        # channel_div = 4 if bottleneck else 1
        conv_0 = hk.Conv2D(
            output_channels=channels[0],
            kernel_shape=1 if bottleneck else 3,
            stride=1 if bottleneck else stride,
            with_bias=False,
            padding="SAME",
            name="conv_0",
            **default_block_conv_config)
        bn_0 = CustomBatchNorm(name="batchnorm_0", **bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels[1],
            kernel_shape=3,
            stride=stride if bottleneck else 1,
            with_bias=False,
            padding="SAME",
            name="conv_1",
            **default_block_conv_config)

        bn_1 = CustomBatchNorm(name="batchnorm_1", **bn_config)
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

            bn_2 = CustomBatchNorm(name="batchnorm_2", **bn_config)
            layers = layers + ((conv_2, bn_2, activation_fn()),)

        self.layers = layers
        self.is_training = is_training
        self.with_bn = with_bn
        self.additional_bn = CustomBatchNorm(name="batchnorm_extra", **bn_config)

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

        self.last_act_name = block_name + act_i.name
        out = out + shortcut
        if self.with_bn:
            out = self.additional_bn(out, self.is_training)
            bn_name = block_name + self.additional_bn.name
            self.activation_mapping[bn_name] = {"preceding": None,
                                                "following": block_name + act_i.name}
        out = act_i(out)
        self.activation_mapping[skip_layer_name] = {"preceding": self.preceding_activation_name,
                                                    "following": self.last_act_name}
        if self.with_bn and self.use_projection:
            self.activation_mapping[skip_bn_name] = {"preceding": None,
                                                     "following": self.last_act_name}
        activations.append(out)

        return out, activations

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name


class ResnetInit(hk.Module):
    """Create the initial layer for resnet models"""
    def __init__(
            self,
            is_training: bool,
            activation_fn: hk.Module,
            conv_config: Optional[Mapping[str, FloatStrOrBool]],
            bn_config: Optional[Mapping[str, FloatStrOrBool]],
            with_bn: bool,
            v2_block: bool = False,
            name: Optional[str] = None,
            parent: Optional[hk.Module] = None):
        super().__init__(name=name)
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.is_training = is_training
        self.bn = CustomBatchNorm(name="init_bn", **bn_config)
        self.conv = hk.Conv2D(**conv_config)
        self.activation_fn = activation_fn()
        self.with_bn = with_bn
        self.v2_block = v2_block

    def __call__(self, inputs):
        activations = []
        block_name = self.name + "/~/"
        x = self.conv(inputs)
        # if self.with_bn:
        #     x = self.bn(x, is_training=self.is_training)

        if not self.v2_block:
            if self.with_bn:
                x = self.bn(x, is_training=self.is_training)
            x = self.activation_fn(x)
            activations.append(x)

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

        return x, activations

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name

    def get_delayed_activations(self):
        return self.activation_fn

    def get_delayed_norm(self):
        return self.bn


def block_group(channels: Sequence[int], num_blocks: int, stride: Union[int, Sequence[int]], activation_fn: hk.Module, bottleneck: bool,
                use_projection: bool, with_bn: bool, bn_config: dict, resnet_block: hk.Module = ResnetBlockV1, avg_in_final_block=False):
    """Adapted from: https://github.com/deepmind/dm-haiku/blob/d6e3c2085253735c3179018be495ebabf1e6b17c/
    haiku/_src/nets/resnet.py#L200"""

    train_layers = []
    test_layers = []
    layer_per_block = len(channels)//num_blocks
    avg_flag = False

    for i in range(num_blocks):
        if avg_in_final_block and (i == (num_blocks - 1)):
            avg_flag = True
        train_layers.append(
            [Partial(resnet_block, channels=channels[i*layer_per_block:(i+1)*layer_per_block],
                     stride=(1 if i else stride),
                     activation_fn=activation_fn,
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     with_bn=with_bn,
                     bn_config=bn_config,
                     is_training=True,
                     final_block=avg_flag)])
                     # name=f"block_{i}")])
        test_layers.append(
            [Partial(resnet_block, channels=channels[i*layer_per_block:(i+1)*layer_per_block],
                     stride=(1 if i else stride),
                     activation_fn=activation_fn,
                     use_projection=(i == 0 and use_projection),
                     bottleneck=bottleneck,
                     with_bn=with_bn,
                     bn_config=bn_config,
                     is_training=False,
                     final_block=avg_flag)])
                     # name=f"block_{i}")])

    return train_layers, test_layers


class LinearBlockV1(hk.Module):
    """Create the final linear layers for resnet models. Typical implementation, with only one fc layer"""

    def __init__(
            self,
            is_training: bool,
            num_classes: int,
            logits_config: Optional[Mapping[str, FloatStrOrBool]],
            with_bn: bool,
            bn_config: dict = base_bn_config,
            v2_block: bool = False,
            name: Optional[str] = None,
            parent: Optional[hk.Module] = None,
            avg_pool_layer: bool = False,
            disable_final_pooling: bool = False,
            pool_proj_channels: Optional[int] = None,  # To replace pooling layer with a projection conv
            activation_fn: Optional[hk.Module] = None,
    ):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.is_training = is_training
        self.with_bn = with_bn
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.logits_layer = hk.Linear(num_classes, **logits_config)  # TODO: de-hardencode the outputs_dim
        if pool_proj_channels:
            self.mean_layer = hk.Conv2D(
                output_channels=pool_proj_channels,
                kernel_shape=4,
                stride=4,
                with_bias=False,
                padding="VALID",
                name="final_proj_conv",
                **default_block_conv_config)
            self.flatten = jax.vmap(jnp.ravel, in_axes=0)
            self.proj_activation = activation_fn()
            self.proj_batchnorm = CustomBatchNorm(name="final_proj_batchnorm", **bn_config)
        elif avg_pool_layer:
            self.mean_layer = hk.AvgPool(window_shape=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding="VALID")
            self.flatten = jax.vmap(jnp.ravel, in_axes=0)
        else:
            self.mean_layer = Partial(jnp.mean, axis=(1, 2))  # Kind of average pooling layer
            self.flatten = lambda x: x
        self.v2_block = v2_block
        if v2_block:
            self.delayed_activation = parent.get_delayed_activations()
            self.delayed_norm = parent.get_delayed_norm()
        self.disable_final_pooling = disable_final_pooling
        self.final_proj_instead_pool = bool(pool_proj_channels)

    def __call__(self, x):
        activations = []
        if self.v2_block:
            if self.with_bn:
                x = self.delayed_norm(x, self.is_training)
            x = self.delayed_activation(x)
            activations.append(x)
        block_name = self.name + "/~/"
        if not self.disable_final_pooling:
            x = self.mean_layer(x)
            if self.final_proj_instead_pool:
                proj_name = block_name + self.mean_layer.name
                self.activation_mapping[proj_name] = {"preceding": self.preceding_activation_name,
                                                        "following": block_name + self.proj_activation.name}
                self.activation_mapping[block_name + self.proj_activation.name] = {"preceding": None,
                                                                                   "following": block_name + self.proj_activation.name}
                if self.with_bn:
                    x = self.proj_batchnorm(x, is_training=self.is_training)
                    bn_name = block_name + self.proj_batchnorm.name
                    self.activation_mapping[bn_name] = {"preceding": None,
                                                        "following": block_name + self.proj_activation.name}
                x = self.proj_activation(x)
                activations.append(x)
        x = self.flatten(x)
        # x = jax.vmap(jnp.ravel, in_axes=0)(x)  # flatten
        x = self.logits_layer(x)

        logits_name = block_name + self.logits_layer.name
        if self.final_proj_instead_pool:
            self.activation_mapping[logits_name] = {"preceding": block_name + self.proj_activation.name,
                                                    "following": None}
        else:
            self.activation_mapping[logits_name] = {"preceding": self.preceding_activation_name,
                                                    "following": None}
        return x, activations

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return None


class LinearBlockV2(hk.Module):
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
            parent: Optional[hk.Module] = None,
            avg_pool_layer: bool = False):
        super().__init__(name=name)
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.with_bn = with_bn
        self.fc_layer = hk.Linear(**fc_config[0])
        # self.fc_layer2 = hk.Linear(**fc_config[1])
        self.bn_layer = CustomBatchNorm(name="lin_bn", **bn_config)
        # self.bn_layer2 = Base_BN(is_training=is_training, bn_config=bn_config, name="lin_bn2")
        self.logits_layer = hk.Linear(num_classes, **logits_config)  # TODO: de-hardencode the outputs_dim
        self.activation_layer = activation_fn()
        # self.activation_layer2 = activation_fn()
        self.is_training = is_training
        if avg_pool_layer:
            self.mean_layer = hk.AvgPool(window_shape=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding="VALID")
        else:
            self.mean_layer = Partial(jnp.mean, axis=(1, 2))  # Kind of average pooling layer

    def __call__(self, inputs):
        activations = []
        block_name = self.name + "/~/"
        x = self.mean_layer(inputs)
        # x = jax.vmap(jnp.ravel, in_axes=0)(inputs)  # flatten
        x = self.fc_layer(x)
        if self.with_bn:
            x = self.bn_layer(x, is_training=self.is_training)
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
                 v2_linear_block: bool = False,
                 name: Optional[str] = None,
                 initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
                 strides: Sequence[int] = (1, 2, 2, 2),
                 max_pool_layer: bool = True,
                 avg_pool_layer: bool = False,
                 disable_final_pooling: bool = False,  # Solely to test impact of pooling on dead neurons in final conv
                 avg_in_final_block: bool = False,
                 pool_proj_channels: Optional[int] = None,  # Number of channels in final proj conv
                 ):

    # act = activation_fn

    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    check_length(4, strides, "strides")

    v2_blocks = (resnet_block == ResnetBlockV2)
    train_layers = [[Partial(ResnetInit, is_training=True, activation_fn=activation_fn, conv_config=initial_conv_config, bn_config=bn_config, with_bn=with_bn, v2_block=v2_blocks)]]
    test_layers = [[Partial(ResnetInit, is_training=False, activation_fn=activation_fn, conv_config=initial_conv_config, bn_config=bn_config, with_bn=with_bn,  v2_block=v2_blocks)]]

    # train_layers.append([Partial(hk.Conv2D, **initial_conv_config),
    #                      Partial(Partial(hk.BatchNorm, name="initial_batchnorm", **bn_config), is_training=True), act])
    # test_layers.append([Partial(hk.Conv2D, **initial_conv_config),
    #                     Partial(Partial(hk.BatchNorm, name="initial_batchnorm", **bn_config), is_training=False), act])

    for i, stride in enumerate(strides):
        if (i == (len(strides)-1)) and avg_in_final_block:
            avg_flag = True
        else:
            avg_flag = False
        block_train_layers, block_test_layers = block_group(channels=channels_per_group[i],
                                                            num_blocks=blocks_per_group[i],
                                                            stride=stride,
                                                            activation_fn=activation_fn,
                                                            bottleneck=bottleneck,
                                                            use_projection=use_projection[i],
                                                            with_bn=with_bn,
                                                            bn_config=bn_config,
                                                            resnet_block=resnet_block,
                                                            avg_in_final_block=avg_flag)
        if i == 0 and max_pool_layer:
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

    if avg_in_final_block:
        disable_final_pooling = True

    if v2_linear_block:
        train_layers.append([Partial(LinearBlockV2, is_training=True, num_classes=num_classes, activation_fn=activation_fn,
                                     fc_config=(default_fc_layer_config, default_fc2_layer_config), logits_config=logits_config, bn_config=bn_config,
                                     with_bn=with_bn, avg_pool_layer=avg_pool_layer)])
        test_layers.append([Partial(LinearBlockV2, is_training=False, num_classes=num_classes, activation_fn=activation_fn,
                                    fc_config=(default_fc_layer_config, default_fc2_layer_config), logits_config=logits_config, bn_config=bn_config,
                                    with_bn=with_bn, avg_pool_layer=avg_pool_layer)])
    else:
        train_layers.append(
            [Partial(LinearBlockV1, is_training=True, num_classes=num_classes, logits_config=logits_config, with_bn=with_bn, bn_config=bn_config, avg_pool_layer=avg_pool_layer, v2_block=v2_blocks, disable_final_pooling=disable_final_pooling, pool_proj_channels=pool_proj_channels, activation_fn=activation_fn)])
        test_layers.append(
            [Partial(LinearBlockV1, is_training=False, num_classes=num_classes, logits_config=logits_config, with_bn=with_bn, bn_config=bn_config, avg_pool_layer=avg_pool_layer, v2_block=v2_blocks, disable_final_pooling=disable_final_pooling, pool_proj_channels=pool_proj_channels, activation_fn=activation_fn)])

    # train_layers += train_final_layers
    # test_layers += test_final_layers

    return train_layers, test_layers


kaiming_normal = hk.initializers.VarianceScaling(2.0, 'fan_in', "truncated_normal")
pytorch_default_init = hk.initializers.VarianceScaling(1/3, 'fan_in', 'uniform')
# default_logits_config = {"w_init": jnp.zeros, "name": "logits"}
default_logits_config = {"name": "logits"}
default_initial_conv_config = {"kernel_shape": 7,
                               "stride": 2,
                               "with_bias": False,
                               "padding": "SAME",
                               "name": "initial_conv", }
                               # "w_init": kaiming_normal}
default_block_conv_config = {} #{"w_init": kaiming_normal}
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
             version: str = 'V1',
             v2_linear_block: bool = False,
             avg_in_final_block: bool = False,):

    assert version in ["V1", "V2", "V3"], "version must be either V1 or V2 or V3"

    if type(size) == int:
        init_conv_size = size
        sizes = [[size*i]*4 for i in [1, 2, 4, 8]]
        if v2_linear_block:
            fc_size = 4 * size
        # fc2_size = 2*size
    else:
        init_conv_size = size[0]
        if v2_linear_block:
            fc_size = size[-1]
            # fc2_size = size[-1]
            sizes = size[1:-1]
        else:
            sizes = size[1:]
        # sizes = size[1:-2]
        sizes = [sizes[i:i+4] for i in range(0, 16, 4)]

    resnet_config = {
                    "blocks_per_group": (2, 2, 2, 2),
                    "bottleneck": False,
                    "channels_per_group": sizes,  # typical resnet18 size = 64
                    "use_projection": (False, True, True, True),
                    "bn_config": bn_config
                    }
    default_initial_conv_config["output_channels"] = init_conv_size
    if v2_linear_block:
        default_fc_layer_config["output_size"] = fc_size
        # default_fc2_layer_config["output_size"] = fc2_size

    if version == "V1":
        resnet_block_type = ResnetBlockV1
    elif version == "V2":
        resnet_block_type = ResnetBlockV2
    elif version == "V3":
        resnet_block_type = ResnetBlockV3

    return resnet_model(num_classes=num_classes,
                        activation_fn=activation_fn,
                        initial_conv_config=initial_conv_config,
                        strides=strides,
                        logits_config=logits_config,
                        with_bn=with_bn,
                        resnet_block=resnet_block_type,
                        v2_linear_block=v2_linear_block,
                        avg_in_final_block=avg_in_final_block,
                        **resnet_config)


def resnet50(size: Union[int, Sequence[int]],
             num_classes: int,
             activation_fn: hk.Module = ReluActivationModule,
             logits_config: Optional[Mapping[str, Any]] = default_logits_config,
             initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = default_initial_conv_config,
             strides: Sequence[int] = (1, 2, 2, 2),
             with_bn: bool = True,
             bn_config: dict = base_bn_config,
             version: str = 'V1',
             v2_lin_block: bool = False,
             avg_in_final_block: bool = False,):

    assert version in ["V1", "V2", "V3"], "version must be either V1 or V2 or V3"

    blocks_per_group = [3, 4, 6, 3]
    if type(size) == int:
        init_conv_size = size
        sizes = [[size*(2**i), size*(2**i), 4*size*(2**i)]*blocks_per_group[i] for i in range(4)]  #[1, 2, 4, 8]]
        if v2_lin_block:
            fc_size = 16 * size
    else:
        init_conv_size = size[0]
        if v2_lin_block:
            fc_size = size[-1]
            sizes = size[1:-1]
        else:
            sizes = size[1:]
        sizes = [sizes[i:j] for i, j in ((0, 9), (9, 21), (21, 39), (39, 48))]

    resnet_config = {
                    "blocks_per_group": (3, 4, 6, 3),
                    "bottleneck": True,
                    "channels_per_group": sizes,  # typical resnet50 size = 64 (so 256 in first block after bottleneck)
                    "use_projection": (True, True, True, True),
                    "bn_config": bn_config
                    }
    default_initial_conv_config["output_channels"] = init_conv_size
    if v2_lin_block:
        default_fc_layer_config["output_size"] = fc_size

    if version == "V1":
        resnet_block_type = ResnetBlockV1
    elif version == "V2":
        resnet_block_type = ResnetBlockV2
    elif version == "V3":
        resnet_block_type = ResnetBlockV3

    return resnet_model(num_classes=num_classes,
                        activation_fn=activation_fn,
                        initial_conv_config=initial_conv_config,
                        strides=strides,
                        logits_config=logits_config,
                        with_bn=with_bn,
                        resnet_block=resnet_block_type,
                        v2_linear_block=v2_lin_block,
                        avg_in_final_block=avg_in_final_block,
                        **resnet_config)


##############################
# SrigL models: https://arxiv.org/pdf/2305.02299.pdf
# https://github.com/calgaryml/condensed-sparsity/blob/main/src/rigl_torch/models/resnet.py
##############################
srigl_initial_conv_config = {"kernel_shape": 3,
                               "stride": 1,
                               "with_bias": False,
                               "padding": "SAME",
                               "name": "initial_conv",
                               "w_init": kaiming_normal}


def srigl_resnet18(size: Union[int, Sequence[int]],
             num_classes: int,
             activation_fn: hk.Module = ReluActivationModule,
             logits_config: Optional[Mapping[str, Any]] = default_logits_config,
             initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = srigl_initial_conv_config,
             strides: Sequence[int] = (1, 2, 2, 2),
             with_bn: bool = True,
             version: str = 'V1',
             bn_config: dict = base_bn_config,
             initializer: str = 'kaiming_conv',
             disable_final_pooling: bool = False,  # Solely to test impact of pooling on dead neurons in final conv
             avg_in_final_block: bool = False,
             proj_instead_pool: bool = False,
                   ):
    assert version in ["V1", "V2", "V3"], "version must be either V1 or V2 or V3"

    pool_proj_channels = None
    if type(size) == int:
        init_conv_size = size
        sizes = [[size*i]*4 for i in [1, 2, 4, 8]]
        if proj_instead_pool:
            pool_proj_channels = size*8
    else:
        init_conv_size = size[0]
        if proj_instead_pool:
            sizes = size[1:-1]
            pool_proj_channels = size[-1]
        else:
            sizes = size[1:]
        sizes = [sizes[i:i+4] for i in range(0, 16, 4)]

    if initializer == "kaiming_conv":
        default_block_conv_config['w_init'] = kaiming_normal
        default_initial_conv_config['w_init'] = kaiming_normal
    elif initializer == "hk_default":
        default_logits_config['w_init'] = None
        default_block_conv_config['w_init'] = None
        default_initial_conv_config['w_init'] = None
    elif initializer == "pt_default":
        default_logits_config['w_init'] = pytorch_default_init
        default_block_conv_config['w_init'] = pytorch_default_init
        default_initial_conv_config['w_init'] = pytorch_default_init
        default_logits_config['b_init'] = pytorch_default_init
    else:
        raise ValueError("initializer not supported")

    resnet_config = {
                    "blocks_per_group": (2, 2, 2, 2),
                    "bottleneck": False,
                    "channels_per_group": sizes,  # typical resnet18 size = 64
                    "use_projection": (False, True, True, True),
                    "bn_config": bn_config
                    }
    initial_conv_config["output_channels"] = init_conv_size

    if version == "V1":
        res_block = ResnetBlockV1
    elif version == "V2":
        res_block = ResnetBlockV2
    elif version == "V3":
        res_block = ResnetBlockV3

    return resnet_model(num_classes=num_classes,
                        activation_fn=activation_fn,
                        initial_conv_config=initial_conv_config,
                        strides=strides,
                        logits_config=logits_config,
                        with_bn=with_bn,
                        resnet_block=res_block,
                        v2_linear_block=False,
                        max_pool_layer=False,
                        avg_pool_layer=True,
                        disable_final_pooling=disable_final_pooling,
                        avg_in_final_block=avg_in_final_block,
                        pool_proj_channels=pool_proj_channels,
                        **resnet_config)


def srigl_resnet50(size: Union[int, Sequence[int]],
             num_classes: int,
             activation_fn: hk.Module = ReluActivationModule,
             logits_config: Optional[Mapping[str, Any]] = default_logits_config,
             initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = srigl_initial_conv_config,
             strides: Sequence[int] = (1, 2, 2, 2),
             with_bn: bool = True,
             version: str = 'V1',
             bn_config: dict = base_bn_config,
             avg_in_final_block: bool = False,):
    assert version in ["V1", "V2", "V3"], "version must be either V1 or V2 or V3"

    blocks_per_group = [3, 4, 6, 3]
    if type(size) == int:
        init_conv_size = size
        sizes = [[size*(2**i), size*(2**i), 4*size*(2**i)]*blocks_per_group[i] for i in range(4)]  #[1, 2, 4, 8]]
    else:
        init_conv_size = size[0]
        sizes = size[1:]
        sizes = [sizes[i:j] for i, j in ((0, 9), (9, 21), (21, 39), (39, 48))]

    resnet_config = {
                    "blocks_per_group": (3, 4, 6, 3),
                    "bottleneck": True,
                    "channels_per_group": sizes,  # typical resnet50 size = 64 (so 256 in first block after bottleneck)
                    "use_projection": (True, True, True, True),
                    "bn_config": bn_config
                    }
    initial_conv_config["output_channels"] = init_conv_size

    if version == "V1":
        res_block = ResnetBlockV1
    elif version == "V2":
        res_block = ResnetBlockV2
    elif version == "V3":
        res_block = ResnetBlockV3

    return resnet_model(num_classes=num_classes,
                        activation_fn=activation_fn,
                        initial_conv_config=initial_conv_config,
                        strides=strides,
                        logits_config=logits_config,
                        with_bn=with_bn,
                        resnet_block=res_block,
                        v2_linear_block=False,
                        max_pool_layer=False,
                        avg_pool_layer=True,
                        avg_in_final_block=avg_in_final_block,
                        **resnet_config)

