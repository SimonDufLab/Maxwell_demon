""" Base units used to build the different models with batchnorm layers"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional

# global batch normalization configuration
base_bn_config = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}


class Base_BN:#(hk.Module):
    """Simply a BN layer, with is_training option pre-specified"""
    def __init__(
            self,
            is_training: bool,
            bn_config: dict = base_bn_config,
            name: Optional[str] = None):
        # super().__init__(name=name)
        self.name = name
        self.is_training = is_training
        self.bn = hk.BatchNorm(name=name, **bn_config)

    def __call__(self, x):
        x = self.bn(x, is_training=self.is_training)
        return x


class Linear_BN(hk.Module):
    """Create a linear layer followed by bn"""
    def __init__(
            self,
            is_training: bool,
            output_size: int,
            activation_fn: hk.Module,
            bn_config: dict = base_bn_config,
            with_bias: bool = True,
            name: Optional[str] = "linBN",
            preceding_activation_name: Optional[str] = None,):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name
        self.is_training = is_training
        self.bn = hk.BatchNorm(name="lin_bn", **bn_config)
        self.linear = hk.Linear(output_size, with_bias=with_bias)
        self.activation_fn = activation_fn()

    def __call__(self, inputs):
        block_name = self.name + "/~/"
        x = jax.vmap(jnp.ravel, in_axes=0)(inputs)  # flatten if need be
        x = self.linear(x)
        x = self.bn(x, is_training=self.is_training)
        x = self.activation_fn(x)

        lin_name = block_name + self.linear.name
        self.activation_mapping[lin_name] = {"preceding": self.preceding_activation_name,
                                             "following": block_name + self.activation_fn.name}
        self.activation_mapping[block_name + self.activation_fn.name] = {"preceding": None,
                                                                         "following": block_name + self.activation_fn.name}
        bn_name = block_name + self.bn.name
        self.activation_mapping[bn_name] = {"preceding": None,
                                            "following": block_name + self.activation_fn.name}
        self.last_act_name = block_name + self.activation_fn.name

        return x

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name


class Conv_BN(hk.Module):
    """Create a convolutional layer followed by bn"""
    def __init__(
            self,
            is_training: bool,
            output_channels: int,
            kernel_size: int,
            activation_fn: hk.Module,
            bn_config: dict = base_bn_config,
            with_bias: bool = True,
            name: Optional[str] = "convBN",
            preceding_activation_name: Optional[str] = None,):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name
        self.is_training = is_training
        self.bn = hk.BatchNorm(name="cv_bn", **bn_config)
        self.conv = hk.Conv2D(output_channels, kernel_size, with_bias=with_bias)
        self.activation_fn = activation_fn()

    def __call__(self, inputs):
        block_name = self.name + "/~/"
        x = self.conv(inputs)
        x = self.bn(x, is_training=self.is_training)
        x = self.activation_fn(x)

        conv_name = block_name + self.conv.name
        self.activation_mapping[conv_name] = {"preceding": self.preceding_activation_name,
                                              "following": block_name + self.activation_fn.name}
        self.activation_mapping[block_name + self.activation_fn.name] = {"preceding": None,
                                                                         "following": block_name + self.activation_fn.name}
        bn_name = block_name + self.bn.name
        self.activation_mapping[bn_name] = {"preceding": None,
                                            "following": block_name + self.activation_fn.name}
        self.last_act_name = block_name + self.activation_fn.name

        return x

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name
