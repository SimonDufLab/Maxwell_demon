""" Base units used to build the different models with batchnorm layers"""

import haiku as hk
import jax
from typing import Optional

# global batch normalization configuration
base_bn_config = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}


class Base_BN(hk.Module):
    """Simply a BN layer, with is_training option pre-specified"""
    def __init__(
            self,
            is_training: bool,
            bn_config: dict = base_bn_config,
            name: Optional[str] = None):
        super().__init__(name=name)
        self.is_training = is_training
        self.bn = hk.BatchNorm(**bn_config)

    def __call__(self, x):
        x = self.bn(x, is_training=self.is_training)
        return x


class Linear_BN(hk.Module):
    """Create a linear layer followed by bn"""
    def __init__(
            self,
            is_training: bool,
            output_size: int,
            with_bias: bool = True,
            bn_config: dict = base_bn_config,
            name: Optional[str] = None):
        super().__init__(name=name)
        self.is_training = is_training
        self.bn = hk.BatchNorm(**bn_config)
        self.linear = hk.Linear(output_size, with_bias=with_bias)

    def __call__(self, inputs):
        x = self.linear(inputs)
        x = self.bn(x, is_training=self.is_training)
        return x


class Conv_BN(hk.Module):
    """Create a convolutional layer followed by bn"""
    def __init__(
            self,
            is_training: bool,
            output_channels: int,
            kernel_size: int,
            with_bias: bool = True,
            bn_config: dict = base_bn_config,
            name: Optional[str] = None):
        super().__init__(name=name)
        self.is_training = is_training
        self.bn = hk.BatchNorm(**bn_config)
        self.conv = hk.Conv2D(output_channels, kernel_size, with_bias=with_bias)

    def __call__(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x, is_training=self.is_training)
        return x
