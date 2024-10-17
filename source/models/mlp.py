""" Models definition for MLP lie architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.tree_util import Partial
from jax import lax
from jax.nn import relu, tanh

from utils.utils import ReluActivationModule, IdentityActivationModule
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from haiku._src.base import current_name
from models.bn_base_unit import Linear_BN, Base_BN
from models.dropout_units import Base_Dropout

FloatStrOrBool = Union[str, float, bool]


class MultivariateNormalInitializer(hk.initializers.Initializer):
    def __call__(self, shape, dtype=jnp.float32):
        if len(shape) != 2:
            raise ValueError("Shape should be 2D (batch_size, D) for multivariate normal.")

        batch_size, D = shape
        mean = jnp.zeros(D)
        cov = jnp.eye(D)  # Identity covariance matrix for D dimensions

        return jax.random.multivariate_normal(
            hk.next_rng_key(), mean=mean, cov=cov, shape=(batch_size,), dtype=dtype
        )

class SoftmaxTLinear(hk.Module):
    """Custom Linear module with temperature-adjusted softmax applied element-wise per output neuron."""

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        temperature: float = 1.0,  # Temperature parameter
        name: Optional[str] = None,
    ):
        """Constructs the CustomLinear module.

        Args:
          output_size: Output dimensionality.
          with_bias: Whether to add a bias to the output.
          w_init: Optional initializer for weights.
          b_init: Optional initializer for bias.
          T: Temperature for the softmax scaling.
          name: Name of the module.
        """
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.T = temperature  # Temperature for the softmax operation

    def __call__(
        self,
        inputs: jax.Array,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jax.Array:
        """Computes a custom linear transform of the input using a temperature-adjusted softmax."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype
        n = input_size  # Dimension of the input vector

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        # Calculate z_i = w_i * x_i for each element i for all outputs
        # Produces a (batch_size, input_size, output_size) tensor when input is batched
        z = jnp.einsum('bi,io->bio', inputs, w)

        # Apply temperature-adjusted softmax within each neuron
        exp_z = jnp.exp(jnp.abs(z) / self.T)
        softmax_z = exp_z / jnp.sum(exp_z, axis=1, keepdims=True)

        # Multiply z by its softmax, sum over input dimension
        output = n * jnp.sum(z * softmax_z, axis=1)

        if self.with_bias:
            b = hk.get_parameter("b", [output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, output.shape)
            output += b

        return output

class LinearLayer(hk.Module):
    def __init__(
            self,
            output_size: int,
            activation_fn: Optional[hk.Module],
            with_bias: bool,
            with_bn: bool = False,
            is_training: bool = True,
            fc_config: Optional[Mapping[str, FloatStrOrBool]] = {},
            bn_config: Optional[Mapping[str, FloatStrOrBool]] = {},
            temperature: Optional[float] = None,
            fourier_transform: Optional[bool] = False,
            name: Optional[str] = None,
            parent: Optional[hk.Module] = None):
        super().__init__(name=name)
        self.activation_mapping = {}
        self.bundle_name = current_name()
        self.fourier_transform = fourier_transform
        self.with_bias = with_bias
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.with_bn = with_bn
        if temperature:
            self.fc_layer = SoftmaxTLinear(output_size, with_bias=with_bias, temperature=temperature, **fc_config)
        else:
            self.fc_layer = hk.Linear(output_size, with_bias=with_bias, **fc_config)
        if with_bn:
            self.bn_layer = Base_BN(is_training=is_training, bn_config=bn_config)
        if activation_fn:
            self.activation_layer = activation_fn()
        else:
            self.activation_layer = None

    def __call__(self, x):
        block_name = self.bundle_name + "/~/"
        x = jax.vmap(jnp.ravel, in_axes=0)(x)  # flatten
        d = jnp.shape(x)[-1]
        x = self.fc_layer(x)
        if self.fourier_transform:
            if self.with_bias:
                x = jnp.sqrt(2/d)  * jnp.cos(x)
            else:
                x = jnp.concatenate([jnp.sqrt(1/d) * jnp.cos(x), jnp.sqrt(1/d) * jnp.sin(x)], axis=-1)
        if self.with_bn:
            x = self.bn_layer(x)
        if self.activation_layer:
            x = self.activation_layer(x)
            activation_name = block_name + self.activation_layer.name
            self.activation_mapping[activation_name] = {"preceding": None,
                                                        "following": activation_name}
        else:
            activation_name = None

        fc_name = block_name + self.fc_layer.name
        if self.fourier_transform and type(activation_name) is str and not self.with_bias:
            self.activation_mapping[fc_name] = {"preceding": self.preceding_activation_name,
                                                "following": activation_name + '_CONCATENATED_FLAG'}
        else:
            self.activation_mapping[fc_name] = {"preceding": self.preceding_activation_name,
                                                "following": activation_name}

        if self.with_bn:
            bn_name = block_name + self.bn_layer.name
            self.activation_mapping[bn_name] = {"preceding": None,
                                                "following": activation_name}

        self.last_act_name = activation_name

        return x

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name


def mlp_5(sizes, number_classes, activation_fn=ReluActivationModule, with_bias=True, with_bn=False, bn_config={},
          temperature=None, fourier_transform=False, tanh_head=False):
    """ Build a MLP with 4 hidden layers inspired by popular LeNet, but with varying number of hidden units"""
    act = activation_fn

    def _tanh():
        return lambda x: tanh(2*x)

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*2, sizes*4, sizes*2]

    if fourier_transform:
        fc_config = {"w_init":MultivariateNormalInitializer(), "b_init":hk.initializers.RandomUniform(minval=0.0, maxval=2 * jnp.pi)}
    else:
        fc_config={}


    layer_1 = [Partial(LinearLayer, sizes[0], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act, temperature=temperature, fourier_transform=fourier_transform, fc_config=fc_config, name='init')]  # hk.Flatten, in LinearLayer
    layer_2 = [Partial(LinearLayer, sizes[1], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act, temperature=temperature,  fourier_transform=fourier_transform, fc_config=fc_config)]
    layer_3 = [
        Partial(LinearLayer, sizes[2], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act,
                temperature=temperature, fourier_transform=fourier_transform, fc_config=fc_config)]
    layer_4 = [
        Partial(LinearLayer, sizes[3], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act,
                temperature=temperature, fourier_transform=fourier_transform, fc_config=fc_config)]
    if tanh_head:
        layer_5 = [Partial(LinearLayer, number_classes, with_bias=with_bias, activation_fn=_tanh, temperature=temperature, name='logits')]
    else:
        layer_5 = [Partial(LinearLayer, number_classes, with_bias=with_bias, activation_fn=None, temperature=temperature, name='logits')]

    return [layer_1, layer_2, layer_3, layer_4, layer_5],


def mlp_3(sizes, number_classes, activation_fn=ReluActivationModule, with_bias=True, with_bn=False, bn_config={},
          temperature=None, fourier_transform=False, tanh_head=False):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units"""
    act = activation_fn

    def _tanh():
        return lambda x: tanh(2*x)

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [Partial(LinearLayer, sizes[0], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act, temperature=temperature, fourier_transform=fourier_transform, name='init')]  # hk.Flatten, in LinearLayer
    layer_2 = [Partial(LinearLayer, sizes[1], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act, temperature=temperature, fourier_transform=fourier_transform)]
    if tanh_head:
        layer_3 = [Partial(LinearLayer, number_classes, with_bias=with_bias, activation_fn=_tanh, temperature=temperature, name='logits')]
    else:
        layer_3 = [Partial(LinearLayer, number_classes, with_bias=with_bias, activation_fn=None, temperature=temperature, name='logits')]

    return [layer_1, layer_2, layer_3],

def mlp_2(sizes, number_classes, activation_fn=ReluActivationModule, with_bias=True, with_bn=False, bn_config={},
          temperature=None):
    """ Build a MLP with a single layer before the classification head"""
    act = activation_fn

    if type(sizes) == int:  # For compatibility with fn that expect a list of layer sizes
        sizes = [sizes,]

    layer_1 = [Partial(LinearLayer, sizes[0], with_bias=with_bias, with_bn=with_bn, bn_config=bn_config, activation_fn=act, temperature=temperature, name='init')]  # hk.Flatten, in LinearLayer
    layer_2 = [Partial(LinearLayer, number_classes, with_bias=with_bias, activation_fn=None, temperature=temperature, name='logits')]

    return [layer_1, layer_2],


def mlp_3_dropout(sizes, number_classes, activation_fn=ReluActivationModule, dropout_rate=0, with_bias=True):
    """ Dropout version of mlp_3 model"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    train_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    train_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act, Partial(Base_Dropout, dropout_rate)]
    layer_3 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act, Partial(Base_Dropout, 0)]  # dropout is zero at eval
    test_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act, Partial(Base_Dropout, 0)]

    return [train_layer_1, train_layer_2, layer_3], [test_layer_1, test_layer_2, layer_3]


def mlp_3_bn(sizes, number_classes, bn_config, activation_fn=ReluActivationModule, with_bias=True):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units,
    , with BN added after every layer apart from the final one"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    train_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), Partial(Base_BN, True, bn_config), act]
    train_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), Partial(Base_BN, True, bn_config), act]
    layer_3 = [Partial(hk.Linear, number_classes, with_bias=with_bias)]

    test_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), Partial(Base_BN, False, bn_config), act]
    test_layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), Partial(Base_BN, False, bn_config), act]

    return [train_layer_1, train_layer_2, layer_3], [test_layer_1, test_layer_2, layer_3]

##############################
# Batchnorm activations architecture  # TODO: Poor solution; should redesign fully how models are built

# The architecture below are to retrieve activations value pre-relu and pre-bn for visualization purposes;
# to build histograms of those at some given time step
##############################


def mlp_3_act_pre_relu(sizes, number_classes, activation_fn=ReluActivationModule):
    """ MLP with 2 hidden units used only to retrieve activations value pre-relu"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0])]
    layer_2 = [act, Partial(hk.Linear, sizes[1])]
    layer_3 = [act, Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3],


def mlp_3_act_pre_bn(sizes, number_classes, bn_config, activation_fn=ReluActivationModule):
    """ MLP with 2 hidden units used only to retrieve activations value pre-bacthnorm"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0])]
    train_layer_2 = [Partial(Base_BN, True, bn_config), act, Partial(hk.Linear, sizes[1])]
    train_layer_3 = [Partial(Base_BN, True, bn_config), act, Partial(hk.Linear, number_classes)]

    test_layer_2 = [Partial(Base_BN, False, bn_config), act, Partial(hk.Linear, sizes[1])]
    test_layer_3 = [Partial(Base_BN, False, bn_config), act, Partial(hk.Linear, number_classes)]

    return [layer_1, train_layer_2, train_layer_3], [layer_1, test_layer_2, test_layer_3]


def mlp_3_act_post_bn(sizes, number_classes, bn_config, activation_fn=ReluActivationModule):
    """ MLP with 2 hidden units used only to retrieve activations value post-bacthnorm"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    train_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0]), Partial(Base_BN, True, bn_config)]
    train_layer_2 = [act, Partial(hk.Linear, sizes[1]), Partial(Base_BN, True, bn_config)]
    layer_3 = [act, Partial(hk.Linear, number_classes)]

    test_layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0]), Partial(Base_BN, False, bn_config)]
    test_layer_2 = [act, Partial(hk.Linear, sizes[1]), Partial(Base_BN, False, bn_config)]

    return [train_layer_1, train_layer_2, layer_3], [test_layer_1, test_layer_2, layer_3]


##############################
# Architecture for regression (scalar outputs)
##############################
def mlp_3_reg(sizes, activation_fn=ReluActivationModule, with_bias=True):
    """ Build a MLP with 2 hidden layers similar to popular LeNet
    Designed for regression tasks: 1 output"""
    act = activation_fn

    if type(sizes) == int:  # Size can be specified with 1 arg, an int
        sizes = [sizes, sizes*3]

    layer_1 = [hk.Flatten, Partial(hk.Linear, sizes[0], with_bias=with_bias), act]
    layer_2 = [Partial(hk.Linear, sizes[1], with_bias=with_bias), act]
    layer_3 = [Partial(hk.Linear, 1, with_bias=with_bias)]

    return [layer_1, layer_2, layer_3],
