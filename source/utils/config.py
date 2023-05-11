"""Optax optimizers configured to work with logger"""
import jax
import optax
import jaxpruner
from jax.tree_util import Partial

import utils.utils
from utils.utils import load_mnist_torch, load_cifar10_torch, load_fashion_mnist_torch, load_cifar100_tf
from utils.utils import load_mnist_tf, load_cifar10_tf, load_fashion_mnist_tf
from utils.utils import constant_schedule, cosine_decay, piecewise_constant_schedule, one_cycle_schedule
from models.mlp import mlp_3, mlp_3_bn, mlp_3_reg
from models.mlp import mlp_3_act_pre_relu, mlp_3_act_pre_bn, mlp_3_act_post_bn
from models.mlp import mlp_3_dropout
from models.convnet import conv_3_2, conv_3_2_bn, conv_4_2, conv_4_2_bn, conv_6_2, conv_6_2_bn
from models.convnet import conv_4_2_act_pre_relu, conv_4_2_act_pre_bn, conv_4_2_act_post_bn
from models.convnet import conv_4_2_dropout, conv_4_2_ln
from models.resnet import resnet18
from utils.utils import identity_fn, threlu

baseline_pruning_method_choice = {
    "WMP": jaxpruner.MagnitudePruning,
    "GMP": jaxpruner.GlobalMagnitudePruning,
    "LMP": utils.utils.LayerMagnitudePruning,
    "saliency": jaxpruner.SaliencyPruning,
    "STE_magnitude": jaxpruner.SteMagnitudePruning,
    "SET": jaxpruner.SET,
    "RigL": jaxpruner.RigL,
}

optimizer_choice = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "adamw_cdg": utils.utils.adamw_cdg,
    "sgd": optax.sgd,
    "noisy_sgd": optax.noisy_sgd,
    "momentum9": Partial(optax.sgd, momentum=0.9),
    "nesterov9": Partial(optax.sgd, momentum=0.9, nesterov=True),
    "momentum7": Partial(optax.sgd, momentum=0.7),
    "nesterov7": Partial(optax.sgd, momentum=0.7, nesterov=True)
}

dataset_choice = {
    "mnist": load_mnist_tf,
    "mnist-torch": load_mnist_torch,
    "fashion mnist": load_fashion_mnist_tf,
    "fashion mnist-torch": load_fashion_mnist_torch,
    "cifar10": load_cifar10_tf,
    "cifar10-torch": load_cifar10_torch,
    "cifar100": load_cifar100_tf,
}

regularizer_choice = (
    "None",
    "cdg_l2",
    "cdg_lasso",
    "l2",
    "lasso",
    "cdg_l2_act",  # l2 regularization applied on the activations, not the weights! (funky, funky ...)
    "cdg_lasso_act"  # lasso regularization applied on the activations
)

lr_scheduler_choice = {
    'None': constant_schedule,
    'piecewise_constant': piecewise_constant_schedule,
    'cosine_decay': cosine_decay,
    'one_cycle': one_cycle_schedule
}

# Return the desired architecture along with a bool indicating if there is a
# is_training flag for this specific model
architecture_choice = {
    "mlp_3": mlp_3,
    "mlp_3_reg": mlp_3_reg,
    "conv_3_2": conv_3_2,
    "conv_4_2": conv_4_2,
    "conv_4_2_ln": conv_4_2_ln,  # TODO: eventually switch to ln_architecture_choice, like bn
    "conv_6_2": conv_6_2,
    "resnet18": Partial(resnet18, with_bn=False),
    "resnet18_v2": Partial(resnet18, with_bn=False, version="V2"),
}

architecture_choice_dropout = {
    "mlp_3": mlp_3_dropout,
    "conv_4_2": conv_4_2_dropout,
}

bn_architecture_choice = {
    "mlp_3": mlp_3_bn,
    "conv_3_2": conv_3_2_bn,
    "conv_4_2": conv_4_2_bn,
    "conv_6_2": conv_6_2_bn,
    "resnet18": resnet18,
    "resnet18_v2": Partial(resnet18, version="V2"),
}

bn_config_choice = {
    "default": {"create_scale": True, "create_offset": True, "decay_rate": 0.9},  # decay was set to 0.999 first
    "no_scale_and_offset": {"create_scale": False, "create_offset": False, "decay_rate": 0.9}
}

activation_choice = {
    "relu": jax.nn.relu,
    "leaky_relu": Partial(jax.nn.leaky_relu, negative_slope=0.05),  # leak = 0.05
    "abs": jax.numpy.abs,  # Absolute value as the activation function
    "elu": jax.nn.elu,
    "swish": jax.nn.swish,
    "tanh": jax.nn.tanh,
    "linear": identity_fn,
    "threlu": threlu,
}

activations_pre_relu = {
    "mlp_3": (mlp_3_act_pre_relu, ),
    "conv_4_2": (conv_4_2_act_pre_relu, )
}

activations_with_bn = {
    "mlp_3": (mlp_3_act_pre_bn, mlp_3_act_post_bn),
    "conv_4_2": (conv_4_2_act_pre_bn, conv_4_2_act_post_bn)
}

dataset_target_cardinality = {  # Hard-encoding the number of classes in given dataset for easy retrieval
    "mnist": 10,
    "fashion mnist": 10,
    "cifar10": 10,
    "cifar100": 100
}


def pick_architecture(with_dropout=False, with_bn=False):
    assert not (with_dropout and with_bn), "No implementation with both bn and dropout currently"
    if with_dropout:
        return architecture_choice_dropout
    elif with_bn:
        return bn_architecture_choice
    else:
        return architecture_choice
