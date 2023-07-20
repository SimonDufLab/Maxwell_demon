"""Optax optimizers configured to work with logger"""
import jax
import optax
import jaxpruner
from jax.tree_util import Partial

import utils.utils as utl
import utils.scores as scores
# from utils.utils import load_mnist_torch, load_cifar10_torch, load_fashion_mnist_torch
from utils.utils import load_mnist_tf, load_cifar10_tf, load_fashion_mnist_tf, load_imagenet_tf, load_cifar100_tf
from utils.utils import constant_schedule, cosine_decay, piecewise_constant_schedule, one_cycle_schedule, fix_step_decay
from models.mlp import mlp_3, mlp_3_bn, mlp_3_reg
from models.mlp import mlp_3_act_pre_relu, mlp_3_act_pre_bn, mlp_3_act_post_bn
from models.mlp import mlp_3_dropout
from models.convnet import conv_3_2, conv_3_2_bn, conv_4_2, conv_4_2_bn, conv_6_2, conv_6_2_bn
from models.convnet import conv_4_2_act_pre_relu, conv_4_2_act_pre_bn, conv_4_2_act_post_bn
from models.convnet import conv_4_2_dropout, conv_4_2_ln
from models.vgg16 import vgg16
from models.resnet import resnet18

baseline_pruning_method_choice = {
    "WMP": jaxpruner.MagnitudePruning,
    "GMP": jaxpruner.GlobalMagnitudePruning,
    "LMP": utl.LayerMagnitudePruning,
    "saliency": jaxpruner.SaliencyPruning,
    "STE_magnitude": jaxpruner.SteMagnitudePruning,
    "SET": jaxpruner.SET,
    "RigL": jaxpruner.RigL,
}

optimizer_choice = {
    "adam": optax.adam,
    "adamw": optax.adamw,
    "adamw_cdg": utl.adamw_cdg,
    "sgd": optax.sgd,
    "noisy_sgd": optax.noisy_sgd,
    "momentum9": Partial(optax.sgd, momentum=0.9),
    "nesterov9": Partial(optax.sgd, momentum=0.9, nesterov=True),
    "momentum7": Partial(optax.sgd, momentum=0.7),
    "nesterov7": Partial(optax.sgd, momentum=0.7, nesterov=True)
}

dataset_choice = {
    "mnist": load_mnist_tf,
    # "mnist-torch": load_mnist_torch,
    "fashion mnist": load_fashion_mnist_tf,
    # "fashion mnist-torch": load_fashion_mnist_torch,
    "cifar10": load_cifar10_tf,
    # "cifar10-ffcv": load_cifar10_ffcv,
    # "cifar10-torch": load_cifar10_torch,
    "cifar100": load_cifar100_tf,
    'imagenet': load_imagenet_tf,
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
    'fix_steps': fix_step_decay,
    'piecewise_constant': piecewise_constant_schedule,
    'cosine_decay': cosine_decay,
    'one_cycle': one_cycle_schedule
}

reg_param_scheduler_choice = {
    'one_cycle': optax.cosine_onecycle_schedule
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
    "vgg16": None,  # TODO: Encode non-bn version of vgg16
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
    "vgg16": vgg16,
    "resnet18": resnet18,
    "resnet18_v2": Partial(resnet18, version="V2"),
}

bn_config_choice = {
    "default": {"create_scale": True, "create_offset": True, "decay_rate": 0.9},  # decay was set to 0.999 first
    "no_scale_and_offset": {"create_scale": False, "create_offset": False, "decay_rate": 0.9}
}

activation_choice = {
    "relu": utl.ReluActivationModule,
    "leaky_relu": utl.LeakyReluActivationModule,  # leak = 0.05
    "abs": utl.AbsActivationModule,  # Absolute value as the activation function
    "elu": utl.EluActivationModule,
    "swish": utl.SwishActivationModule,
    "tanh": utl.TanhActivationModule,
    "linear": utl.IdentityActivationModule,
    "threlu": utl.ThreluActivationModule,
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
    "cifar10-ffcv": 10,
    "cifar100": 100,
    "imagenet": 1000,
}

pruning_criterion_choice = {
    "earlycrop": (scores.early_crop_score, scores.test_earlycrop_pruning_step),
    "earlysnap": (scores.snap_score, scores.test_earlycrop_pruning_step),
    "snap": (scores.snap_score, scores.prune_before_training),
}


def pick_architecture(with_dropout=False, with_bn=False):
    assert not (with_dropout and with_bn), "No implementation with both bn and dropout currently"
    if with_dropout:
        return architecture_choice_dropout
    elif with_bn:
        return bn_architecture_choice
    else:
        return architecture_choice
