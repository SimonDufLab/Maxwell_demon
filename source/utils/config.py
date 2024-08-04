"""Optax optimizers configured to work with logger"""
import jax
import optax
import jaxpruner
from jax.tree_util import Partial
import jax.numpy as jnp

import utils.utils as utl
import utils.scores as scores
# from utils.utils import load_mnist_torch, load_cifar10_torch, load_fashion_mnist_torch
from utils.utils import load_mnist_tf, load_cifar10_tf, load_fashion_mnist_tf, load_imagenet_tf, load_cifar100_tf
from utils.utils import constant_schedule, cosine_decay, piecewise_constant_schedule, one_cycle_schedule, fix_step_decay
from utils.utils import warmup_cosine_decay, warmup_piecewise_decay_schedule
from models.mlp import mlp_3, mlp_3_bn, mlp_3_reg
from models.mlp import mlp_3_act_pre_relu, mlp_3_act_pre_bn, mlp_3_act_post_bn
from models.mlp import mlp_3_dropout
from models.convnet import conv_3_2, conv_3_2_bn, conv_4_2, conv_4_2_bn, conv_6_2, conv_6_2_bn
from models.convnet import conv_4_2_act_pre_relu, conv_4_2_act_pre_bn, conv_4_2_act_post_bn
from models.convnet import conv_4_2_dropout, conv_4_2_ln
from models.vgg16 import vgg16, conv_2_2
from models.resnet import resnet18, resnet50, srigl_resnet18, srigl_resnet50
from models.vit import vit_b_4, vit_b_16
from models.grokking_model import grok_models
from datasets.grok_datasets import ModDivisonDataset, ModSubtractDataset, ModSumDataset, PermutationGroup, load_grok_ds, n_out_mapping
from datasets.spurious_ds import load_colmnist_tf


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
    "adamw_b2_98": Partial(optax.adamw, b2=0.98),
    "adamw_eps_1e-10": Partial(optax.adamw, eps=1e-10),  # Rework into parser after rebuttal, quick workaround for now
    "adamw_eps_1e-9": Partial(optax.adamw, eps=1e-9),
    "adamw_eps_1e-6": Partial(optax.adamw, eps=1e-6),
    "adamw_eps_1e-4": Partial(optax.adamw, eps=1e-4),
    "adamw_cdg": utl.adamw_cdg,
    "adam_to_momentum": utl.adam_to_momentum,  # Adam on schedule -> become momentum after ~10k steps
    "new_adam_to_momentum": utl.adam_to_momentum_v2,  # Required to know total number of steps
    "new_adamw_to_momentumw": utl.adamw_to_momentumw_v2,  # Required to know total number of steps
    "sgd": optax.sgd,
    "noisy_sgd": optax.noisy_sgd,
    "momentum9": Partial(optax.sgd, momentum=0.9),
    "momentum9w": Partial(utl.sgdw, momentum=0.9),
    "nesterov9": Partial(optax.sgd, momentum=0.9, nesterov=True),
    "momentum7": Partial(optax.sgd, momentum=0.7),
    "nesterov7": Partial(optax.sgd, momentum=0.7, nesterov=True),
    "momentum_loschiwd": Partial(utl.sgd_loschilov_wd, momentum=0.9, cdg=False),
    "momentum_loschiwd_cdg": Partial(utl.sgd_loschilov_wd, momentum=0.9, cdg=True),
    "adam_loschiwd": Partial(utl.adam_loschilov_wd, cdg=False),
    "adam_loschiwd_cdg": Partial(utl.adam_loschilov_wd, cdg=True),
}

dataset_choice = {
    "mnist": load_mnist_tf,
    "color_mnist_0": Partial(load_colmnist_tf, sp_noise_train=0.0, sp_noise_test=0.9, core_noise=0.25),
    "color_mnist_25": Partial(load_colmnist_tf, sp_noise_train=0.25, sp_noise_test=0.9, core_noise=0.25),
    # "mnist-torch": load_mnist_torch,
    "fashion mnist": load_fashion_mnist_tf,
    # "fashion mnist-torch": load_fashion_mnist_torch,
    "cifar10": load_cifar10_tf,
    "cifar10_srigl": Partial(load_cifar10_tf, dataset="cifar10_srigl"),
    # "cifar10-ffcv": load_cifar10_ffcv,
    # "cifar10-torch": load_cifar10_torch,
    "cifar100": load_cifar100_tf,
    'imagenet': load_imagenet_tf,
    'imagenet_vit': Partial(load_imagenet_tf, dataset="imagenet_vit"),
    "mod_division_dataset": Partial(load_grok_ds, dataset=ModDivisonDataset(0.4, p=97, k=5)),
    "mod_subtract_dataset": Partial(load_grok_ds, dataset=ModSubtractDataset(0.4, p=97, k=5)),
    "mod_sum_dataset": Partial(load_grok_ds, dataset=ModSumDataset(0.4, p=97, k=5)),
    "permutation_group_dataset": Partial(load_grok_ds, dataset=PermutationGroup(0.4, p=97, k=5)),
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
    'constant': constant_schedule,
    'fix_steps': fix_step_decay,
    'piecewise_constant': piecewise_constant_schedule,
    'cosine_decay': cosine_decay,
    'warmup_cosine_decay': warmup_cosine_decay,
    'one_cycle': one_cycle_schedule,
    'warmup_piecewise_decay': warmup_piecewise_decay_schedule,
    'step_warmup': utl.step_warmup,  # The very short warmup for grokking experiments
}

reg_param_scheduler_choice = {
    'one_cycle': optax.cosine_onecycle_schedule,
    'warmup': utl.linear_warmup,
    'cosine_decay': Partial(utl.cosine_decay, final_lr=0.0, decay_bounds=None, scaling_factor=None),
    'constant': Partial(constant_schedule, final_lr=None, decay_bounds=None, scaling_factor=None)

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
    "srigl_resnet18": Partial(srigl_resnet18, with_bn=False),
    "srigl_resnet18_proj_instead": Partial(srigl_resnet18, with_bn=False, proj_instead_pool=True),
    "srigl_resnet18_pool_in_block": Partial(srigl_resnet18, with_bn=False, avg_in_final_block=True),
    "srigl_resnet18_v2": Partial(srigl_resnet18, with_bn=False, version='V2'),
    "srigl_resnet18_v3": Partial(srigl_resnet18, with_bn=False, version='V3'),
    "srigl_resnet18_nopool": Partial(srigl_resnet18, with_bn=False, disable_final_pooling=True),
    "srigl_resnet18_hk_init": Partial(srigl_resnet18, with_bn=False, initializer="hk_default"),
    "srigl_resnet18_pt_init": Partial(srigl_resnet18, with_bn=False, initializer="pt_default"),
    "resnet50": Partial(resnet50, with_bn=False),
    "srigl_resnet50": Partial(srigl_resnet50, with_bn=False)
}

architecture_choice_dropout = {
    "mlp_3": mlp_3_dropout,
    "conv_4_2": conv_4_2_dropout,
}

bn_architecture_choice = {
    "mlp_3": Partial(mlp_3, with_bn=True),
    "conv_3_2": conv_3_2_bn,
    "conv_4_2": conv_4_2_bn,
    "conv_6_2": conv_6_2_bn,
    "vgg16": vgg16,
    "conv_2_2": conv_2_2,
    "resnet18": resnet18,
    "resnet19": Partial(resnet18, v2_linear_block=True),
    "srigl_resnet18": srigl_resnet18,
    "srigl_resnet18_proj_instead": Partial(srigl_resnet18, proj_instead_pool=True),
    "srigl_resnet18_pool_in_block": Partial(srigl_resnet18, avg_in_final_block=True),
    "srigl_resnet18_v2": Partial(srigl_resnet18, version='V2'),
    "srigl_resnet18_v3": Partial(srigl_resnet18, version='V3'),
    "srigl_resnet18_nopool": Partial(srigl_resnet18, disable_final_pooling=True),  # Testing purpose, do not use elsewhere
    "srigl_resnet18_hk_init": Partial(srigl_resnet18, initializer="hk_default"),
    "srigl_resnet18_pt_init": Partial(srigl_resnet18, initializer="pt_default"),
    "resnet50": resnet50,
    "srigl_resnet50": srigl_resnet50,
    "vit_b_4": vit_b_4,
    "vit_b_16": vit_b_16,
    "grok_models": grok_models,
}

bn_config_choice = {
    "default": {"create_scale": True, "create_offset": True, "decay_rate": 0.9},  # decay was set to 0.999 first
    "no_scale_and_offset": {"create_scale": False, "create_offset": False, "decay_rate": 0.9},
    "bigger_eps": {"create_scale": True, "create_offset": True, "decay_rate": 0.9, "eps": 1e-3},
    "constant_scale": {"create_scale": False, "create_offset": True, "decay_rate": 0.9, "constant_scale": 10.0},
    "small_scale_init": {"create_scale": True, "create_offset": True, "decay_rate": 0.9,
                         "scale_init": lambda x, y: 0.1*jnp.ones(x, y)},
    "deactivate_small_units": {"create_scale": True, "create_offset": True, "decay_rate": 0.9,
                               "deactivate_small_units": True},
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
    "gelu": utl.GeluActivationModule
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
    "color_mnist_0": 2,
    "color_mnist_25": 2,
    "fashion mnist": 10,
    "cifar10": 10,
    "cifar10_srigl": 10,
    "cifar10-ffcv": 10,
    "cifar100": 100,
    "imagenet": 1000,
    "imagenet_vit": 1000,
    "mod_division_dataset": n_out_mapping["mod_division_dataset"],
    "mod_subtract_dataset": n_out_mapping["mod_subtract_dataset"],
    "mod_sum_dataset": n_out_mapping["mod_sum_dataset"],
    "permutation_group_dataset": n_out_mapping["permutation_group_dataset"],
    # "mod_division_dataset": ModDivisonDataset(0.4, p=97, k=5).n_out,
    # "mod_subtract_dataset": ModSubtractDataset(0.4, p=97, k=5).n_out,
    # "mod_sum_dataset": ModSumDataset(0.4, p=97, k=5).n_out,
    # "permutation_group_dataset": PermutationGroup(0.4, p=97, k=5).n_out,
}

pruning_criterion_choice = {
    "earlycrop": (scores.early_crop_score, scores.test_earlycrop_pruning_step),
    "earlysnap": (scores.snap_score, scores.test_earlycrop_pruning_step),
    "snap": (scores.snap_score, scores.prune_before_training),
    "crop-it": (scores.early_crop_score, scores.prune_before_training)
}


def pick_architecture(with_dropout=False, with_bn=False):
    assert not (with_dropout and with_bn), "No implementation with both bn and dropout currently"
    if with_dropout:
        return architecture_choice_dropout
    elif with_bn:
        return bn_architecture_choice
    else:
        return architecture_choice
