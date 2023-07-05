""" Small experiment setting to support theoritical assumption made in the paper"""

import copy
import random

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from aim import Run, Figure, Distribution, Image
import time
from datetime import timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, List
from ast import literal_eval
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree

import utils.utils as utl
from utils.utils import build_models
from utils.config import activation_choice, optimizer_choice, dataset_choice, dataset_target_cardinality
from utils.config import regularizer_choice, architecture_choice, lr_scheduler_choice, bn_config_choice
from utils.config import pick_architecture

# Experience name -> for aim logger
exp_name = "minimnist_small_exp"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    record_freq: int = 100
    lr: float = 1e-3
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    train_batch_size: int = 32
    full_batch_size: int = 1000
    optimizer: str = "adam"
    activation: str = "relu"  # Activation function used throughout the model
    dataset: str = "mnist"
    dataset_size: Optional[int] = None  # How many example to keep from training dataset (to quickly overfit)
    normalize_inputs: bool = False  # Subtract mean across channels from inputs and divide by variance
    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
    noise_std: float = 1.0  # std deviation of the normal distribution (mean=0) added to training data
    architecture: str = "mlp_3"
    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
    with_bn: bool = False  # Add batchnorm layers or not in the models
    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
    size: Any = 100
    regularizer: Optional[str] = "None"
    reg_param: float = 1e-4
    wd_param: Optional[float] = None
    init_seed: int = 41
    with_rng_seed: int = 428
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name + "_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:
    run_start_time = time.time()

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config.dataset in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(
        architecture_choice.keys())
    assert exp_config.activation in activation_choice.keys(), "Current activation function available: " + str(
        activation_choice.keys())
    assert exp_config.lr_schedule in lr_scheduler_choice.keys(), "Current lr scheduler function available: " + str(
        lr_scheduler_choice.keys())
    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    assert (not (("adamw" in exp_config.optimizer) and bool(
        exp_config.regularizer))) or bool(
        exp_config.wd_param), "Set wd_param if adamw is used with a regularization loss"
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.lr_decay_steps) == str:
        exp_config.lr_decay_steps = literal_eval(exp_config.lr_decay_steps)

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./MiniMnist_experiments", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    net_config = {}

    if not exp_config.with_bias:
        net_config['with_bias'] = exp_config.with_bias

    if exp_config.with_bn:
        assert exp_config.architecture in pick_architecture(
            with_bn=True).keys(), "Current architectures available with batchnorm: " + str(
            pick_architecture(with_bn=True).keys())
        net_config['bn_config'] = bn_config_choice[exp_config.bn_config]

    # Load the different dataset
    load_data = dataset_choice[exp_config.dataset]
    eval_size = exp_config.full_batch_size
    death_minibatch_size = exp_config.full_batch_size
    train_ds_size, train, train_eval, test_death = load_data(split="train", is_training=True,
                                                             batch_size=exp_config.train_batch_size,
                                                             other_bs=[eval_size, death_minibatch_size],
                                                             cardinality=True,
                                                             augment_dataset=exp_config.augment_dataset,
                                                             normalize=exp_config.normalize_inputs,
                                                             reduced_ds_size=eval_size)
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size,
                                     cardinality=True, augment_dataset=exp_config.augment_dataset,
                                     normalize=exp_config.normalize_inputs)