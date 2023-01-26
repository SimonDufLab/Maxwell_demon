""" In this experiment, we build a very simple dataset from a sine wave signal, that we intend to draw
very few samples from in order to overfit it easily. We want to check what happen with dead neurons
when we add gaussian noise to the sample every time we draw a minibatch during training. Our hypothese
is that it will prevent memorization and eventually regularize by allowing more neurons to die."""

import copy
import random

import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from aim import Run, Figure, Distribution, Image
import os
import time
from datetime import timedelta
import pickle
import json
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
exp_name = "asymptotic_live_neurons"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    lr: float = 1e-3
    train_batch_size: int = 128
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    activation: str = "relu"  # Activation function used throughout the model
    dataset_size: int = 12  # We want to keep it small to allow overfitting
    dataset_seed: int = 1234  # Random seed to vary the training samples picked
    noise_std: float = 1.0  # std deviation of the normal distribution (mean=0) added to training data
    architecture: str = "mlp_3_reg"
    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
    with_bn: bool = False  # Add batchnorm layers or not in the models
    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
    size: Any = 50
    regularizer: Optional[str] = "None"
    reg_param: float = 1e-4
    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
    init_seed: int = 41
    with_rng_seed: int = 428
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name + "_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:

    run_start_time = time.time()

    assert "reg" in exp_config.architecture, "must be a regression architecture"

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(
        architecture_choice.keys())
    assert exp_config.activation in activation_choice.keys(), "Current activation function available: " + str(
        activation_choice.keys())
    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.epsilon_close) == str:
        exp_config.epsilon_close = literal_eval(exp_config.epsilon_close)

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    net_config = {}

    if not exp_config.with_bias:
        net_config['with_bias'] = exp_config.with_bias

    if exp_config.with_bn:
        assert exp_config.architecture in pick_architecture(
            with_bn=True).keys(), "Current architectures available with batchnorm: " + str(
            pick_architecture(with_bn=True).keys())
        net_config['bn_config'] = bn_config_choice[exp_config.bn_config]

    # Create the dataset, by sampling from a sine wave that span [0,10]
    full_span = jnp.linspace(0, 10, jnp.maximum(1e4, exp_config.dataset_size*10))
    train_key, test_key = jax.random.split(jax.random.PRNGKey(exp_config.dataset_seed))
    train_x = jax.random.choice(train_key, full_span, (exp_config.dataset_size,), replace=False)
    train_y = jnp.sin(train_x)

    test_x = jax.random.choice(test_key, full_span, (exp_config.dataset_size,), replace=False)
    test_y = jnp.sin(test_x)

    # Make a data iterator
    class sine_data_iter():
        def __init__(self, features, targets, batch_size, noise_std, noise_seed):
            self.dataset = [(features[i], targets[i]) for i in range(features.size)]
            self.counter = 0
            self.b = batch_size
            self.noise_std = noise_std
            self.key = jax.random.PRNGKey(noise_seed)
            self.key, _ = jax.random.split(self.key)

        def __iter__(self):
            # shuffle dataset:
            random.shuffle(self.dataset)
            # counter to 0
            self.counter = 0
            return self

        def __next__(self):
            try:
                selection = self.dataset[self.counter:self.counter+self.b]
            except IndexError:
                self.__iter__()
                selection = self.dataset[self.counter:self.counter + self.b]
            self.counter += self.b
            # Apply gaussian noise
            self.key, next_key = jax.random.split(self.key)
            gauss_noise = jax.random.normal(next_key, (self.b,))
            gauss_noise *= self.noise_std
            selection = [(selection[i][0]+gauss_noise[i], selection[i][1]) for i in range(self.b)]
            return selection

    train = sine_data_iter(train_x, train_y, exp_config.train_batch_size, exp_config.noise_std, exp_config.dataset_seed)
    train_eval = sine_data_iter(train_x, train_y, exp_config.eval_batch_size, 0, 0)
    test_death = sine_data_iter(train_x, train_y, exp_config.death_batch_size, 0, 0)
    test_eval = sine_data_iter(test_x, test_y, exp_config.eval_batch_size, 0, 0)

    # Make the network and optimiser
    architecture = pick_architecture()[exp_config.architecture]
    architecture = architecture(exp_config.size, activation_fn=activation_fn, **net_config)
    net = build_models(*architecture)

    opt = optimizer_choice[exp_config.optimizer](exp_config.lr)