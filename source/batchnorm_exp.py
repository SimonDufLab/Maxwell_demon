""" This experiment is used to test the following hypothesis: Does batchnorm promote neurons saturation by
replacing moving the death border such that neurons weight are maintained close to it throughout training. Here the
activations value pre-relu will be used as a proxy for the neuron distance to the border"""

import jax
import jax.numpy as jnp
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
from utils.config import optimizer_choice, dataset_choice, dataset_target_cardinality, regularizer_choice, architecture_choice


# Experience name -> for aim logger
exp_name = "batchnorm_exp"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    record_freq: int = 100
    lr: float = 1e-3
    train_batch_size: int = 128
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    dataset: str = "mnist"
    architecture: str = "mlp_3"
    size: Any = 50
    regularizer: Optional[str] = "None"
    reg_param: float = 1e-4
    epsilon_close: float = 0.0  # Relaxing criterion for dead neurons, epsilon-close to relu gate
    init_seed: int = 41
    add_noise: bool = False  # Add Gaussian noise to the gradient signal
    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones
    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
    noise_eta: float = 0.01
    noise_gamma: float = 0.0
    noise_seed: int = 1


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)

# Using tf on CPU for data loading
# tf.config.experimental.set_visible_devices([], "GPU")


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:

    run_start_time = time.time()

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(optimizer_choice.keys())
    assert exp_config.dataset in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(architecture_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.sizes) == str:
        exp_config.sizes = literal_eval(exp_config.sizes)
    if type(exp_config.noise_imp) == str:
        exp_config.noise_imp = literal_eval(exp_config.noise_imp)

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Create pickle directory
    pickle_dir_path = "./logs/metadata/" + exp_name + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    os.makedirs(pickle_dir_path)
    # Dump config file in it as well
    with open(pickle_dir_path+'config.json', 'w') as fp:
        json.dump(OmegaConf.to_container(exp_config), fp, indent=4)

    # Load the different dataset
    load_data = dataset_choice[exp_config.dataset]
    train = load_data(split="train", is_training=True, batch_size=exp_config.train_batch_size)

    eval_size = exp_config.eval_batch_size
    train_size, train_eval = load_data(split="train", is_training=False, batch_size=eval_size, cardinality=True)
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size, cardinality=True)
    dataset_size, test_death = load_data(split="train", is_training=False,
                                         batch_size=exp_config.death_batch_size, cardinality=True)
