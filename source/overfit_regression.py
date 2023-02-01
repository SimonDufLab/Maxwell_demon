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
exp_name = "overfit_regression"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    record_freq: int = 100
    pruning_freq: int = 2000
    drawing_freq: int = 20000
    final_smoothing: int = 0  # Remove noise for n final smoothing steps
    lr: float = 1e-3
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    train_batch_size: int = 12
    eval_batch_size: int = 100
    death_batch_size: int = 12
    optimizer: str = "adam"
    activation: str = "relu"  # Activation function used throughout the model
    dataset_size: int = 12  # We want to keep it small to allow overfitting
    eval_dataset_size: int = 100  # Evaluate on more points
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
    assert exp_config.lr_schedule in lr_scheduler_choice.keys(), "Current lr scheduler function available: " + str(
        lr_scheduler_choice.keys())
    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.epsilon_close) == str:
        exp_config.epsilon_close = literal_eval(exp_config.epsilon_close)

    activation_fn = activation_choice[exp_config.activation]

    exp_config.train_batch_size = min(exp_config.train_batch_size, exp_config.dataset_size)  # Train bs can't be bigger than dataset size
    exp_config.death_batch_size = min(exp_config.death_batch_size, exp_config.dataset_size)  # same as above

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
    full_span = jnp.linspace(0, 10, jnp.maximum(1e4, exp_config.dataset_size*10).astype(int))
    train_key, test_key = jax.random.split(jax.random.PRNGKey(exp_config.dataset_seed))
    # train_x = jax.random.choice(train_key, full_span, (exp_config.dataset_size, 1),
    #                             replace=False)  # Randomly drawn train_x
    train_x = jnp.linspace(0, 10, exp_config.dataset_size).astype(int).reshape((-1, 1))  # Evenly spaced train_x
    train_y = jnp.sin(train_x)

    test_x = jax.random.choice(test_key, full_span, (exp_config.eval_dataset_size, 1), replace=False)
    test_y = jnp.sin(test_x)

    full_span_ = full_span.reshape((-1, 1))  # For plotting
    full_span_ = full_span_, jnp.zeros(full_span_.shape)

    # Make a data iterator
    class sine_data_iter:
        def __init__(self, features, targets, batch_size, noise_std, noise_seed):
            # self.dataset = [(features[i], targets[i]) for i in range(features.size)]
            self.features = features
            self.targets = targets
            self.len = self.features.size
            self.counter = 0
            self.b = batch_size
            self.noise_std = noise_std
            self.key = jax.random.PRNGKey(noise_seed)
            self.key, _ = jax.random.split(self.key)

        def __iter__(self):
            # shuffle dataset:
            self.key, next_key = jax.random.split(self.key)
            self.features = jax.random.permutation(next_key, self.features)
            self.targets = jax.random.permutation(next_key, self.targets)  # Same key, same permutation
            # counter to 0
            self.counter = 0
            return self

        def __next__(self):
            if self.counter + self.b <= self.len:
                selected_features = self.features[self.counter:self.counter+self.b]
                selected_targets = self.targets[self.counter:self.counter+self.b]
            else:
                self.__iter__()
                selected_features = self.features[self.counter:self.counter + self.b]
                selected_targets = self.targets[self.counter:self.counter + self.b]
            self.counter += self.b
            # Apply gaussian noise
            self.key, next_key = jax.random.split(self.key)
            gauss_noise = jax.random.normal(next_key, (self.b, 1))
            gauss_noise *= self.noise_std
            # selection = [(selection[i][0]+gauss_noise[i], selection[i][1]) for i in range(self.b)]
            # print(selected_features)
            # print(self.counter)
            selected_features = selected_features + gauss_noise
            return jnp.stack(selected_features), jnp.stack(selected_targets)  # Create the batched array

    train = sine_data_iter(train_x, train_y, exp_config.train_batch_size, exp_config.noise_std, exp_config.dataset_seed)
    train_eval = sine_data_iter(train_x, train_y, exp_config.train_batch_size, 0, 0)
    test_death = sine_data_iter(train_x, train_y, exp_config.death_batch_size, 0, 0)
    test_eval = sine_data_iter(test_x, test_y, exp_config.eval_batch_size, 0, 0)

    # Make the network and optimiser
    architecture = pick_architecture()[exp_config.architecture]
    architecture = architecture(exp_config.size, activation_fn=activation_fn, **net_config)
    net = build_models(*architecture)

    lr_schedule = lr_scheduler_choice[exp_config.lr_schedule](exp_config.training_steps, exp_config.lr,
                                                              exp_config.final_lr, exp_config.lr_decay_steps)
    opt = optimizer_choice[exp_config.optimizer](lr_schedule)

# Set training/monitoring functions
    loss = utl.mse_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param)
    test_loss_fn = utl.mse_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                        is_training=False)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt)
    death_check_fn = utl.death_check_given_model(net)
    scan_len = exp_config.dataset_size // exp_config.death_batch_size
    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
        utl.death_check_given_model(net, with_activations=True), scan_len, True)

    params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
    opt_state = opt.init(params)

    starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, exp_config.size)
    total_neurons, total_per_layer = starting_neurons, starting_per_layer

    for step in range(exp_config.training_steps+exp_config.final_smoothing):
        if step % exp_config.record_freq == 0:
            train_loss = test_loss_fn(params, state, next(train_eval))
            test_loss = test_loss_fn(params, state, next(test_eval))
            # Periodically print classification accuracy on train & test sets.
            if step % exp_config.report_freq == 0:
                print(f"[Step {step}] Train / Test Loss: {train_loss:.3f} / {test_loss:.3f}")
            test_death_batch = next(test_death)
            dead_neurons = death_check_fn(params, state, test_death_batch)
            # Record some metrics
            dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
            exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step)
            exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step)
            if exp_config.epsilon_close:
                for eps in exp_config.epsilon_close:
                    eps_dead_neurons = death_check_fn(params, state, test_death_batch, eps)
                    eps_dead_neurons_count, _ = utl.count_dead_neurons(eps_dead_neurons)
                    exp_run.track(jax.device_get(eps_dead_neurons_count),
                                  name="Quasi-dead neurons", step=step,
                                  context={"epsilon": eps})
                    exp_run.track(jax.device_get(total_neurons - eps_dead_neurons_count),
                                  name="Quasi-live neurons", step=step,
                                  context={"epsilon": eps})
            exp_run.track(jax.device_get(train_loss), name="Train loss", step=step)
            exp_run.track(jax.device_get(test_loss), name="Test loss", step=step)

        if step % exp_config.pruning_freq == 0:
            dead_neurons = scan_death_check_fn(params, state, test_death)
            dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)
            exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                          step=step)
            exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                          name="Live neurons; whole training dataset",
                          step=step)
            if exp_config.epsilon_close:
                for eps in exp_config.epsilon_close:
                    eps_dead_neurons = scan_death_check_fn(params, state, test_death, eps)
                    eps_dead_neurons_count, eps_dead_per_layers = utl.count_dead_neurons(eps_dead_neurons)
                    exp_run.track(jax.device_get(eps_dead_neurons_count),
                                  name="Quasi-dead neurons; whole training dataset",
                                  step=step,
                                  context={"epsilon": eps})
                    exp_run.track(jax.device_get(total_neurons - eps_dead_neurons_count),
                                  name="Quasi-live neurons; whole training dataset",
                                  step=step,
                                  context={"epsilon": eps})
            for i, layer_dead in enumerate(dead_per_layers):
                total_neuron_in_layer = total_per_layer[i]
                exp_run.track(jax.device_get(layer_dead),
                              name=f"Dead neurons in layer {i}; whole training dataset", step=step)
                exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                              name=f"Live neurons in layer {i}; whole training dataset", step=step)
            del dead_per_layers

        if step % exp_config.drawing_freq == 0:
            # Record a plot of the learned function to expose overfiting
            fig = plt.figure(figsize=(15, 10))
            plt.plot(jax.device_get(full_span), jax.device_get(jnp.sin(full_span)), label="True function (sin)",
                     linewidth=4)
            learned_curve, _ = net.apply(params, state, full_span_, return_activations=False, is_training=False)
            plt.plot(jax.device_get(full_span), jax.device_get(learned_curve.flatten()), label="Learned function",
                     linewidth=4)
            plt.scatter(jax.device_get(train_x.flatten()), jax.device_get(train_y.flatten()), c="red",
                        label="Training points", linewidth=4)
            # plt.xlabel("Number of neurons in NN", fontsize=16)
            # plt.ylabel("Live neurons at end of training", fontsize=16)
            # plt.title("Learned function", fontweight='bold', fontsize=20)
            plt.legend(prop={'size': 16})
            aim_img = Image(fig)
            exp_run.track(aim_img, name="Overfiting curve; img", step=step)
            plt.close(fig)

        # Train step over single batch
        params, state, opt_state = update_fn(params, state, opt_state, next(train))

        if step >= exp_config.training_steps:  # Remove noise from training ds for final smoothing phase
            train = sine_data_iter(train_x, train_y, exp_config.train_batch_size, 0, 0)

    activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
    activations_max, activations_mean, activations_count = activations_data
    activations_max, _ = ravel_pytree(activations_max)
    activations_max = jax.device_get(activations_max)
    activations_mean, _ = ravel_pytree(activations_mean)
    activations_mean = jax.device_get(activations_mean)
    activations_count, _ = ravel_pytree(activations_count)
    activations_count = jax.device_get(activations_count)
    activations_max_dist = Distribution(activations_max, bin_count=100)
    exp_run.track(activations_max_dist, name='Maximum activation distribution after convergence', step=0)
    activations_mean_dist = Distribution(activations_mean, bin_count=100)
    exp_run.track(activations_mean_dist, name='Mean activation distribution after convergence', step=0)
    activations_count_dist = Distribution(activations_count, bin_count=50)
    exp_run.track(activations_count_dist, name='Activation count per neuron after convergence', step=0)

    # # Record a plot of the learned function to expose overfiting
    # fig = plt.figure(figsize=(15, 10))
    # plt.plot(jax.device_get(full_span), jax.device_get(jnp.sin(full_span)), label="True function (sin)", linewidth=4)
    # full_span_ = full_span.reshape((-1, 1))
    # full_span_ = full_span_, jnp.zeros(full_span_.shape)
    # learned_curve, _ = net.apply(params, state, full_span_, return_activations=False, is_training=False)
    # plt.plot(jax.device_get(full_span), jax.device_get(learned_curve.flatten()), label="Learned function", linewidth=4)
    # plt.scatter(jax.device_get(train_x.flatten()), jax.device_get(train_y.flatten()), c="red", label="Training points", linewidth=4)
    # # plt.xlabel("Number of neurons in NN", fontsize=16)
    # # plt.ylabel("Live neurons at end of training", fontsize=16)
    # # plt.title("Learned function", fontweight='bold', fontsize=20)
    # plt.legend(prop={'size': 16})
    # aim_img = Image(fig)
    # exp_run.track(aim_img, name="Overfiting curve; img", step=0)

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
