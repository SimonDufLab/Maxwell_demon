"""This experiment tries to verify the following hypothesis: what's the relative importance of noise
for pruning neurons throughout training? As a partial answer, we train a network in a setting that promotes
 neuron's death to identify a subnetwork. After initial training, we reinitialize weights and retrain again in a
 less noisy setting (high bs, low lr) a non-pruned version and a pruned version. We expect much less neurons death
  for the unpruned version and better performance in pruned version in overfitting regime. We want to study the
  performance gap provided by the implicit noise pruning effect"""
import copy

import jax
import jax.numpy as jnp

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

from jax.flatten_util import ravel_pytree

import utils.utils as utl
from utils.utils import build_models
from utils.config import activation_choice, optimizer_choice, dataset_choice, dataset_target_cardinality
from utils.config import regularizer_choice, architecture_choice, lr_scheduler_choice
from utils.config import pick_architecture

# Experience name -> for aim logger
exp_name = "prune_and_reinit_exp"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    record_freq: int = 100
    full_ds_eval_freq: int = 2000
    pruning_cycles: int = 1
    lr: float = 1e-3
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    train_batch_size: int = 128
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    activation: str = "relu"  # Activation function used throughout the model
    dataset: str = "mnist"
    architecture: str = "mlp_3"
    size: Any = 50
    regularizer: Optional[str] = "cdg_l2"
    reg_param: float = 1e-4
    init_seed: int = 41
    dynamic_pruning: bool = False
    dropout_rate: float = 0
    with_rng_seed: int = 428
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)


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

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.sizes) == str:
        exp_config.sizes = literal_eval(exp_config.sizes)
    if type(exp_config.noise_imp) == str:
        exp_config.noise_imp = literal_eval(exp_config.noise_imp)

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Experiments with dropout
    with_dropout = exp_config.dropout_rate > 0
    if with_dropout:
        dropout_key = jax.random.PRNGKey(exp_config.with_rng_seed)
        assert exp_config.architecture in pick_architecture(
            True).keys(), "Current architectures available with dropout: " + str(
            pick_architecture(True).keys())
        drop_config = {"dropout_rate": exp_config.dropout_rate}
    else:
        drop_config = {}

    # Load the different dataset
    load_data = dataset_choice[exp_config.dataset]
    eval_size = exp_config.eval_batch_size
    death_minibatch_size = exp_config.death_batch_size
    train_ds_size, train, train_eval, test_death = load_data(split="train", is_training=True,
                                                             batch_size=exp_config.train_batch_size,
                                                             other_bs=[eval_size, death_minibatch_size],
                                                             cardinality=True)
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size, cardinality=True)

    # Make the network and optimiser
    architecture = pick_architecture(with_dropout)[exp_config.architecture]
    classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
    size = exp_config.size
    architecture = architecture(size, classes, activation_fn=activation_fn, **drop_config)
    net = build_models(*architecture, with_dropout=with_dropout)
    lr_schedule = lr_scheduler_choice[exp_config.lr_schedule](exp_config.training_steps, exp_config.lr,
                                                              exp_config.final_lr, exp_config.lr_decay_steps)
    opt = optimizer_choice[exp_config.optimizer](lr_schedule)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                   classes=classes, with_dropout=with_dropout)
    test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                           classes=classes, is_training=False, with_dropout=with_dropout)
    accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                    exp_config.noise_live_only, with_dropout=with_dropout)
    death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
    scan_len = train_ds_size // death_minibatch_size
    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
        utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_ds_size // eval_size)

    # Initialize
    params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
    opt_state = opt.init(params)
    initial_params = copy.deepcopy(params)  # We need to keep a copy of the initial params for later reset
    initial_state = copy.deepcopy(state)

    starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, size)
    total_neurons, total_per_layer = starting_neurons, starting_per_layer

    pruning_mask_sequence = []

    for step in range(exp_config.training_steps):
        if step % exp_config.record_freq == 0:
            train_loss = test_loss_fn(params, state, next(train_eval))
            train_accuracy = accuracy_fn(params, state, next(train_eval))
            test_accuracy = accuracy_fn(params, state, next(test_eval))
            test_loss = test_loss_fn(params, state, next(test_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
            # Periodically print classification accuracy on train & test sets.
            if step % exp_config.report_freq == 0:
                print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                      f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
            dead_neurons = death_check_fn(params, state, next(test_death))
            # Record some metrics
            dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
            exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                          context={"net size": utl.size_to_string(size)})
            exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step,
                          context={"net size": utl.size_to_string(size)})
            exp_run.track(test_accuracy, name="Test accuracy", step=step,
                          context={"net size": utl.size_to_string(size)})
            exp_run.track(train_accuracy, name="Train accuracy", step=step,
                          context={"net size": utl.size_to_string(size)})
            exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                          context={"net size": utl.size_to_string(size)})
            exp_run.track(jax.device_get(test_loss), name="Test loss", step=step,
                          context={"net size": utl.size_to_string(size)})

        if step % exp_config.full_ds_eval_freq == 0:
            dead_neurons = scan_death_check_fn(params, state, test_death)
            dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)

            exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                          step=step,
                          context={"net size": utl.size_to_string(size)})
            exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                          name="Live neurons; whole training dataset",
                          step=step,
                          context={"net size": utl.size_to_string(size)})
            for i, layer_dead in enumerate(dead_per_layers):
                total_neuron_in_layer = total_per_layer[i]
                exp_run.track(jax.device_get(layer_dead),
                              name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                              name=f"Live neurons in layer {i}; whole training dataset", step=step,
                              context={"net size": utl.size_to_string(size)})
            del dead_per_layers
            del dead_neurons  # Freeing memory
            train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, state, train_eval))
            exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                          step=step,
                          context={"net size": utl.size_to_string(size)})

        # Train step over single batch
        if with_dropout:
            params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train),
                                                              dropout_key)
        else:
            params, state, opt_state = update_fn(params, state, opt_state, next(train))

    # Making sure compiled fn cache was cleared
    loss.clear_cache()
    test_loss_fn.clear_cache()
    accuracy_fn.clear_cache()
    update_fn.clear_cache()
    death_check_fn.clear_cache()

    # Print running time
    print()
    print(f"Running time for run before pruning cycles: " + str(timedelta(seconds=time.time() - run_start_time)))
    print("----------------------------------------------")
    print()

    for i in range(1, exp_config.pruning_cycles):
        pruning_mask = None  # TODO: Recover pruning mask
        pruning_mask_sequence.append(pruning_mask)  # TODO: keep track of masks to be able to prune from initial weights
        pass

    # TODO: Add final pruning and comparison in low noise env -> Add variable for this setting in parser