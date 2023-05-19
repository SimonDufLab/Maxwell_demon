"""This experiment tries to verify the following hypothesis: what's the relative importance of noise
for pruning neurons throughout training? As a partial answer, we train a network in a setting that promotes
 neuron's death to identify a subnetwork. After initial training, we reinitialize weights and retrain again in a
 less noisy setting (high bs, low lr) a non-pruned version and a pruned version. We expect much less neurons death
  for the unpruned version and better performance in pruned version in overfitting regime. We want to study the
  performance gap provided by the implicit noise pruning effect"""
import copy

import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
from aim import Run, Figure, Distribution, Image
import os
import time
from datetime import timedelta
import pickle
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, List, Union
from ast import literal_eval
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

import utils.utils as utl
from utils.utils import build_models
from utils.config import activation_choice, optimizer_choice, dataset_choice, dataset_target_cardinality
from utils.config import regularizer_choice, architecture_choice, lr_scheduler_choice, bn_config_choice
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
    pruning_cycles: int = 2
    cycle_tolerance: float = 0.01  # Stop pruning cycle early is variation in dead neurons is smaller than tolerance
    lr: float = 1e-3
    end_lr: float = 1e-5  # lr used during low noise evaluation
    gradient_clipping: bool = False
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    end_final_lr: float = 1e-6  # final lr used for scheduler during low noise evaluation
    lr_decay_steps: Any = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    train_batch_size: int = 16
    end_train_batch_size: int = 512  # final training batch size used during low noise evaluation
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    activation: str = "relu"  # Activation function used throughout the model
    dataset: str = "mnist"
    normalize_inputs: bool = False  # Substract mean across channels from inputs and divide by variance
    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
    architecture: str = "mlp_3"
    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
    with_bn: bool = False  # Add batchnorm layers or not in the models
    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
    size: Any = 50
    regularizer: Optional[str] = 'None'
    reg_param: float = 1e-4
    wd_param: Optional[float] = None
    cycling_regularizer: Optional[str] = 'None'
    cycling_reg_param: float = 1e-4
    rdm_reinit: bool = False  # Randomly reinit the pruned network and train, for baseline comparison
    rewinding: bool = False  # Checkpoint early on like LTR and compare performance to complete rewinding to init
    init_seed: int = 41
    dynamic_pruning: bool = False
    dropout_rate: float = 0
    with_rng_seed: int = 428
    run_low_noise: bool = False
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
    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if exp_config.cycling_regularizer == 'None':
        exp_config.cycling_regularizer = None
    if exp_config.wd_param == 'None':
        exp_config.wd_param = None
    assert (not (("adamw" in exp_config.optimizer) and bool(
        exp_config.regularizer))) or bool(
        exp_config.wd_param), "Set wd_param if adamw is used with a regularization loss"
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.lr_decay_steps) == str:
        exp_config.lr_decay_steps = literal_eval(exp_config.lr_decay_steps)

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./Neurips2023_LTH_compare", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Experiments with dropout
    with_dropout = exp_config.dropout_rate > 0
    if with_dropout:
        dropout_key = jax.random.PRNGKey(exp_config.with_rng_seed)
        assert exp_config.architecture in pick_architecture(
            with_dropout=True).keys(), "Current architectures available with dropout: " + str(
            pick_architecture(with_dropout=True).keys())
        net_config = {"dropout_rate": exp_config.dropout_rate}
    else:
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
    eval_size = exp_config.eval_batch_size
    death_minibatch_size = exp_config.death_batch_size
    train_ds_size, train, train_eval, test_death = load_data(split="train", is_training=True,
                                                             batch_size=exp_config.train_batch_size,
                                                             other_bs=[eval_size, death_minibatch_size],
                                                             cardinality=True,
                                                             augment_dataset=exp_config.augment_dataset,
                                                             normalize=exp_config.normalize_inputs)
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size,
                                     cardinality=True, augment_dataset=exp_config.augment_dataset,
                                     normalize=exp_config.normalize_inputs)

    # Make the network and optimiser
    architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
    classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
    size = exp_config.size
    architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
    net = build_models(*architecture, with_dropout=with_dropout)

    optimizer = optimizer_choice[exp_config.optimizer]
    if "adamw" in exp_config.optimizer:  # Pass reg_param to wd argument of adamw
        if exp_config.wd_param:  # wd_param overwrite reg_param when specified
            optimizer = Partial(optimizer, weight_decay=exp_config.wd_param)
        else:
            optimizer = Partial(optimizer, weight_decay=exp_config.reg_param)
    opt_chain = []
    if exp_config.gradient_clipping:
        opt_chain.append(optax.clip(10))

    if 'noisy' in exp_config.optimizer:
        opt_chain.append(optimizer(exp_config.lr, eta=exp_config.noise_eta,
                                   gamma=exp_config.noise_gamma))
    else:
        lr_schedule = lr_scheduler_choice[exp_config.lr_schedule](exp_config.training_steps, exp_config.lr,
                                                                  exp_config.final_lr, exp_config.lr_decay_steps)
        opt_chain.append(optimizer(lr_schedule))
    opt = optax.chain(*opt_chain)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                   classes=classes, with_dropout=with_dropout)
    test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                           classes=classes, is_training=False, with_dropout=with_dropout)
    accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt, with_dropout=with_dropout)
    death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
    scan_len = train_ds_size // death_minibatch_size
    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
        utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_ds_size // eval_size)

    # Initialize
    params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
    reinit_fn = Partial(net.init, jax.random.PRNGKey(exp_config.init_seed + 42))  # Checkpoint initial init function
    opt_state = opt.init(params)
    initial_params = copy.deepcopy(params)  # We need to keep a copy of the initial params for later reset
    initial_state = copy.deepcopy(state)
    frozen_layer_lists = utl.extract_layer_lists(params)

    starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, size)
    total_neurons, total_per_layer = starting_neurons, starting_per_layer
    initial_params_count = utl.count_params(params)

    # Rewinding
    def checkpoint_fn(_step):
        return utl.get_checkpoint_step(exp_config.architecture, _step)
    checkpoints = []

    def get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn, scan_death_check_fn, full_train_acc_fn,
                                     final_accuracy_fn):
        def print_and_record_metrics(step, context, params, state, total_neurons, total_per_layer):
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
                    print(f"current lr : {lr_schedule(step):.3f}")
                dead_neurons = death_check_fn(params, state, next(test_death))
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                              context={"experiment phase": context})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step,
                              context={"experiment phase": context})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"experiment phase": context})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={"experiment phase": context})
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                              context={"experiment phase": context})
                exp_run.track(jax.device_get(test_loss), name="Test loss", step=step,
                              context={"experiment phase": context})

            if step % exp_config.full_ds_eval_freq == 0 or step == exp_config.training_steps - 1:
                dead_neurons = scan_death_check_fn(params, state, test_death)
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)

                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                              step=step,
                              context={"experiment phase": context})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                              name="Live neurons; whole training dataset",
                              step=step,
                              context={"experiment phase": context})
                exp_run.track(jax.device_get((total_neurons - dead_neurons_count)/starting_neurons),
                              name="Live neurons ratio; whole training dataset",
                              step=step,
                              context={"experiment phase": context})
                for i, layer_dead in enumerate(dead_per_layers):
                    total_neuron_in_layer = total_per_layer[i]
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                                  context={"experiment phase": context})
                    exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                                  name=f"Live neurons in layer {i}; whole training dataset", step=step,
                                  context={"experiment phase": context})
                train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, state, train_eval))
                exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                              step=step,
                              context={"experiment phase": context})
                test_acc_whole_ds = jax.device_get(final_accuracy_fn(params, state, test_eval))
                exp_run.track(test_acc_whole_ds, name="Test accuracy; whole eval dataset",
                              step=step,
                              context={"experiment phase": context})
        return print_and_record_metrics

    print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                            scan_death_check_fn, full_train_acc_fn, final_accuracy_fn)

    for step in range(exp_config.training_steps):
        # Metrics and logs:
        print_and_record_metrics(step, "noisy", params, state, total_neurons, total_per_layer)
        # record checkpoints if rewinding:
        if exp_config.rewinding:
            if step == checkpoint_fn(step):
                checkpoints.append((copy.deepcopy(params), step)) # Checkpoint params and step where recorded
        # Train step over single batch
        if with_dropout:
            params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train),
                                                              dropout_key)
        else:
            params, state, opt_state = update_fn(params, state, opt_state, next(train))

    final_accuracy = jax.device_get(final_accuracy_fn(params, state, test_eval))
    activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
    remaining_params = utl.remove_dead_neurons_weights(params, final_dead_neurons,
                                                    frozen_layer_lists, opt_state,
                                                    state)[0]
    final_params_count = utl.count_params(remaining_params)
    del final_dead_neurons  # Freeing memory
    log_params_sparsity_step = final_params_count / initial_params_count * 1000
    compression_ratio = initial_params_count/final_params_count
    exp_run.track(final_accuracy,
                  name="Accuracy after convergence w/r percent*10 of params remaining",
                  step=log_params_sparsity_step,
                  context={"experiment phase": "noisy"})
    exp_run.track(final_accuracy,
                  name="Accuracy after convergence w/r params compression ratio",
                  step=compression_ratio,
                  context={"experiment phase": "noisy"})
    exp_run.track(1-(log_params_sparsity_step/1000),
                  name="Sparsity w/r params compression ratio",
                  step=compression_ratio,
                  context={"experiment phase": "noisy"})

    # Print running time
    print()
    print(f"Running time for run before pruning cycles: " + str(timedelta(seconds=time.time() - run_start_time)))
    print("----------------------------------------------")
    print()

    cycle_step = 1
    tol_flag = True
    dead_neurons = scan_death_check_fn(params, state, test_death)
    if exp_config.rdm_reinit or exp_config.rewinding:
        preserve_dead_state = copy.deepcopy(dead_neurons)
    pruned_init_params = initial_params
    state = initial_state
    while (cycle_step < exp_config.pruning_cycles) and tol_flag:
        subtask_start_time = time.time()
        # prune the init params
        pruned_init_params, opt_state, state, new_sizes = utl.remove_dead_neurons_weights(pruned_init_params, dead_neurons,
                                                                                          frozen_layer_lists, opt_state,
                                                                                          state)
        params = copy.deepcopy(pruned_init_params)
        # state = initial_state
        opt_state = opt.init(params)
        architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
        architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
        net = build_models(*architecture)
        del dead_neurons  # Freeing memory

        context = f'noisy pruning cycle {cycle_step}'
        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes)

        # Clear previous cache
        loss.clear_cache()
        test_loss_fn.clear_cache()
        accuracy_fn.clear_cache()
        update_fn.clear_cache()
        death_check_fn.clear_cache()

        # Recompile training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.cycling_regularizer,
                                       reg_param=exp_config.cycling_reg_param, classes=classes,
                                       with_dropout=with_dropout)
        test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.cycling_regularizer,
                                               reg_param=exp_config.cycling_reg_param,
                                               classes=classes,
                                               is_training=False, with_dropout=with_dropout)
        accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt, with_dropout=with_dropout)
        death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
        scan_len = train_ds_size // death_minibatch_size
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
            utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_ds_size // eval_size)

        print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                                scan_death_check_fn, full_train_acc_fn,
                                                                final_accuracy_fn)

        for step in range(exp_config.training_steps):
            # Metrics and logs:
            print_and_record_metrics(step, context, params, state, total_neurons, total_per_layer)
            # Train step over single batch
            if with_dropout:
                params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train),
                                                                          dropout_key)
            else:
                params, state, opt_state = update_fn(params, state, opt_state, next(train))

        final_accuracy = jax.device_get(final_accuracy_fn(params, state, test_eval))
        activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
        remaining_params = utl.remove_dead_neurons_weights(params, final_dead_neurons,
                                                           frozen_layer_lists, opt_state,
                                                           state)[0]
        final_params_count = utl.count_params(remaining_params)
        del final_dead_neurons  # Freeing memory
        log_params_sparsity_step = final_params_count / initial_params_count * 1000
        compression_ratio = initial_params_count/final_params_count
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r percent*10 of params remaining",
                      step=log_params_sparsity_step,
                      context={"experiment phase": context})
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r params compression ratio",
                      step=compression_ratio,
                      context={"experiment phase": context})
        exp_run.track(1 - (log_params_sparsity_step / 1000),
                      name="Sparsity w/r params compression ratio",
                      step=compression_ratio,
                      context={"experiment phase": context})

        # Print running time
        print()
        print(f"Running time for pruning cycle {cycle_step}: " + str(
            timedelta(seconds=time.time() - subtask_start_time)))
        print("----------------------------------------------")
        print()

        cycle_step += 1
        dead_neurons = scan_death_check_fn(params, state, test_death)
        curr_dead_amount, _ = utl.count_dead_neurons(dead_neurons)
        tol_flag = (curr_dead_amount/total_neurons) >= exp_config.cycle_tolerance

    del dead_neurons  # Freeing memory

    contexts = []
    if exp_config.rdm_reinit:
        contexts.append("pruned, random reinit")
    for chkpt in checkpoints:
        contexts.append(f'pruned, rewinding to step {chkpt[1]}')

    for i, context in enumerate(contexts):
        subtask_start_time = time.time()
        if "random" in context:
            to_prune = reinit_fn(next(train))[0]  # Reinitialize params
            checkpoints = [(None, None)] + checkpoints
        else:
            to_prune = checkpoints[i][0]  # Retrieving params at checkpoint i
        opt_state = opt.init(to_prune)
        params, opt_state, state, new_sizes = utl.remove_dead_neurons_weights(to_prune,
                                                                              preserve_dead_state,
                                                                              frozen_layer_lists, opt_state,
                                                                              initial_state)
        opt_state = opt.init(params)
        architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
        architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
        net = build_models(*architecture)

        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes)

        # Clear previous cache
        loss.clear_cache()
        test_loss_fn.clear_cache()
        accuracy_fn.clear_cache()
        update_fn.clear_cache()
        death_check_fn.clear_cache()

        # Recompile training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.cycling_regularizer,
                                       reg_param=exp_config.cycling_reg_param, classes=classes,
                                       with_dropout=with_dropout)
        test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.cycling_regularizer,
                                               reg_param=exp_config.cycling_reg_param,
                                               classes=classes,
                                               is_training=False, with_dropout=with_dropout)
        accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt, with_dropout=with_dropout)
        death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
        scan_len = train_ds_size // death_minibatch_size
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
            utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_ds_size // eval_size)

        print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                                scan_death_check_fn, full_train_acc_fn,
                                                                final_accuracy_fn)

        for step in range(exp_config.training_steps):
            # Metrics and logs:
            print_and_record_metrics(step, context, params, state, total_neurons, total_per_layer)
            # Train step over single batch
            if with_dropout:
                params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train),
                                                                  dropout_key)
            else:
                params, state, opt_state = update_fn(params, state, opt_state, next(train))

        final_accuracy = jax.device_get(final_accuracy_fn(params, state, test_eval))
        activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
        remaining_params = utl.remove_dead_neurons_weights(params, final_dead_neurons,
                                                           frozen_layer_lists, opt_state,
                                                           state)[0]
        final_params_count = utl.count_params(remaining_params)
        del final_dead_neurons  # Freeing memory
        log_params_sparsity_step = final_params_count / initial_params_count * 1000
        compression_ratio = initial_params_count / final_params_count
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r percent*10 of params remaining",
                      step=log_params_sparsity_step,
                      context={"experiment phase": context})
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r params compression ratio",
                      step=compression_ratio,
                      context={"experiment phase": context})
        exp_run.track(1 - (log_params_sparsity_step / 1000),
                      name="Sparsity w/r params compression ratio",
                      step=compression_ratio,
                      context={"experiment phase": context})

        # Print running time
        print()
        print(f"Running time for context: {context}: " + str(
            timedelta(seconds=time.time() - subtask_start_time)))
        print("----------------------------------------------")
        print()

    # TODO: Add final pruning and comparison in low noise env -> Add variable for this setting in parser
    if exp_config.run_low_noise:
        for context in ['pruned/low noise', 'not pruned/low noise']:
            subtask_start_time = time.time()
            # different bs and lr
            load_data = dataset_choice[exp_config.dataset]
            train = load_data(split="train", is_training=True,
                              batch_size=exp_config.end_train_batch_size,
                              cardinality=False)
            lr_schedule = lr_scheduler_choice[exp_config.lr_schedule](exp_config.training_steps, exp_config.end_lr,
                                                                      exp_config.end_final_lr, exp_config.lr_decay_steps)
            opt = optimizer_choice[exp_config.optimizer](lr_schedule)
            if context == 'not pruned/low noise':
                end_params = copy.deepcopy(initial_params)
                end_state = copy.deepcopy(initial_state)
                opt_state = opt.init(end_params)  # reinit optimizer state

                architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
                architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
                net = build_models(*architecture)
                total_neurons, total_per_layer = starting_neurons, starting_per_layer
            else:
                dead_neurons = scan_death_check_fn(pruned_init_params, state, test_death)
                # Pruning the network
                end_state = initial_state
                end_params, opt_state, state, new_sizes = utl.remove_dead_neurons_weights(initial_params, dead_neurons,
                                                                                          frozen_layer_lists, opt_state,
                                                                                          end_state)
                # end_state = initial_state
                del dead_neurons  # Freeing memory
                opt_state = opt.init(end_params)  # reinit optimizer state

                architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
                architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
                net = build_models(*architecture)
                total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes)

            # Clear previous cache
            loss.clear_cache()
            test_loss_fn.clear_cache()
            accuracy_fn.clear_cache()
            update_fn.clear_cache()
            death_check_fn.clear_cache()

            # Recompile training/monitoring functions
            loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                           reg_param=exp_config.reg_param, classes=classes,
                                           with_dropout=with_dropout)
            test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                   reg_param=exp_config.reg_param,
                                                   classes=classes,
                                                   is_training=False, with_dropout=with_dropout)
            accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
            update_fn = utl.update_given_loss_and_optimizer(loss, opt, with_dropout=with_dropout)
            death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
            scan_len = train_ds_size // death_minibatch_size
            scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
            scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
                utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
            final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
            full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_ds_size // eval_size)

            print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                                    scan_death_check_fn, full_train_acc_fn,
                                                                    final_accuracy_fn)

            for step in range(exp_config.training_steps):
                # Metrics and logs:
                print_and_record_metrics(step, context, end_params, end_state, total_neurons, total_per_layer)
                # Train step over single batch
                if with_dropout:
                    end_params, end_state, opt_state, dropout_key = update_fn(end_params, end_state, opt_state, next(train),
                                                                      dropout_key)
                else:
                    end_params, end_state, opt_state = update_fn(end_params, end_state, opt_state, next(train))

            final_accuracy = jax.device_get(final_accuracy_fn(end_params, end_state, test_eval))
            activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(end_params, end_state, test_death)
            remaining_params = utl.remove_dead_neurons_weights(end_params, final_dead_neurons,
                                                               frozen_layer_lists, opt_state,
                                                               end_state)[0]
            final_params_count = utl.count_params(remaining_params)
            del final_dead_neurons  # Freeing memory
            log_params_sparsity_step = final_params_count / initial_params_count * 1000
            compression_ratio = initial_params_count/final_params_count
            exp_run.track(final_accuracy,
                          name="Accuracy after convergence w/r percent*10 of params remaining",
                          step=log_params_sparsity_step,
                          context={"experiment phase": context})
            exp_run.track(final_accuracy,
                          name="Accuracy after convergence w/r params compression ratio",
                          step=compression_ratio,
                          context={"experiment phase": context})
            exp_run.track(1 - (log_params_sparsity_step / 1000),
                          name="Sparsity w/r params compression ratio",
                          step=compression_ratio,
                          context={"experiment phase": context})

            # Print running time
            print()
            print(f"Running time for ends runs with context {context}: " + str(
                timedelta(seconds=time.time() - subtask_start_time)))
            print("----------------------------------------------")
            print()


    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
