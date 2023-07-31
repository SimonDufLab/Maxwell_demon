""" Code that implements pruning with structured baselines, including EarlyCrop-S"""

import copy
import optax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
from aim import Run, Distribution
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
from jax.tree_util import Partial

import utils.utils as utl
import utils.scores as scr
from utils.utils import build_models
from utils.config import activation_choice, optimizer_choice, dataset_choice, dataset_target_cardinality
from utils.config import regularizer_choice, architecture_choice, lr_scheduler_choice, bn_config_choice
from utils.config import pick_architecture, pruning_criterion_choice


# Experience name -> for aim logger
exp_name = "structured_baseline"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 250001
    report_freq: int = 2500
    record_freq: int = 250
    full_ds_eval_freq: int = 1000
    live_freq: int = 25000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
    lr: float = 1e-3
    gradient_clipping: bool = False
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    train_batch_size: int = 512
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adamw"
    activation: str = "relu"  # Activation function used throughout the model
    dataset: str = "mnist"
    normalize_inputs: bool = False  # Substract mean across channels from inputs and divide by variance
    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
    noisy_label: float = 0.0  # ratio (between [0,1]) of labels to randomly (uniformly) flip
    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
    architecture: str = "mlp_3"
    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
    with_bn: bool = False  # Add batchnorm layers or not in the models
    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
    size: Any = 50  # Can also be a tuple for convnets
    regularizer: Optional[str] = "None"
    reg_param: float = 5e-4
    wd_param: Optional[float] = None
    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
    pruning_criterion: Optional[str] = None
    pruning_density: float = 0.0
    pruning_steps: int = 1  # Number of steps for single shot iterative pruning
    modulate_target_density: bool = True  # Not in paper but in code, modify the threshold calculation
    pruning_args: Any = None
    init_seed: int = 41
    dropout_rate: float = 0
    with_rng_seed: int = 428
    save_wanda: bool = False  # Whether to save weights and activations value or not
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name + "_config", node=ExpConfig)

# Using tf on CPU for data loading # Set earlier
# tf.config.experimental.set_visible_devices([], "GPU")

@hydra.main(version_base=None, config_name=exp_name + "_config")
def run_exp(exp_config: ExpConfig) -> None:

    run_start_time = time.time()

    if "imagenet" in exp_config.dataset:
        dataset_dir = exp_config.dataset
        exp_config.dataset = "imagenet"

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config.dataset in dataset_choice.keys(), "Currently supported datasets: " + str(
        dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(
        regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(
        architecture_choice.keys())
    assert exp_config.activation in activation_choice.keys(), "Current activation function available: " + str(
        activation_choice.keys())
    assert exp_config.lr_schedule in lr_scheduler_choice.keys(), "Current lr scheduler function available: " + str(
        lr_scheduler_choice.keys())
    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())
    assert exp_config.pruning_criterion in pruning_criterion_choice.keys(), "Current pruning criterions available: " + str(
        pruning_criterion_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if exp_config.wd_param == 'None':
        exp_config.wd_param = None
    if exp_config.pruning_criterion == 'None':
        exp_config.pruning_criterion = None
    assert (not (("adamw" in exp_config.optimizer) and bool(
        exp_config.regularizer))) or bool(
        exp_config.wd_param), "Set wd_param if adamw is used with a regularization loss"
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.epsilon_close) == str:
        exp_config.epsilon_close = literal_eval(exp_config.epsilon_close)

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./Neurips2023_structured_baseline", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    if exp_config.save_wanda:
        # Create pickle directory
        pickle_dir_path = "./Neurips2023_structured_baseline/metadata/" + exp_name + time.strftime(
            "/%Y-%m-%d---%B %d---%H:%M:%S/")
        os.makedirs(pickle_dir_path)
        # Dump config file in it as well
        with open(pickle_dir_path + 'config.json', 'w') as fp:
            json.dump(OmegaConf.to_container(exp_config), fp, indent=4)

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
    if exp_config.kept_classes:
        assert exp_config.kept_classes <= dataset_target_cardinality[
            exp_config.dataset], "subset must be smaller or equal to total number of classes in ds"
        kept_indices = np.random.choice(dataset_target_cardinality[exp_config.dataset], exp_config.kept_classes,
                                        replace=False)
    else:
        kept_indices = None
    load_data = dataset_choice[exp_config.dataset]
    if exp_config.dataset == 'imagenet':
        load_data = Partial(load_data, dataset_dir)
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
    steps_per_epoch = train_ds_size // exp_config.train_batch_size
    if exp_config.dataset == 'imagenet':
        partial_train_ds_size = train_ds_size/1000  # .1% of dataset used for evaluation on train
        test_death = train_eval  # Don't want to prefetch too many ds
    else:
        partial_train_ds_size = train_ds_size

    if exp_config.save_wanda:
        # Recording metadata about activations that will be pickled
        @dataclass
        class ActivationMeta:
            maximum: List[float] = field(default_factory=list)
            mean: List[float] = field(default_factory=list)
            count: List[int] = field(default_factory=list)

        activations_meta = ActivationMeta()

    size = exp_config.size
    # Make the network and optimiser
    architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
    if not exp_config.kept_classes:
        classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
    else:
        classes = exp_config.kept_classes
    architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
    net, raw_net = build_models(*architecture, with_dropout=with_dropout)

    optimizer = optimizer_choice[exp_config.optimizer]
    opt_chain = []
    if exp_config.gradient_clipping:
        opt_chain.append(optax.clip(10))
    if "adamw" in exp_config.optimizer:  # Pass reg_param to wd argument of adamw
        if exp_config.wd_param:  # wd_param overwrite reg_param when specified
            optimizer = Partial(optimizer, weight_decay=exp_config.wd_param)
        else:
            optimizer = Partial(optimizer, weight_decay=exp_config.reg_param)
    elif exp_config.wd_param:  # TODO: Maybe exclude adamw?
        opt_chain.append(optax.add_decayed_weights(weight_decay=exp_config.wd_param))

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
    scan_len = int(partial_train_ds_size // death_minibatch_size)
    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
    # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
        utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))
    if exp_config.pruning_criterion:
        pruning_score_fn, pruning_step_test_fn = pruning_criterion_choice[exp_config.pruning_criterion]
        step_test_carry = 0.0
    pruned_flag = not bool(exp_config.pruning_criterion)

    params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
    initial_params = copy.deepcopy(params)
    activation_layer_order = list(state.keys())
    neuron_states = utl.NeuronStates(activation_layer_order)
    acti_map = utl.get_activation_mapping(raw_net, next(train))
    opt_state = opt.init(params)

    starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, size)
    total_neurons, total_per_layer = starting_neurons, starting_per_layer
    init_total_neurons = copy.copy(total_neurons)
    init_total_per_layer = copy.copy(total_per_layer)

    initial_params_count = utl.count_params(params)

    decaying_reg_param = copy.deepcopy(exp_config.reg_param)
    decay_cycles = exp_config.reg_param_decay_cycles + int(exp_config.zero_end_reg_param)
    if decay_cycles == 2:
        reg_param_decay_period = int(0.8 * exp_config.training_steps)
    else:
        reg_param_decay_period = exp_config.training_steps // decay_cycles

    target_density_for_th = exp_config.pruning_density

    def get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn, scan_death_check_fn, full_train_acc_fn,
                                     final_accuracy_fn):
        def print_and_record_metrics(step, params, state, total_neurons, total_per_layer):
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
                    print(f"current lr : {lr_schedule(step):.5f}")
                dead_neurons = death_check_fn(params, state, next(test_death))
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step)
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step)
                exp_run.track(test_accuracy, name="Test accuracy", step=step)
                exp_run.track(train_accuracy, name="Train accuracy", step=step)
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step)
                exp_run.track(jax.device_get(test_loss), name="Test loss", step=step)

            if step % exp_config.full_ds_eval_freq == 0 or step == exp_config.training_steps - 1:
                dead_neurons = scan_death_check_fn(params, state, test_death)
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)

                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                              step=step)
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                              name="Live neurons; whole training dataset",
                              step=step)
                exp_run.track(jax.device_get((total_neurons - dead_neurons_count)/starting_neurons),
                              name="Live neurons ratio; whole training dataset",
                              step=step)
                for i, layer_dead in enumerate(dead_per_layers):
                    total_neuron_in_layer = total_per_layer[i]
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step)
                    exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                                  name=f"Live neurons in layer {i}; whole training dataset", step=step)
                if exp_config.dataset != "imagenet":
                    train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, state, train_eval))
                    exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                                  step=step)
                    test_acc_whole_ds = jax.device_get(final_accuracy_fn(params, state, test_eval))
                    exp_run.track(test_acc_whole_ds, name="Test accuracy; whole eval dataset",
                                  step=step)
        return print_and_record_metrics

    print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                            scan_death_check_fn, full_train_acc_fn, final_accuracy_fn)

    for step in range(exp_config.training_steps):
        # Decaying reg_param if applicable
        if (decay_cycles > 1) and (step % reg_param_decay_period == 0) and \
                (not (step % exp_config.training_steps == 0)):
            decaying_reg_param = decaying_reg_param / 10
            if (step >= (decay_cycles - 1) * reg_param_decay_period) and exp_config.zero_end_reg_param:
                decaying_reg_param = 0
            print(f"decaying reg param: {decaying_reg_param:.5f}")
            print()
            loss.clear_cache()
            test_loss_fn.clear_cache()
            update_fn.clear_cache()
            loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=decaying_reg_param,
                                           classes=classes, with_dropout=with_dropout)
            test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                   reg_param=decaying_reg_param,
                                                   classes=classes, is_training=False, with_dropout=with_dropout)
            update_fn = utl.update_given_loss_and_optimizer(loss, opt,  with_dropout=with_dropout)
            print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                                    scan_death_check_fn, full_train_acc_fn,
                                                                    final_accuracy_fn)

        if not pruned_flag and (step % steps_per_epoch == 0):
            if step == steps_per_epoch and exp_config.modulate_target_density:  # First epoch
                initial_params = copy.deepcopy(params)
            if step == 2*steps_per_epoch and exp_config.modulate_target_density:  # Second epoch
                print(f"old th: {target_density_for_th:.4f}")
                target_density_for_th = scr.modulate_target_density(exp_config.pruning_density, params, initial_params)
                print(f"new th: {target_density_for_th:.4f}")
            pruned_flag, step_test_carry = pruning_step_test_fn(target_density_for_th, params, initial_params, step_test_carry)
            if pruned_flag:  # Performs pruning
                del initial_params  # Need to free memory
                print(f"Performing pruning at step {step}")
                _architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                    exp_config.architecture]
                _architecture = Partial(_architecture, num_classes=classes, activation_fn=activation_fn, **net_config)
                _get_loss_test_fn = Partial(utl.ce_loss_given_model, regularizer=exp_config.regularizer,
                                            reg_param=decaying_reg_param,
                                            classes=classes,
                                            is_training=False, with_dropout=with_dropout)
                # neuron_scores = pruning_score_fn(params, state, test_loss_fn, train_eval, train_ds_size//eval_size)
                # neuron_states.update(scr.score_to_neuron_mask(exp_config.pruning_density, neuron_scores))
                # params, opt_state, state, new_sizes = utl.prune_params_state_optstate(params,
                #                                                                       acti_map,
                #                                                                       neuron_states,
                #                                                                       opt_state,
                #                                                                       state)

                neuron_states, params, opt_state, state, new_sizes = scr.iterative_single_shot_pruning(
                    exp_config.pruning_density, params, state, opt_state, acti_map, neuron_states, pruning_score_fn,
                    _architecture, test_loss_fn, _get_loss_test_fn, train_eval, pruning_steps=exp_config.pruning_steps)

                # Build pruned net
                architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                    exp_config.architecture]
                architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
                net, raw_net = build_models(*architecture)
                total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes)

                # Clear previous cache
                # loss.clear_cache()
                # test_loss_fn.clear_cache()
                # accuracy_fn.clear_cache()
                # update_fn.clear_cache()
                # death_check_fn.clear_cache()
                # scan_death_check_fn.clear_cache()
                # scan_death_check_fn_with_activations_data.clear_cache()

                # Recompile training/monitoring functions
                loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                               reg_param=decaying_reg_param, classes=classes,
                                               with_dropout=with_dropout)
                test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                       reg_param=decaying_reg_param,
                                                       classes=classes,
                                                       is_training=False, with_dropout=with_dropout)
                accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
                update_fn = utl.update_given_loss_and_optimizer(loss, opt,
                                                                with_dropout=with_dropout)
                death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
                scan_len = int(partial_train_ds_size // death_minibatch_size)
                scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
                scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
                    utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
                final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
                full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

                print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                                        scan_death_check_fn, full_train_acc_fn,
                                                                        final_accuracy_fn)

        # Metrics and logs:
        print_and_record_metrics(step, params, state, total_neurons, total_per_layer)
        # Train step over single batch
        if with_dropout:
            params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train),
                                                              dropout_key)
        else:
            params, state, opt_state = update_fn(params, state, opt_state, next(train))

    final_accuracy = jax.device_get(final_accuracy_fn(params, state, test_eval))
    activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
    # remaining_params = utl.remove_dead_neurons_weights(params, final_dead_neurons,
    #                                                 frozen_layer_lists, opt_state,
    #                                                 state)[0]
    neuron_states.update_from_ordered_list(final_dead_neurons)
    remaining_params = utl.prune_params_state_optstate(params, acti_map,
                                                       neuron_states, opt_state,
                                                       state)[0]
    final_params_count = utl.count_params(remaining_params)
    final_dead_neurons_count, final_dead_per_layer = utl.count_dead_neurons(final_dead_neurons)
    total_live_neurons = total_neurons - final_dead_neurons_count
    del final_dead_neurons  # Freeing memory
    log_params_sparsity_step = final_params_count / initial_params_count * 1000
    compression_ratio = initial_params_count/final_params_count
    exp_run.track(final_accuracy,
                  name="Accuracy after convergence w/r percent*10 of params remaining",
                  step=log_params_sparsity_step)
    exp_run.track(final_accuracy,
                  name="Accuracy after convergence w/r params compression ratio",
                  step=compression_ratio)
    exp_run.track(1-(log_params_sparsity_step/1000),
                  name="Sparsity w/r params compression ratio",
                  step=compression_ratio)
    log_sparsity_step = jax.device_get(total_live_neurons / init_total_neurons) * 1000
    exp_run.track(final_accuracy,
                  name="Accuracy after convergence w/r percent*10 of neurons remaining", step=log_sparsity_step)

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
