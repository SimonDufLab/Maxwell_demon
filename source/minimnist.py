""" Small experiment setting to support theoritical assumption made in the paper"""

import copy
import random

import jax
import jax.numpy as jnp
import optax
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
    training_steps: int = 25000
    report_freq: int = 2500
    record_freq: int = 100
    full_ds_eval_freq: int = 1000
    lr: float = 1e-3
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    train_batch_size: int = 32
    full_batch_size: int = 1000
    optimizer: str = "adam"
    noisy_part_of_signal_only: bool = False  # Take grad to be full-batch gradient minus minibatch gradient
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

    with_dropout = False
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

    # Make the network and optimiser
    architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
    classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
    size = exp_config.size
    architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
    net, raw_net = build_models(*architecture, with_dropout=with_dropout)
    # for i in hk.experimental.eval_summary(net)(next(train)):
    #     print(i.module_details.module.module_name)
    # raise SystemExit

    optimizer = optimizer_choice[exp_config.optimizer]
    opt_chain = []
    # if exp_config.gradient_clipping:
    #     opt_chain.append(optax.clip(10))
    if "adamw" in exp_config.optimizer:  # Pass reg_param to wd argument of adamw
        if exp_config.wd_param:  # wd_param overwrite reg_param when specified
            optimizer = Partial(optimizer, weight_decay=exp_config.wd_param)
        else:
            optimizer = Partial(optimizer, weight_decay=exp_config.reg_param)
    elif exp_config.wd_param:  # TODO: Maybe exclude adamw?
        opt_chain.append(optax.add_decayed_weights(weight_decay=exp_config.wd_param))

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
    if not exp_config.noisy_part_of_signal_only:
        update_fn = utl.update_given_loss_and_optimizer(loss, opt, with_dropout=with_dropout)
    else:
        update_fn = utl.update_from_noise(loss, opt, with_dropout=with_dropout)
    death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
    scan_len = train_ds_size // death_minibatch_size
    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
        utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_ds_size // eval_size)

    # Initialize
    params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
    reinit_params, _ = net.init(jax.random.PRNGKey(exp_config.init_seed + 42), next(train))
    activation_layer_order = list(state.keys())
    neuron_states = utl.NeuronStates(activation_layer_order)
    ordered_layer_list = list(params.keys())
    acti_map = utl.get_activation_mapping(raw_net, next(train))
    # reinit_fn = Partial(net.init, jax.random.PRNGKey(exp_config.init_seed + 42))  # Checkpoint initial init function
    opt_state = opt.init(params)
    initial_params = copy.deepcopy(params)  # We need to keep a copy of the initial params for later reset
    initial_state = copy.deepcopy(state)
    # frozen_layer_lists = utl.extract_layer_lists(params)

    # # Visualize NN with tabulate
    # print(hk.experimental.tabulate(net.init)(next(train)))
    # raise SystemExit

    starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, size)
    total_neurons, total_per_layer = starting_neurons, starting_per_layer
    initial_params_count = utl.count_params(params)

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
                    # print(f"current lr : {lr_schedule(step):.5f}")
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
                train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, state, train_eval))
                exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                              step=step)
                test_acc_whole_ds = jax.device_get(final_accuracy_fn(params, state, test_eval))
                exp_run.track(test_acc_whole_ds, name="Test accuracy; whole eval dataset",
                              step=step)
                # Record magnitude metrics for live and dead neurons
                neuron_states.update_from_ordered_list(dead_neurons)
                _acti_map = {key: {"preceding": None, 'following': acti_map[key]["following"]} for key in acti_map.keys()}
                live_params = utl.prune_params_state_optstate(params, _acti_map,
                                                                   neuron_states, opt_state,
                                                                   state)[0]
                dead_params = utl.prune_params_state_optstate(params, _acti_map,
                                                                   neuron_states.invert_state(), opt_state,
                                                                   state)[0]
                full_params_norm = jnp.linalg.norm(ravel_pytree(params)[0])
                live_params_norm = jnp.linalg.norm(ravel_pytree(live_params)[0])
                dead_params_norm = jnp.linalg.norm(ravel_pytree(dead_params)[0])
                exp_run.track(
                    jax.device_get(full_params_norm),
                    name="Weights magnitude",
                    step=step, context={"Params subset": "All"})
                exp_run.track(
                    jax.device_get(live_params_norm),
                    name="Weights magnitude",
                    step=step, context={"Params subset": "Live"})
                exp_run.track(
                    jax.device_get(dead_params_norm),
                    name="Weights magnitude",
                    step=step, context={"Params subset": "Dead"})
                for i, key in enumerate(ordered_layer_list[:-1]):
                    exp_run.track(
                        jax.device_get(utl.avg_neuron_magnitude_in_layer(params[key])),
                        name=f"Average neuron magnitude in layer {i}",
                        step=step, context={"Params subset": "All"})
                    exp_run.track(
                        jax.device_get(utl.avg_neuron_magnitude_in_layer(live_params[key])),
                        name=f"Average neuron magnitude in layer {i}",
                        step=step, context={"Params subset": "Live"})
                    exp_run.track(
                        jax.device_get(utl.avg_neuron_magnitude_in_layer(dead_params[key])),
                        name=f"Average neuron magnitude in layer {i}",
                        step=step, context={"Params subset": "Dead"})

        return print_and_record_metrics

    print_and_record_metrics = get_print_and_record_metrics(test_loss_fn, accuracy_fn, death_check_fn,
                                                            scan_death_check_fn, full_train_acc_fn, final_accuracy_fn)

    for step in range(exp_config.training_steps):
        # Metrics and logs:
        print_and_record_metrics(step, params, state, total_neurons, total_per_layer)
        # Train step over single batch
        if not exp_config.noisy_part_of_signal_only:
            params, state, opt_state = update_fn(params, state, opt_state, next(train))
        else:
            params, state, opt_state = update_fn(params, state, opt_state, next(train), next(train_eval))

    final_accuracy = jax.device_get(final_accuracy_fn(params, state, test_eval))
    activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
    neuron_states.update_from_ordered_list(final_dead_neurons)
    remaining_params = utl.prune_params_state_optstate(params, acti_map,
                                                       neuron_states, opt_state,
                                                       state)[0]
    final_params_count = utl.count_params(remaining_params)
    del final_dead_neurons  # Freeing memory
    log_params_sparsity_step = final_params_count / initial_params_count * 1000
    compression_ratio = initial_params_count / final_params_count
    exp_run.track(final_accuracy,
                  name="Accuracy after convergence w/r percent*10 of params remaining",
                  step=log_params_sparsity_step)
    # exp_run.track(final_accuracy,
    #               name="Accuracy after convergence w/r params compression ratio",
    #               step=compression_ratio,
    #               context={"experiment phase": "noisy"})
    # exp_run.track(1 - (log_params_sparsity_step / 1000),
    #               name="Sparsity w/r params compression ratio",
    #               step=compression_ratio,
    #               context={"experiment phase": "noisy"})

    # Print running time
    print()
    print(f"Running time for whole experiment: " + str(timedelta(seconds=time.time() - run_start_time)))
    print("----------------------------------------------")
    print()


if __name__ == "__main__":
    run_exp()
