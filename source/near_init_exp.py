""" Experiment in place to verify empirically if the assumptions for our main hypothesis is respected close to the
initialization. In particular, we'll record the average gradient over different batch seed for the first 100 iterations,
for which we'll assume neurons predisposed for early death we'll have 0 gradient on average. We also verify the impact
of the weights initialization variance over the death probability."""

import copy
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
from utils.config import regularizer_choice, architecture_choice, lr_scheduler_choice
from utils.config import pick_architecture


# Experience name -> for aim logger
exp_name = "initialization_assumptions"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 1000
    total_runs: int = 300  # Amount of time we repeat the experiment before averaging (varying batch order seed)
    record_freq: int = 1
    report_freq: int = 250
    record_full_ds: bool = False  # Record metrics or not w/r to full ds (both train and test)
    record_full_ds_freq: int = 20  # Frequency at which we record metric with respect to full datasets
    lr: float = 1e-3
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    lr_decay_steps: int = 0  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    train_batch_size: int = 128
    eval_batch_size: int = 128
    death_batch_size: int = 128
    optimizer: str = "sgd"
    activation: str = "relu"  # Activation function used throughout the model
    dataset: str = "mnist"
    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
    noisy_label: float = 0  # ratio (between [0,1]) of labels to randomly (uniformly) flip
    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
    architecture: str = "mlp_3"
    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
    with_bn: bool = False  # Add bathnorm layers or not in the models
    size: int = 100
    regularizer: Optional[str] = 'None'
    reg_param: float = 1e-4
    epsilon_close: float = 0.0  # Relaxing criterion for dead neurons, epsilon-close to relu gate
    init_seed: int = 41
    add_noise: bool = False  # Add Gaussian noise to the gradient signal
    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones
    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
    noise_eta: float = 0.01
    noise_gamma: float = 0.0
    noise_seed: int = 1
    dropout_rate: float = 0
    with_rng_seed: int = 428
    save_wanda: bool = False  # Whether to save weights and activations value or not
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name + "_config", node=ExpConfig)

# Using tf on CPU for data loading
# tf.config.experimental.set_visible_devices([], "GPU")


@hydra.main(version_base=None, config_name=exp_name + "_config")
def run_exp(exp_config: ExpConfig) -> None:

    run_start_time = time.time()

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

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.noise_imp) == str:
        exp_config.noise_imp = literal_eval(exp_config.noise_imp)

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    if exp_config.save_wanda:
        # Create pickle directory
        pickle_dir_path = "./logs/metadata/" + exp_name + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
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

    # Load the different dataset
    if exp_config.kept_classes:
        assert exp_config.kept_classes <= dataset_target_cardinality[
            exp_config.dataset], "subset must be smaller or equal to total number of classes in ds"
        kept_indices = np.random.choice(dataset_target_cardinality[exp_config.dataset], exp_config.kept_classes,
                                        replace=False)
    else:
        kept_indices = None
    load_data = dataset_choice[exp_config.dataset]
    eval_size = exp_config.eval_batch_size
    death_minibatch_size = exp_config.death_batch_size
    train_ds_size, train, train_eval, test_death = load_data(split="train", is_training=True,
                                                             batch_size=exp_config.train_batch_size,
                                                             other_bs=[eval_size, death_minibatch_size],
                                                             subset=kept_indices,
                                                             cardinality=True,
                                                             noisy_label=exp_config.noisy_label,
                                                             permuted_img_ratio=exp_config.permuted_img_ratio,
                                                             gaussian_img_ratio=exp_config.gaussian_img_ratio)
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size, subset=kept_indices,
                                     cardinality=True)

    if exp_config.save_wanda:
        # Recording metadata about activations that will be pickled
        @dataclass
        class ActivationMeta:
            maximum: List[float] = field(default_factory=list)
            mean: List[float] = field(default_factory=list)
            count: List[int] = field(default_factory=list)

        activations_meta = ActivationMeta()

        # Recording params value at the end of the training as well
        @dataclass
        class FinalParamsMeta:
            parameters: List[float] = field(default_factory=list)

        params_meta = FinalParamsMeta()

    # Make the network and optimiser
    architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
    if not exp_config.kept_classes:
        classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
    else:
        classes = exp_config.kept_classes
    architecture = architecture(exp_config.size, classes, activation_fn=activation_fn, **net_config)
    net = build_models(*architecture, with_dropout=with_dropout)

    if 'noisy' in exp_config.optimizer:
        opt = optimizer_choice[exp_config.optimizer](exp_config.lr, eta=exp_config.noise_eta,
                                                     gamma=exp_config.noise_gamma)
    else:
        lr_schedule = lr_scheduler_choice[exp_config.lr_schedule](exp_config.training_steps, exp_config.lr,
                                                                  exp_config.final_lr, exp_config.lr_decay_steps)
        opt = optimizer_choice[exp_config.optimizer](lr_schedule)
    accuracies_log = []

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                   classes=classes, with_dropout=with_dropout)
    test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                           classes=classes, is_training=False, with_dropout=with_dropout)
    accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                    exp_config.noise_live_only, with_dropout=with_dropout,
                                                    return_grad=True)
    death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
    scan_len = train_ds_size // death_minibatch_size
    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
        utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_ds_size // eval_size)

    params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
    initial_params = copy.deepcopy(params)  # Keep a copy of the initial params for relative change metric
    opt_state = opt.init(params)
    init_opt_state = copy.deepcopy(opt_state)

    noise_key = jax.random.PRNGKey(exp_config.noise_seed)

    starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, exp_config.size)
    total_neurons, total_per_layer = starting_neurons, starting_per_layer

    all_grad_sum = []
    all_activation_mean = [[] for i in range(len(starting_per_layer))]

    for run_number in range(exp_config.total_runs):
        # Time the subrun for the different runs
        subrun_start_time = time.time()
        # Reinitialize the params and opt for each run
        params, opt_state = initial_params, init_opt_state
        grad_sum = jax.tree_map(lambda x: 0*x, params)
        # TODO: change batch seed (or keep the same loader that is shuffled?)

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
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)
                accuracies_log.append(test_accuracy)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                              context={"run_number": int(run_number)})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step,
                              context={"run_number": int(run_number)})
                for i, layer_dead in enumerate(dead_per_layers):
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}", step=step,
                                  context={"run_number": int(run_number)})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"run_number": int(run_number)})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={"run_number": int(run_number)})
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                              context={"run_number": int(run_number)})
                exp_run.track(jax.device_get(test_loss), name="Test loss", step=step,
                              context={"run_number": int(run_number)})

            if (step % exp_config.record_full_ds_freq == 0) and exp_config.record_full_ds:
                dead_neurons = scan_death_check_fn(params, state, test_death)
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)

                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                              step=step,
                              context={"run_number": int(run_number)})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                              name="Live neurons; whole training dataset",
                              step=step,
                              context={"run_number": int(run_number)})
                for i, layer_dead in enumerate(dead_per_layers):
                    total_neuron_in_layer = total_per_layer[i]
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                                  context={"run_number": int(run_number)})
                    exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                                  name=f"Live neurons in layer {i}; whole training dataset", step=step,
                                  context={"run_number": int(run_number)})
                del dead_per_layers
                del dead_neurons  # Freeing memory
                train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, state, train_eval))
                exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                              step=step,
                              context={"run_number": int(run_number)})

            # Train step over single batch
            if with_dropout:
                grad, params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train),
                                                                  dropout_key)
            else:
                if not exp_config.add_noise:
                    grad, params, state, opt_state = update_fn(params, state, opt_state, next(train))
                    # TODO: chec if we should return updates instead of grad
                else:
                    noise_var = exp_config.noise_eta / ((1 + step) ** exp_config.noise_gamma)
                    noise_var = exp_config.lr * noise_var  # Apply lr for consistency with update size
                    grad, params, state, opt_state, noise_key = update_fn(params, state, opt_state, next(train),
                                                                          noise_var,
                                                                          noise_key)

            grad_sum = jax.tree_map(jnp.add, grad_sum, grad)

        all_grad_sum.append({key: grad_sum[key] for key in list(grad_sum.keys())[:-1]}) # Removing head from stats
        activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
        activations_max, activations_mean, activations_count, _ = activations_data
        for i in range(len(activations_mean)):  # max or mean?
            all_activation_mean[i].append(activations_mean[i])

        # Print running time
        print()
        print(f"Running time for run {run_number}: " + str(timedelta(seconds=time.time() - subrun_start_time)))
        print("----------------------------------------------")
        print()

    # Recover average 'movement' over the different runs
    avg_trajectories, var_trajectories = utl.mean_var_over_pytree_list(all_grad_sum)
    # Concatenate bias to neurons weight and return a dict containing list of neurons vectors
    avg_trajectories = utl.concatenate_bias_to_weights(avg_trajectories)
    var_trajectories = utl.concatenate_bias_to_weights(var_trajectories)
    # Recover the euclidean distance of each average trajectories
    avg_distance = jax.tree_map(jnp.linalg.norm, avg_trajectories)
    var_distance = jax.tree_map(jnp.linalg.norm, var_trajectories)

    flat_avg_distance, _ = ravel_pytree(avg_distance)
    flat_avg_distance = jax.device_get(flat_avg_distance)
    flat_var_distance, _ = ravel_pytree(var_distance)
    flat_var_distance = jax.device_get(flat_var_distance)
    flat_avg_distance_histo = Distribution(flat_avg_distance, bin_count=100)
    exp_run.track(flat_avg_distance_histo, name='Average traveled distance by neurons', step=0,
                  context={"neurons subset": "all"})
    flat_var_distance_histo = Distribution(flat_var_distance, bin_count=100)
    exp_run.track(flat_var_distance_histo, name='Variance of the average traveled distance', step=0,
                  context={"neurons subset": "all"})

    # Some metrics about activation values
    avg_activation_mean = [jnp.mean(jnp.stack(layer), axis=0) for layer in all_activation_mean]
    for layer_num in range(len(avg_activation_mean)):
        histo_act_layer = jax.device_get(avg_activation_mean[layer_num])
        histo_act_layer = Distribution(histo_act_layer, bin_count=100)
        exp_run.track(histo_act_layer, name='Avg max activations value', step=0, context={'layer number': layer_num})

    quasi_dead, _ = jax.device_get(ravel_pytree(avg_activation_mean))
    quasi_dead = quasi_dead <= 1e-3  # TODO: use epsilon arg from config
    print(f'total neurons: {len(flat_avg_distance)}')
    avg_dist_dead = flat_avg_distance[quasi_dead]
    print(f'dead neurons: {len(avg_dist_dead)}')
    avg_dist_dead = Distribution(avg_dist_dead, bin_count=100)
    exp_run.track(avg_dist_dead, name='Average traveled distance by neurons', step=0,
                  context={"neurons subset": "dead"})
    var_dist_dead = flat_var_distance[quasi_dead]
    var_dist_dead = Distribution(var_dist_dead, bin_count=100)
    exp_run.track(var_dist_dead, name='Variance of the average traveled distance', step=0,
                  context={"neurons subset": "dead"})
    avg_dist_live = flat_avg_distance[np.logical_not(quasi_dead)]
    print(f'live neurons: {len(avg_dist_live)}')
    avg_dist_live = Distribution(avg_dist_live, bin_count=100)
    exp_run.track(avg_dist_live, name='Average traveled distance by neurons', step=0,
                  context={"neurons subset": "live"})
    avg_var_live = flat_var_distance[np.logical_not(quasi_dead)]
    avg_var_live = Distribution(avg_var_live, bin_count=100)
    exp_run.track(avg_var_live, name='Variance of the average traveled distance', step=0,
                  context={"neurons subset": "live"})


if __name__ == "__main__":
    run_exp()
