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
from utils.config import optimizer_choice, dataset_choice, dataset_target_cardinality, regularizer_choice, architecture_choice, bn_architecture_choice
from utils.config import activations_pre_relu, activations_with_bn


# Experience name -> for aim logger
exp_name = "batchnorm_exp"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    record_freq: int = 100
    snapshot_freq: int = 10_000
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
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


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
    assert exp_config.architecture in bn_architecture_choice.keys(), "Current architectures available: " + str(bn_architecture_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
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

    def desired_activations(context, architecture):
        if context == 'Without BN':
            return activations_pre_relu[architecture]
        if context == 'With BN':
            return activations_with_bn[architecture]

    labels_dict = {
        'Without BN': ("pre_relu",) ,
        'With BN': ("pre_bn", "post_bn")
    }

    for arc_choice, context in ((architecture_choice, 'Without BN'), (bn_architecture_choice, 'With BN')):
        # Time the subrun, with or without BN
        subrun_start_time = time.time()

        # Make the network and optimiser
        architecture = arc_choice[exp_config.architecture]
        classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
        architecture = architecture(exp_config.size, classes)
        net = build_models(*architecture)
        opt = optimizer_choice[exp_config.optimizer](exp_config.lr)

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param)
        test_loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                            is_training=False)
        accuracy_fn = utl.accuracy_given_model(net)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp)
        death_check_fn = utl.death_check_given_model(net)
        scan_len = dataset_size // exp_config.death_batch_size
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
            utl.death_check_given_model(net, with_activations=True), scan_len, True)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_size // eval_size)

        params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
        opt_state = opt.init(params)

        starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, exp_config.size)
        total_neurons, total_per_layer = starting_neurons, starting_per_layer
        num_layers = len(starting_per_layer)

        for step in range(exp_config.training_steps):
            if step % exp_config.record_freq == 0:
                train_loss = test_loss(params, state, next(train_eval))
                train_accuracy = accuracy_fn(params, state, next(train_eval))
                test_accuracy = accuracy_fn(params, state, next(test_eval))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                # Periodically print classification accuracy on train & test sets.
                if step % exp_config.report_freq == 0:
                    print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                          f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
                dead_neurons = death_check_fn(params, state, next(test_death))
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                              context={'Normalization': context})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step,
                              context={'Normalization': context})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={'Normalization': context})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={'Normalization': context})
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                              context={'Normalization': context})

            if step % exp_config.snapshot_freq == 0:
                num_col = 2
                dist_start_time = time.time()
                num_rows = 1 + (num_layers-1)//num_col
                fig, axs = plt.subplots(num_rows, num_col, figsize=(20, 5*num_rows))
                if axs.ndim == 1:
                    axs = np.expand_dims(axs, axis=0)
                for k, _arch in enumerate(desired_activations(context, exp_config.architecture)):
                    _label = labels_dict[context][k]
                    _arch = _arch(exp_config.size, classes)
                    _net = build_models(*_arch)
                    (_, activations), _ = _net.apply(params, state, next(test_death), True, False)
                    mean_activations = jax.tree_map(Partial(jnp.mean, axis=0), activations)
                    bin_limit = jnp.max(jnp.abs(ravel_pytree(mean_activations)[0]))
                    bins = jnp.linspace(-bin_limit, bin_limit, 100)
                    mean_activations = jax.device_get(mean_activations)
                    for j, layer_act in enumerate(mean_activations):
                        # layer_act = jnp.mean(layer_act, axis=0)
                        axs[j//num_col, j % num_col].hist(layer_act, bins, alpha=0.5, label=f"{context} :{_label}", histtype='bar')
                        axs[j // num_col, j % num_col].set_title(f"layer {j}")

                for ax in axs.flat:
                    ax.set(xlabel='activations value', ylabel='count')

                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in axs.flat:
                    ax.label_outer()

                # Legend:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels)

                # aim_fig = Figure(fig)
                aim_img = Image(fig)
                # exp_run.track(aim_fig, name=f"Activations distribution {context}", step=step)
                exp_run.track(aim_img, name=f"Activations distribution {context}; img", step=step)

                print()
                print(f"Activations dist generated in: " + str(timedelta(seconds=time.time() - dist_start_time)))
                print("----------------------------------------------")
                print()

            params, state, opt_state = update_fn(params, state, opt_state, next(train))

        activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)

        activations_max, activations_mean, activations_count = activations_data
        activations_max, _ = ravel_pytree(activations_max)
        activations_max = jax.device_get(activations_max)
        activations_mean, _ = ravel_pytree(activations_mean)
        activations_mean = jax.device_get(activations_mean)
        activations_count, _ = ravel_pytree(activations_count)
        activations_count = jax.device_get(activations_count)

        activations_max_dist = Distribution(activations_max, bin_count=100)
        exp_run.track(activations_max_dist, name='Maximum activation distribution after convergence', step=0,
                      context={'Normalization': context})
        activations_mean_dist = Distribution(activations_mean, bin_count=100)
        exp_run.track(activations_mean_dist, name='Mean activation distribution after convergence', step=0,
                      context={'Normalization': context})
        activations_count_dist = Distribution(activations_count, bin_count=50)
        exp_run.track(activations_count_dist, name='Activation count per neuron after convergence', step=0,
                      context={'Normalization': context})

        # Print running time
        print()
        print(f"Running time for {context}: " + str(timedelta(seconds=time.time() - subrun_start_time)))
        print("----------------------------------------------")
        print()

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
