"""Experiment where periodic reset of neurons is done while monitoring dead neurons. Can be run over shifting datasets,
a setup that seems to benefit from neurons reinitialisation."""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import matplotlib.pyplot as plt
from aim import Run, Figure, Image
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from ast import literal_eval
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import utils.utils as utl
from utils.utils import build_models
from utils.config import activation_choice, architecture_choice, dataset_choice, dataset_target_cardinality,\
    optimizer_choice, regularizer_choice
from copy import deepcopy

# Experience name -> for aim logger
exp_name = "reinitialisation_experiment"


# Configuration
@dataclass
class ExpConfig:
    total_steps: int = 20001
    report_freq: int = 500
    record_freq: int = 10
    switching_period: int = 2000  # Switch dataset periodically
    reset_period: int = 500  # After reset_period steps, reinitialize the parameters
    reset_horizon: float = 1.0  # Set to lower than one if you want to stop resetting before final steps
    kept_classes: int = 3  # Number of classes in the randomly selected subset
    compare_to_reset: bool = True  # Include comparison with a partial reset of the parameters
    compare_full_reset: bool = True  # Include the comparison with a complete reset of the parameters
    architecture: str = "mlp_3"
    size: Any = 100  # Number of hidden units in the different layers
    lr: float = 1e-3
    train_batch_size: int = 128
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    dataset: str = "mnist"
    activation: str = "relu"  # Activation function used throughout the model
    regularizer: Optional[str] = "None"
    reg_param: float = 1e-4
    init_seed: int = 41
    norm_grad: bool = False


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config.dataset in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(
        architecture_choice.keys())
    assert exp_config.activation in activation_choice.keys(), "Current activation function available: " + str(
        activation_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)

    if not exp_config.compare_to_reset:
        exp_config.compare_full_reset = False

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Retrieve total number of neurons in the model
    total_neurons, _ = utl.get_total_neurons(exp_config.architecture, exp_config.size)

    # Help function to stack parameters
    if exp_config.compare_full_reset:
        def dict_stack(xx, y, z):
            return jnp.stack([xx, y, z])
    else:
        def dict_stack(xx, y):
            return jnp.stack([xx, y])

    # Load the dataset
    load_data = dataset_choice[exp_config.dataset]
    total_classes = dataset_target_cardinality[exp_config.dataset]
    assert exp_config.kept_classes <= total_classes, "subset must be smaller or equal to total number of classes in ds"
    indices = np.random.choice(total_classes, exp_config.kept_classes, replace=False)
    train = load_data(split="train", is_training=True, batch_size=exp_config.train_batch_size, subset=indices)
    train_eval = load_data(split="train", is_training=False, batch_size=exp_config.eval_batch_size, subset=indices)
    test_eval = load_data(split="test", is_training=False, batch_size=exp_config.eval_batch_size, subset=indices)
    test_death = load_data(split="train", is_training=False, batch_size=exp_config.death_batch_size, subset=indices)

    # Create network/optimizer and initialize params
    architecture = architecture_choice[exp_config.architecture]
    architecture = architecture(exp_config.size, exp_config.kept_classes, activation_fn=activation_fn)
    net = build_models(*architecture)
    opt = optimizer_choice[exp_config.optimizer](exp_config.lr)

    # First prng key
    key = jax.random.PRNGKey(exp_config.init_seed)

    params, state = net.init(key, next(train))
    opt_state = opt.init(params)
    if exp_config.compare_to_reset:
        params_partial_reinit = deepcopy(params)
        state_partial_reinit = deepcopy(state)
        opt_partial_reinit_state = opt.init(params_partial_reinit)
        if exp_config.compare_full_reset:
            params_hard_reinit = deepcopy(params)
            state_hard_reinit = deepcopy(state)
            opt_hard_reinit_state = opt.init(params_hard_reinit)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                   classes=exp_config.kept_classes)
    test_loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                        classes=exp_config.kept_classes, is_training=False)
    accuracy_fn = utl.accuracy_given_model(net)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt, norm_grad=exp_config.norm_grad)
    if exp_config.activation in ['tanh']:  # Extend list if needed
        death_check_fn = utl.death_check_given_model(net, epsilon=0.005, check_tail=True)
    else:
        death_check_fn = utl.death_check_given_model(net)

    # Monitoring:
    no_reinit_perf = []
    no_reinit_dead_neurons = []
    if exp_config.compare_to_reset:
        partial_reinit_perf = []
        partial_reinit_dead_neurons = []
        if exp_config.compare_full_reset:
            hard_reinit_perf = []
            hard_reinit_dead_neurons = []

    for step in range(exp_config.total_steps):
        if step % exp_config.report_freq == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            train_loss = test_loss(params, state, next(train_eval))
            train_accuracy = accuracy_fn(params, state, next(train_eval))
            test_accuracy = accuracy_fn(params, state, next(test_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
            print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                  f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")

        if (step % exp_config.switching_period == 0) and \
                (step < exp_config.total_steps-exp_config.switching_period-1):  # switch task
            # new datasets
            indices = np.random.choice(total_classes, exp_config.kept_classes, replace=False)
            train = load_data(split="train", is_training=True, batch_size=exp_config.train_batch_size, subset=indices)
            train_eval = load_data(split="train", is_training=False, batch_size=exp_config.eval_batch_size,
                                   subset=indices)
            test_eval = load_data(split="test", is_training=False, batch_size=exp_config.eval_batch_size,
                                  subset=indices)
            test_death = load_data(split="train", is_training=False, batch_size=exp_config.death_batch_size,
                                   subset=indices)

            # reinitialize optimizers state
            opt_state = opt.init(params)
            if exp_config.compare_to_reset:
                opt_partial_reinit_state = opt.init(params_partial_reinit)
                if exp_config.compare_full_reset:
                    opt_hard_reinit_state = opt.init(params_hard_reinit)

        if (step % exp_config.reset_period == 0) and (step < exp_config.reset_horizon*exp_config.total_steps):
            if exp_config.compare_to_reset:
                # reinitialize dead neurons
                _, key = jax.random.split(key)
                new_params, new_state = net.init(key, next(train))
                dead_neurons = death_check_fn(params_partial_reinit, state_partial_reinit, next(test_death))
                params_partial_reinit = utl.reinitialize_dead_neurons(dead_neurons, params_partial_reinit, new_params)
                state_partial_reinit = new_state
                opt_partial_reinit_state = opt.init(params_partial_reinit)

                if exp_config.compare_full_reset:
                    _, key = jax.random.split(key)
                    params_hard_reinit, state_hard_reinit = net.init(key, next(train))
                    opt_hard_reinit_state = opt.init(params_hard_reinit)

        if step % exp_config.record_freq == 0:
            # Record accuracies
            test_batch = next(test_eval)
            test_accuracy = accuracy_fn(params, state, test_batch)
            no_reinit_perf.append(test_accuracy)
            exp_run.track(np.array(no_reinit_perf[-1]),
                          name="Test accuracy", step=step, context={"reinitialisation": 'None'})

            # Record dead neurons
            death_test_batch = next(test_death)
            no_reinit_dead_neurons.append(utl.count_dead_neurons(
                death_check_fn(params, state, death_test_batch))[0])
            exp_run.track(np.array(no_reinit_dead_neurons[-1]),
                          name="Dead neurons", step=step, context={"reinitialisation": 'None'})
            exp_run.track(np.array(no_reinit_dead_neurons[-1]/total_neurons),
                          name="Dead neurons ratio", step=step, context={"reinitialisation": 'None'})
            if exp_config.compare_to_reset:
                test_accuracy_partial_reinit = accuracy_fn(params_partial_reinit, state_partial_reinit, test_batch)
                partial_reinit_perf.append(test_accuracy_partial_reinit)
                exp_run.track(np.array(partial_reinit_perf[-1]),
                              name="Test accuracy", step=step, context={"reinitialisation": 'Partial'})
                partial_reinit_dead_neurons.append(utl.count_dead_neurons(
                    death_check_fn(params_partial_reinit, state_partial_reinit, death_test_batch))[0])
                exp_run.track(np.array(partial_reinit_dead_neurons[-1]),
                              name="Dead neurons", step=step, context={"reinitialisation": 'Partial'})
                exp_run.track(np.array(partial_reinit_dead_neurons[-1]/total_neurons),
                              name="Dead neurons ratio", step=step, context={"reinitialisation": 'Partial'})
                if exp_config.compare_full_reset:
                    test_accuracy_hard_reinit = np.array(accuracy_fn(params_hard_reinit, state_hard_reinit, test_batch))
                    hard_reinit_perf.append(test_accuracy_hard_reinit)
                    exp_run.track(np.array(hard_reinit_perf[-1]),
                                  name="Test accuracy", step=step, context={"reinitialisation": 'Complete'})
                    hard_reinit_dead_neurons.append(utl.count_dead_neurons(
                        death_check_fn(params_hard_reinit, state_hard_reinit, death_test_batch))[0])
                    exp_run.track(np.array(hard_reinit_dead_neurons[-1]),
                                  name="Dead neurons", step=step, context={"reinitialisation": 'Complete'})
                    exp_run.track(np.array(hard_reinit_dead_neurons[-1]/total_neurons),
                                  name="Dead neurons ratio", step=step, context={"reinitialisation": 'Complete'})

        # Training step
        train_batch = next(train)
        # Train in parallel
        if exp_config.compare_full_reset:
            all_params = jax.tree_map(dict_stack, params, params_partial_reinit, params_hard_reinit)
            all_states = jax.tree_map(dict_stack, state, state_partial_reinit, state_hard_reinit)
            all_opt_states = jax.tree_map(dict_stack, opt_state, opt_partial_reinit_state, opt_hard_reinit_state)
        elif exp_config.compare_to_reset:
            all_params = jax.tree_map(dict_stack, params, params_partial_reinit)
            all_states = jax.tree_map(dict_stack, state, state_partial_reinit)
            all_opt_states = jax.tree_map(dict_stack, opt_state, opt_partial_reinit_state)
        else:
            all_params = jax.tree_map(Partial(jnp.expand_dims, axis=0), params)
            all_states = jax.tree_map(Partial(jnp.expand_dims, axis=0), state)
            all_opt_states = jax.tree_map(Partial(jnp.expand_dims, axis=0), opt_state)

        all_params, all_states, all_opt_states = jax.vmap(update_fn, in_axes=(
        utl.vmap_axes_mapping(params), utl.vmap_axes_mapping(state),
        utl.vmap_axes_mapping(opt_state), None))(
            all_params,
            all_states,
            all_opt_states,
            train_batch)

        if exp_config.compare_full_reset:
            params, params_partial_reinit, params_hard_reinit = utl.dict_split(all_params)
            state, state_partial_reinit, state_hard_reinit = utl.dict_split(all_states)
            opt_state, opt_partial_reinit_state, opt_hard_reinit_state = utl.dict_split(all_opt_states)
        elif exp_config.compare_to_reset:
            params, params_partial_reinit = utl.dict_split(all_params)
            state, state_partial_reinit = utl.dict_split(all_states)
            opt_state, opt_partial_reinit_state = utl.dict_split(all_opt_states)
        else:
            params = jax.tree_map(jnp.squeeze, all_params)
            state = jax.tree_map(jnp.squeeze, all_states)
            opt_state = jax.tree_map(jnp.squeeze, all_opt_states)

        # Train sequentially the networks instead than in parallel
        # params, opt_state = update_fn(params, opt_state, train_batch)
        # params_partial_reinit, opt_partial_reinit_state = update_fn(params_partial_reinit,
        #                                                             opt_partial_reinit_state, train_batch)
        # if exp_config.compare_full_reset:
        #     params_hard_reinit, opt_hard_reinit_state = update_fn(params_hard_reinit,
        #                                                           opt_hard_reinit_state, train_batch)

    # Plots
    dir_path = "./logs/plots/"+exp_name+time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    os.makedirs(dir_path)

    x = list(range(0, exp_config.total_steps, exp_config.record_freq))
    fig1 = plt.figure(figsize=(15, 10))
    plt.plot(x, no_reinit_perf, color='red', label="accuracy, no reinitialisation")
    plt.plot(x, np.array(no_reinit_dead_neurons) / total_neurons, color='red', linewidth=3, linestyle=':',
             label="dead neurons, no reinitialisation")
    if exp_config.compare_to_reset:
        plt.plot(x, partial_reinit_perf, color='green', label="accuracy, with partial reinitialisation")
        plt.plot(x, np.array(partial_reinit_dead_neurons) / total_neurons, color='green', linewidth=3, linestyle=':',
                 label="dead neurons, with partial reinitialisation")
        if exp_config.compare_full_reset:
            plt.plot(x, hard_reinit_perf, color='cyan', label="accuracy, complete reset")
            plt.plot(x, np.array(hard_reinit_dead_neurons) / total_neurons, color='cyan', linewidth=3, linestyle=':',
                     label="dead neurons, complete reset")

    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Inactive neurons (ratio)/accuracy", fontsize=16)
    plt.title(f"Performance vs dead neurons, switching between {exp_config.kept_classes}"
              f" classes on {exp_config.dataset}", fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 12})
    fig1.savefig(dir_path+"perf_vs_dead_neurons.png")
    aim_fig1 = Figure(fig1)
    aim_img1 = Image(fig1)
    exp_run.track(aim_fig1, name="Switching task performance w/r to dead neurons", step=0)
    exp_run.track(aim_img1, name="Switching task performance w/r to dead neurons; img", step=0)

    fig2 = plt.figure(figsize=(15, 10))
    plt.plot(no_reinit_dead_neurons, label="without reinitialisation")
    if exp_config.compare_to_reset:
        plt.plot(partial_reinit_dead_neurons, label="partial reinitialisation")
        if exp_config.compare_full_reset:
            plt.plot(hard_reinit_dead_neurons, label="complete reinitialisation")
    plt.title("Dead neurons during training", fontweight='bold', fontsize=20)
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Inactive neurons", fontsize=16)
    plt.legend(prop={'size': 12})
    fig2.savefig(dir_path+"dead_neurons.png")
    aim_fig2 = Figure(fig2)
    aim_img2 = Image(fig2)
    exp_run.track(aim_fig2, name="Dead neurons over training time", step=0)
    exp_run.track(aim_img2, name="Dead neurons over training time; img", step=0)


if __name__ == "__main__":
    run_exp()
