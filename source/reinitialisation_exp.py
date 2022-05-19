"""Experiment where periodic reset of neurons is done while monitoring dead neurons. Can be run over shifting datasets,
a setup that seems to benefit from neurons reinitialisation."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from aim import Run, Figure
import os
import time

from models.mlp import lenet_var_size
import utils.utils as utl
from utils.utils import build_models
from utils.config import dataset_choice, optimizer_choice, regularizer_choice
from copy import deepcopy

if __name__ == "__main__":

    # Configuration
    exp_config = {
        "size": 100,  # Number of hidden units in first layer; size*3 in second hidden layer
        "total_steps": 2001,
        "report_freq": 50,
        "record_freq": 10,
        "switching_period": 500,  # Switch dataset periodically
        "reset_period": 250,  # After reset_period steps, reinitialize the parameters
        "reset_horizon": 1.0,  # Set to lower than one if you want to stop resetting before final steps
        "kept_classes": 3,  # Number of classes in the randomly selected subset
        "compare_full_reset": True,  # Include the comparison with a complete reset of the parameters
        "lr": 1e-3,
        "optimizer": "adam",
        "dataset": "mnist",
        "regularizer": "cdg_l2",
        "reg_param": 1e-4,
    }

    total_neurons = exp_config["size"] + 3 * exp_config["size"]

    assert exp_config["optimizer"] in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config["dataset"] in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config["regularizer"] in regularizer_choice, "Currently supported datasets: " + str(regularizer_choice)

    # Logger config
    exp_name = "reinitialisation_experiment"
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = exp_config

    # Load the dataset
    load_data = dataset_choice[exp_config["dataset"]]
    assert exp_config["kept_classes"] < 10, "subset must be smaller than 10"
    indices = np.random.choice(10, exp_config["kept_classes"], replace=False)
    train = load_data(is_training=True, batch_size=50, subset=indices)
    train_eval = load_data(is_training=True, batch_size=250, subset=indices)
    test_eval = load_data(is_training=False, batch_size=250, subset=indices)
    test_death = load_data(is_training=True, batch_size=1000, subset=indices)

    # Create network/optimizer and initialize params
    architecture = lenet_var_size(exp_config["size"], exp_config["kept_classes"])
    net = build_models(architecture)
    opt = optimizer_choice[exp_config["optimizer"]](exp_config["lr"])

    params = net.init(jax.random.PRNGKey(42-1), next(train))
    opt_state = opt.init(params)
    params_partial_reinit = deepcopy(params)
    opt_partial_reinit_state = opt.init(params_partial_reinit)
    if exp_config["compare_full_reset"]:
        params_hard_reinit = deepcopy(params)
        opt_hard_reinit_state = opt.init(params_hard_reinit)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config["regularizer"], reg_param=exp_config["reg_param"],
                                   classes=exp_config["kept_classes"])
    accuracy_fn = utl.accuracy_given_model(net)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt)

    # Monitoring:
    no_reinit_perf = []
    no_reinit_dead_neurons = []
    partial_reinit_perf = []
    partial_reinit_dead_neurons = []
    if exp_config["compare_full_reset"]:
        hard_reinit_perf = []
        hard_reinit_dead_neurons = []

    # First prng key
    key = jax.random.PRNGKey(0)

    for step in range(exp_config["total_steps"]):
        if step % exp_config["report_freq"] == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            train_loss = loss(params, next(train_eval))
            train_accuracy = accuracy_fn(params, next(train_eval))
            test_accuracy = accuracy_fn(params, next(test_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
            print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                  f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")

        if (step % exp_config["switching_period"] == 0) and \
                (step < exp_config["total_steps"]-exp_config["switching_period"]-1):  # switch task
            # new datasets
            indices = np.random.choice(10, exp_config["kept_classes"], replace=False)
            train = load_data(is_training=True, batch_size=50, subset=indices)
            train_eval = load_data(is_training=True, batch_size=250, subset=indices)
            test_eval = load_data(is_training=False, batch_size=250, subset=indices)
            test_death = load_data(is_training=True, batch_size=1000, subset=indices)

            # reinitialize optimizers state
            opt_state = opt.init(params)
            opt_partial_reinit_state = opt.init(params_partial_reinit)
            if exp_config["compare_full_reset"]:
                opt_hard_reinit_state = opt.init(params_hard_reinit)

        if (step % exp_config["reset_period"] == 0) and (step < exp_config["reset_horizon"]*exp_config["total_steps"]):
            # reinitialize dead neurons
            _, key = jax.random.split(key)
            new_params = net.init(key, next(train))
            dead_neurons = utl.death_check_given_model(net)(params_partial_reinit, next(test_death))
            params_partial_reinit = utl.reinitialize_dead_neurons(dead_neurons, params_partial_reinit, new_params)
            opt_partial_reinit_state = opt.init(params_partial_reinit)

            if exp_config["compare_full_reset"]:
                _, key = jax.random.split(key)
                params_hard_reinit = net.init(key, next(train))
                opt_hard_reinit_state = opt.init(params_hard_reinit)

        if step % exp_config["record_freq"] == 0:
            # Record accuracies
            test_batch = next(test_eval)
            test_accuracy = accuracy_fn(params, test_batch)
            test_accuracy_partial_reinit = accuracy_fn(params_partial_reinit, test_batch)
            no_reinit_perf.append(test_accuracy)
            exp_run.track(jax.device_get(no_reinit_perf[-1]),
                          name="Test accuracy", step=step, context={"reinitialisation": 'None'})
            partial_reinit_perf.append(test_accuracy_partial_reinit)
            exp_run.track(jax.device_get(partial_reinit_perf[-1]),
                          name="Test accuracy", step=step,  context={"reinitialisation": 'Partial'})

            # Record dead neurons
            death_test_batch = next(test_death)
            no_reinit_dead_neurons.append(utl.count_dead_neurons(
                utl.death_check_given_model(net)(params, death_test_batch)))
            exp_run.track(jax.device_get(no_reinit_dead_neurons[-1]),
                          name="Dead neurons", step=step, context={"reinitialisation": 'None'})
            exp_run.track(jax.device_get(no_reinit_dead_neurons[-1]/total_neurons),
                          name="Dead neurons ratio", step=step, context={"reinitialisation": 'None'})
            partial_reinit_dead_neurons.append(utl.count_dead_neurons(
                utl.death_check_given_model(net)(params_partial_reinit, death_test_batch)))
            exp_run.track(jax.device_get(partial_reinit_dead_neurons[-1]),
                          name="Dead neurons", step=step, context={"reinitialisation": 'Partial'})
            exp_run.track(jax.device_get(partial_reinit_dead_neurons[-1]/total_neurons),
                          name="Dead neurons ratio", step=step, context={"reinitialisation": 'Partial'})
            if exp_config["compare_full_reset"]:
                test_accuracy_hard_reinit = jax.device_get(accuracy_fn(params_hard_reinit, test_batch))
                hard_reinit_perf.append(test_accuracy_hard_reinit)
                exp_run.track(jax.device_get(hard_reinit_perf[-1]),
                              name="Test accuracy", step=step, context={"reinitialisation": 'Complete'})
                hard_reinit_dead_neurons.append(utl.count_dead_neurons(
                    utl.death_check_given_model(net)(params_hard_reinit, death_test_batch)))
                exp_run.track(jax.device_get(hard_reinit_dead_neurons[-1]),
                              name="Dead neurons", step=step, context={"reinitialisation": 'Complete'})
                exp_run.track(jax.device_get(hard_reinit_dead_neurons[-1]/total_neurons),
                              name="Dead neurons ratio", step=step, context={"reinitialisation": 'Complete'})

        # Training step
        train_batch = next(train)
        params, opt_state = update_fn(params, opt_state, train_batch)
        params_partial_reinit, opt_partial_reinit_state = update_fn(params_partial_reinit,
                                                                    opt_partial_reinit_state, train_batch)
        if exp_config["compare_full_reset"]:
            params_hard_reinit, opt_hard_reinit_state = update_fn(params_hard_reinit,
                                                                  opt_hard_reinit_state, train_batch)

    # Plots
    dir_path = "./logs/plots/"+exp_name+time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    os.makedirs(dir_path)
    x = list(range(0, exp_config["total_steps"], exp_config["record_freq"]))
    fig1 = plt.figure(figsize=(15, 10))
    plt.plot(x, no_reinit_perf, color='red', label="accuracy, no reinitialisation")
    plt.plot(x, partial_reinit_perf, color='green', label="accuracy, with partial reinitialisation")
    plt.plot(x, np.array(no_reinit_dead_neurons) / total_neurons, color='red', linewidth=3, linestyle=':',
             label="dead neurons, no reinitialisation")
    plt.plot(x, np.array(partial_reinit_dead_neurons) / total_neurons, color='green', linewidth=3, linestyle=':',
             label="dead neurons, with partial reinitialisation")
    if exp_config["compare_full_reset"]:
        plt.plot(x, hard_reinit_perf, color='cyan', label="accuracy, complete reset")
        plt.plot(x, np.array(hard_reinit_dead_neurons) / total_neurons, color='cyan', linewidth=3, linestyle=':',
                 label="dead neurons, complete reset")

    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Inactive neurons/accuracy", fontsize=16)
    plt.title(f"Performance vs dead neurons, switching between {exp_config['kept_classes']}"
              f" classes on {exp_config['dataset']}", fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 12})
    fig1.savefig(dir_path+"perf_vs_dead_neurons.png")
    aim_fig1 = Figure(fig1)
    exp_run.track(aim_fig1, name="Switching task performance w/r to dead neurons", step=0)

    fig2 = plt.figure(figsize=(15, 10))
    plt.plot(no_reinit_dead_neurons, label="without reinitialisation")
    plt.plot(partial_reinit_dead_neurons, label="partial reinitialisation")
    if exp_config["compare_full_reset"]:
        plt.plot(hard_reinit_dead_neurons, label="complete reinitialisation")
    plt.title("Dead neurons during training", fontweight='bold', fontsize=20)
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Inactive neurons", fontsize=16)
    plt.legend(prop={'size': 12})
    fig2.savefig(dir_path+"dead_neurons.png")
    aim_fig2 = Figure(fig2)
    exp_run.track(aim_fig2, name="Dead neurons over training time", step=0)
