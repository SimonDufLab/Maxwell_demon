"""Experiment where periodic reset of neurons is done while monitoring dead neurons. Can be run over shifting datasets,
a setup that seems to benefit from neurons reinitialisation."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from models.mlp import lenet_var_size
import utils.utils as utl
from utils.utils import load_mnist, build_models
from copy import deepcopy

if __name__ == "__main__":

    size = 100  # Number of hidden units in first layer; size*3 in second hidden layer
    total_steps = 20001
    report_freq = 500
    record_freq = 10
    switching_period = 1000  # Switch dataset periodically
    reset_period = 250  # After reset_period steps, reinitialize the parameters
    reset_horizon = 1.0  # Set to lower than one if you want to stop resetting before final steps
    kept_classes = 3  # Number of classes in the randomly selected subset
    compare_full_reset = False  # Additional training where a complete reset of the parameters is done (vs a partial)

    # Load the dataset
    assert kept_classes < 10, "subset must be smaller than 10"
    indices = np.random.choice(10, kept_classes, replace=False)
    train = load_mnist(is_training=True, batch_size=50, subset=indices)
    train_eval = load_mnist(is_training=True, batch_size=250, subset=indices)
    test_eval = load_mnist(is_training=False, batch_size=250, subset=indices)
    test_death = load_mnist(is_training=True, batch_size=1000, subset=indices)

    # Create network/optimizer and initialize params
    architecture = lenet_var_size(size, kept_classes)
    net = build_models(architecture)
    opt = optax.adam(1e-3)

    params = net.init(jax.random.PRNGKey(42-1), next(train))
    opt_state = opt.init(params)
    params_partial_reinit = deepcopy(params)
    opt_partial_reinit_state = opt.init(params_partial_reinit)
    if compare_full_reset:
        params_hard_reinit = deepcopy(params)
        opt_hard_reinit_state = opt.init(params_hard_reinit)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer="cdg_l2", reg_param=1e-4, classes=kept_classes)
    accuracy_fn = utl.accuracy_given_model(net)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt)

    # Monitoring:
    no_reinit_perf = []
    no_reinit_dead_neurons = []
    partial_reinit_perf = []
    partial_reinit_dead_neurons = []
    if compare_full_reset:
        hard_reinit_perf = []
        hard_reinit_dead_neurons = []

    # First prng key
    key = jax.random.PRNGKey(0)

    for step in range(total_steps):
        if step % report_freq == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            train_loss = loss(params, next(train_eval))
            train_accuracy = accuracy_fn(params, next(train_eval))
            test_accuracy = accuracy_fn(params, next(test_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
            print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                  f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")

        if (step % switching_period == 0) and (step < total_steps-switching_period-1):  # switch task
            # new datasets
            indices = np.random.choice(10, kept_classes, replace=False)
            train = load_mnist(is_training=True, batch_size=50, subset=indices)
            train_eval = load_mnist(is_training=True, batch_size=250, subset=indices)
            test_eval = load_mnist(is_training=False, batch_size=250, subset=indices)
            test_death = load_mnist(is_training=True, batch_size=1000, subset=indices)

            # reinitialize optimizers state
            opt_state = opt.init(params)
            opt_partial_reinit_state = opt.init(params_partial_reinit)
            if compare_full_reset:
                opt_hard_reinit_state = opt.init(params_hard_reinit)

        if (step % reset_period == 0) and (step < reset_horizon*total_steps):
            # reinitialize dead neurons
            _, key = jax.random.split(key)
            new_params = net.init(key, next(train))
            dead_neurons = utl.death_check_given_model(net)(params_partial_reinit, next(test_death))
            params_partial_reinit = utl.reinitialize_dead_neurons(dead_neurons, params_partial_reinit, new_params)
            opt_partial_reinit_state = opt.init(params_partial_reinit)

            if compare_full_reset:
                _, key = jax.random.split(key)
                params_hard_reinit = net.init(key, next(train))
                opt_hard_reinit_state = opt.init(params_hard_reinit)

        if step % record_freq == 0:
            # Record accuracies
            test_batch = next(test_eval)
            test_accuracy = accuracy_fn(params, test_batch)
            test_accuracy_partial_reinit = accuracy_fn(params_partial_reinit, test_batch)
            no_reinit_perf.append(test_accuracy)
            partial_reinit_perf.append(test_accuracy_partial_reinit)

            # Record dean neurons
            death_test_batch = next(test_death)
            no_reinit_dead_neurons.append(utl.count_dead_neurons(utl.death_check_given_model(net)(params,
                                                                                                  death_test_batch)))
            partial_reinit_dead_neurons.append(utl.count_dead_neurons(
                utl.death_check_given_model(net)(params_partial_reinit, death_test_batch)))
            if compare_full_reset:
                test_accuracy_hard_reinit = jax.device_get(accuracy_fn(params_hard_reinit, test_batch))
                hard_reinit_perf.append(test_accuracy_hard_reinit)
                hard_reinit_dead_neurons.append(utl.count_dead_neurons(
                    utl.death_check_given_model(net)(params_hard_reinit, death_test_batch)))

        # Training step
        train_batch = next(train)
        params, opt_state = update_fn(params, opt_state, train_batch)
        params_partial_reinit, opt_partial_reinit_state = update_fn(params_partial_reinit,
                                                                    opt_partial_reinit_state, train_batch)
        if compare_full_reset:
            params_hard_reinit, opt_hard_reinit_state = update_fn(params_hard_reinit,
                                                                  opt_hard_reinit_state, train_batch)

    # Plots
    total_neurons = size + 3 * size
    x = list(range(0, total_steps, record_freq))
    plt.figure(figsize=(15, 10))
    plt.plot(x, no_reinit_perf, color='red', linewidth=1, label="accuracy, no reinitialisation")
    plt.plot(x, partial_reinit_perf, color='green', linewidth=1, label="accuracy, with partial reinitialisation")
    plt.plot(x, np.array(no_reinit_dead_neurons) / total_neurons, color='red', linewidth=3, linestyle=':',
             label="dead neurons, no reinitialisation")
    plt.plot(x, np.array(partial_reinit_dead_neurons) / total_neurons, color='green', linewidth=3, linestyle=':',
             label="dead neurons, with partial reinitialisation")
    if compare_full_reset:
        plt.plot(x, hard_reinit_perf, color='blue', linewidth=1, label="accuracy, complete reset")
        plt.plot(x, np.array(hard_reinit_dead_neurons) / total_neurons, color='blue', linewidth=3, linestyle=':',
                 label="dead neurons, complete reset")

    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Inactive neurons/accuracy", fontsize=16)
    plt.title(f"Relevant title, {kept_classes} classes", fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(no_reinit_dead_neurons, label="without reinitialisation")
    plt.plot(partial_reinit_dead_neurons, label="with reinitialisation")
    plt.legend()
    plt.show()
