""" Experiment trying to measure the effective capacity in overfitting regime with
the final number of live neurons. Knowing that if left to train for a long period neurons keep dying until a plateau is
reached and that accuracy behave in the same manner (convergence/overfitting regime) the set up is as follow:
For a given dataset and a given architecture, vary the width (to increase capacity) and measure the number of live
neurons after reaching the overfitting regime and the plateau. Empirical observation: Even with increased capacity, the
number of live neurons at the end eventually also reaches a plateau."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from aim import Run, Figure

from models.mlp import lenet_var_size
import utils.utils as utl
from utils.utils import build_models
from utils.config import optimizer_choice, dataset_choice, regularizer_choice

if __name__ == "__main__":

    # Configuration
    exp_config = {
        "training_steps": 20001,  # 20001
        "report_freq": 500,  # 500
        "lr": 1e-3,
        "optimizer": "adam",
        "dataset": "mnist",
        "regularizer": "cdg_l2",
        "reg_param": 1e-4,
    }

    assert exp_config["optimizer"] in optimizer_choice.keys(), "Currently supported optimizers: " + str(optimizer_choice.keys())
    assert exp_config["dataset"] in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config["regularizer"] in regularizer_choice, "Currently supported datasets: " + str(regularizer_choice)

    # Logger config
    exp_run = Run(repo="./logs", experiment="asymptotic_live_neurons__lenet")
    exp_run["configuration"] = exp_config

    # Load the different dataset
    load_data = dataset_choice[exp_config["dataset"]]
    train = load_data(is_training=True, batch_size=50)
    # tf_train = load_mnist_tf("train", is_training=True, batch_size=50)

    train_eval = load_data(is_training=True, batch_size=500)
    test_eval = load_data(is_training=False, batch_size=500)
    final_test_eval = load_data(is_training=False, batch_size=10000)

    test_death = load_data(is_training=True, batch_size=1000)
    final_test_death = load_data(is_training=True, batch_size=60000)

    # Recording over all widths
    live_neurons = []
    size_arr = []
    f_acc = []

    for size in [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000]:  # Vary the NN width
        # Make the network and optimiser
        architecture = lenet_var_size(size, 10)
        net = build_models(architecture)

        opt = optimizer_choice[exp_config["optimizer"]](exp_config["lr"])
        dead_neurons_log = []
        accuracies_log = []

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config["regularizer"], reg_param=exp_config["reg_param"])
        accuracy_fn = utl.accuracy_given_model(net)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt)

        params = net.init(jax.random.PRNGKey(42 - 1), next(train))
        opt_state = opt.init(params)

        for step in range(exp_config["training_steps"]):
            if step % exp_config["report_freq"] == 0:
                # Periodically evaluate classification accuracy on train & test sets.
                train_loss = loss(params, next(train_eval))
                train_accuracy = accuracy_fn(params, next(train_eval))
                test_accuracy = accuracy_fn(params, next(test_eval))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                      f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")

                dead_neurons = utl.death_check_given_model(net)(params, next(test_death))

                # Record some metrics
                dead_neurons_count = utl.count_dead_neurons(dead_neurons)
                dead_neurons_log.append(dead_neurons_count)
                accuracies_log.append(test_accuracy)
                exp_run.track(jax.device_get(dead_neurons_count), name=f"Dead neurons, lenet size={size}", step=step)
                exp_run.track(test_accuracy, name=f"Accuracy, lenet size={size}", step=step)

            # Train step over single batch
            params, opt_state = update_fn(params, opt_state, next(train))

        total_neurons = size + 3*size  # TODO: Adapt to other NN than 2-hidden MLP
        final_accuracy = jax.device_get(accuracy_fn(params, next(final_test_eval)))
        size_arr.append(total_neurons)
        final_dead_neurons = utl.death_check_given_model(net)(params, next(final_test_death))
        final_dead_neurons_count = utl.count_dead_neurons(final_dead_neurons)

        exp_run.track(jax.device_get(total_neurons - final_dead_neurons_count),
                      name="Live neurons after convergence w/r total neurons", step=total_neurons)
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r total neurons", step=total_neurons)

        live_neurons.append(total_neurons - final_dead_neurons_count)
        f_acc.append(final_accuracy)

    # Plots
    fig1 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, live_neurons, label="Live neurons", linewidth=4)
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Live neurons at end of training", fontsize=16)
    plt.title("Effective capacity, 2-hidden layers MLP on "+exp_config["dataset"], fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    aim_fig1 = Figure(fig1)
    exp_run.track(aim_fig1, name="effective capacity")

    fig2 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, jnp.array(live_neurons) / jnp.array(size_arr), label="alive ratio")
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Ratio of live neurons at end of training", fontsize=16)
    plt.title("MLP effective capacity on "+exp_config["dataset"], fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    aim_fig2 = Figure(fig2)
    exp_run.track(aim_fig2, name="live neurons ratio")

    fig3 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, f_acc, label="accuracy", linewidth=4)
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Final accuracy", fontsize=16)
    plt.title("Performance, 2-hidden layers MLP on "+exp_config["dataset"], fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    aim_fig3 = Figure(fig3)
    exp_run.track(aim_fig3, name="final performance")
