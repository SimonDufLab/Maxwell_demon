""" Experiment trying to measure the effective capacity in overfitting regime with
the final number of live neurons. Knowing that if left to train for a long period neurons keep dying until a plateau is
reached and that accuracy behave in the same manner (convergence/overfitting regime) the set up is as follow:
For a given dataset and a given architecture, vary the width (to increase capacity) and measure the number of live
neurons after reaching the overfitting regime and the plateau. Empirical observation: Even with increased capacity, the
number of live neurons at the end eventually also reaches a plateau."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from aim import Run, Figure
import os
import time
from dataclasses import dataclass
from typing import Optional
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from models.mlp import lenet_var_size
import utils.utils as utl
from utils.utils import build_models
from utils.config import optimizer_choice, dataset_choice, regularizer_choice

# Experience name -> for aim logger
exp_name = "asymptotic_live_neurons__lenet"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001  # 20001
    report_freq: int = 500  # 500
    lr: float = 1e-3
    optimizer: str = "adam"
    dataset: str = "mnist"
    regularizer: Optional[str] = "cdg_l2"
    reg_param: float = 1e-4
    epsilon_close: float = 0.0  # Relaxing criterion for dead neurons, epsilon-close to relu gate


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(optimizer_choice.keys())
    assert exp_config.dataset in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Load the different dataset
    load_data = dataset_choice[exp_config.dataset]
    train = load_data(split="train", is_training=True, batch_size=50)

    train_eval = load_data(split="train", is_training=False, batch_size=500)
    test_eval = load_data(split="test", is_training=False, batch_size=500)
    final_test_eval = load_data(split="test", is_training=False, batch_size=10000)

    test_death = load_data(split="train", is_training=False, batch_size=1000)
    final_test_death = load_data(split="train", is_training=False, batch_size=60000)

    # Recording over all widths
    live_neurons = []
    size_arr = []
    f_acc = []

    for size in [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000]:  # Vary the NN width
        # Make the network and optimiser
        architecture = lenet_var_size(size, 10)
        net = build_models(architecture)

        opt = optimizer_choice[exp_config.optimizer](exp_config.lr)
        dead_neurons_log = []
        accuracies_log = []

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param)
        accuracy_fn = utl.accuracy_given_model(net)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt)

        params = net.init(jax.random.PRNGKey(42 - 1), next(train))
        opt_state = opt.init(params)

        for step in range(exp_config.training_steps):
            if step % exp_config.report_freq == 0:
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
                exp_run.track(np.array(dead_neurons_count), name="Dead neurons", step=step,
                              context={"lenet size": f"{size}"})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"lenet size": f"{size}"})

            # Train step over single batch
            params, opt_state = update_fn(params, opt_state, next(train))

        total_neurons = size + 3*size  # TODO: Adapt to other NN than 2-hidden MLP
        final_accuracy = np.array(accuracy_fn(params, next(final_test_eval)))
        size_arr.append(total_neurons)
        final_dead_neurons = utl.death_check_given_model(net)(params, next(final_test_death))
        final_dead_neurons_count = utl.count_dead_neurons(final_dead_neurons)

        exp_run.track(np.array(total_neurons - final_dead_neurons_count),
                      name="Live neurons after convergence w/r total neurons", step=total_neurons)
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r total neurons", step=total_neurons)

        live_neurons.append(total_neurons - final_dead_neurons_count)
        f_acc.append(final_accuracy)

    # Plots
    dir_path = "./logs/plots/" + exp_name + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    os.makedirs(dir_path)

    fig1 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, live_neurons, label="Live neurons", linewidth=4)
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Live neurons at end of training", fontsize=16)
    plt.title("Effective capacity, 2-hidden layers MLP on "+exp_config.dataset, fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    fig1.savefig(dir_path+"effective_capacity.png")
    aim_fig1 = Figure(fig1)
    exp_run.track(aim_fig1, name="Effective capacity", step=0)

    fig2 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, jnp.array(live_neurons) / jnp.array(size_arr), label="alive ratio")
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Ratio of live neurons at end of training", fontsize=16)
    plt.title("MLP effective capacity on "+exp_config.dataset, fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    fig2.savefig(dir_path+"live_neurons_ratio.png")
    aim_fig2 = Figure(fig2)
    exp_run.track(aim_fig2, name="Live neurons ratio", step=0)

    fig3 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, f_acc, label="accuracy", linewidth=4)
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Final accuracy", fontsize=16)
    plt.title("Performance at convergence, 2-hidden layers MLP on "+exp_config.dataset, fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    fig3.savefig(dir_path+"performance_at_convergence.png")
    aim_fig3 = Figure(fig3)
    exp_run.track(aim_fig3, name="Performance at convergence", step=0)


if __name__ == "__main__":
    run_exp()
