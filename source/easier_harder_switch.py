"""Experiment where we try to verify if solving on an easier task before switching to a harder one is hurtful on
generalisation and is linked to the amount of dead neurons. Also run on an MLP"""

import jax
import numpy as np
import matplotlib.pyplot as plt
from aim import Run, Figure
import os
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Union, Optional
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from models.mlp import lenet_var_size
import utils.utils as utl
from utils.utils import build_models
from utils.config import dataset_choice, optimizer_choice, regularizer_choice
from copy import deepcopy

# Experience name -> for aim logger
exp_name = "easier_harder_switch_experiment"


# Configuration
@dataclass
class ExpConfig:
    size: int = 100  # Number of hidden units in first layer; size*3 in second hidden layer
    total_steps: int = 10001
    report_freq: int = 500
    record_freq: int = 10
    lr: float = 1e-3
    optimizer: str = "adam"
    datasets: Tuple[str] = ("mnist", "fashion mnist")  # Datasets to use, listed from easier to harder
    kept_classes: Tuple[Union[int, None]] = (None, None)  # Number of classes to use, listed from easier to harder
    regularizer: Optional[str] = "cdg_l2"
    reg_param: float = 1e-4


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:

    total_neurons = exp_config.size + 3 * exp_config.size
    dataset_total_classes = 10  # TODO: allow compatibility with dataset > 10 classes

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config.datasets[0] in dataset_choice.keys(), "Currently supported datasets: " + str(
        dataset_choice.keys())
    assert exp_config.datasets[1] in dataset_choice.keys(), "Currently supported datasets: " + str(
        dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None

    if 'None' in exp_config.kept_classes:  # TODO: Probably a better way than to accept None as argument...
        kept_classes = list(exp_config.kept_classes)
        for i, item in enumerate(kept_classes):
            if item == 'None':
                kept_classes[i] = None
        exp_config.kept_classes = tuple(kept_classes)

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Load the dataset for the 2 tasks (ez and hard)
    load_data_easier = dataset_choice[exp_config.datasets[0]]
    if exp_config.kept_classes[0]:
        indices = np.random.choice(10, exp_config.kept_classes[0], replace=False)
    else:
        indices = None
    train_easier = load_data_easier(split='train', is_training=True, batch_size=50, subset=indices, transform=False)
    train_eval_easier = load_data_easier(split='train', is_training=False, batch_size=250, subset=indices, transform=False)
    test_eval_easier = load_data_easier(split='test', is_training=False, batch_size=250, subset=indices, transform=False)
    test_death_easier = load_data_easier(split='train', is_training=False, batch_size=1000, subset=indices, transform=False)

    load_data_harder = dataset_choice[exp_config.datasets[1]]
    if exp_config.kept_classes[1]:
        indices = np.random.choice(10, exp_config.kept_classes[1], replace=False)
    else:
        indices = None
    train_harder = load_data_harder(split='train', is_training=True, batch_size=50, subset=indices, transform=False)
    train_eval_harder = load_data_harder(split='train', is_training=False, batch_size=250, subset=indices, transform=False)
    test_eval_harder = load_data_harder(split='test', is_training=False, batch_size=250, subset=indices, transform=False)
    test_death_harder = load_data_harder(split='train', is_training=False, batch_size=1000, subset=indices, transform=False)

    train = [train_easier, train_harder]
    train_eval = [train_eval_easier, train_eval_harder]
    test_eval = [test_eval_easier, test_eval_harder]
    test_death = [test_death_easier, test_death_harder]

    # Create network/optimizer and initialize params
    architecture = lenet_var_size(exp_config.size, dataset_total_classes)
    net = build_models(architecture)
    opt = optimizer_choice[exp_config.optimizer](exp_config.lr)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                   classes=dataset_total_classes)
    accuracy_fn = utl.accuracy_given_model(net)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt)

    # First prng key
    key = jax.random.PRNGKey(0)

    setting = ["easier to harder", "harder to easier"]
    dir_path = "./logs/plots/" + exp_name + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    os.makedirs(dir_path)

    for order in [0, -1]:
        # Initialize params
        params = net.init(jax.random.PRNGKey(42 - 1), next(train_easier))
        opt_state = opt.init(params)
        params_partial_reinit = deepcopy(params)
        opt_partial_reinit_state = opt.init(params_partial_reinit)

        # Monitoring:
        no_reinit_perf = []
        no_reinit_dead_neurons = []
        partial_reinit_perf = []
        partial_reinit_dead_neurons = []
        idx = order

        for step in range(exp_config.total_steps):
            if (step+1) / (exp_config.total_steps//2) == 1:  # Switch task at mid-training
                # reinitialize dead neurons
                _, key = jax.random.split(key)
                new_params = net.init(key, next(train[idx]))
                dead_neurons = utl.death_check_given_model(net)(params_partial_reinit, next(test_death[idx]))
                params_partial_reinit = utl.reinitialize_dead_neurons(dead_neurons, params_partial_reinit, new_params)
                opt_partial_reinit_state = opt.init(params_partial_reinit)

                # switch task
                idx += -1

            if step % exp_config.report_freq == 0:
                # Periodically evaluate classification accuracy on train & test sets.
                train_loss = loss(params, next(train_eval[idx]))
                train_accuracy = accuracy_fn(params, next(train_eval[idx]))
                test_accuracy = accuracy_fn(params, next(test_eval[idx]))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                      f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")

            if step % exp_config.record_freq == 0:
                # Record accuracies
                test_batch = next(test_eval[idx])
                test_accuracy = accuracy_fn(params, test_batch)
                test_accuracy_partial_reinit = accuracy_fn(params_partial_reinit, test_batch)
                test_loss = loss(params, test_batch)
                test_loss_partial_reinit = loss(params_partial_reinit, test_batch)
                no_reinit_perf.append(test_accuracy)
                exp_run.track(np.array(no_reinit_perf[-1]),
                              name=f"Test accuracy; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                partial_reinit_perf.append(test_accuracy_partial_reinit)
                exp_run.track(np.array(partial_reinit_perf[-1]),
                              name=f"Test accuracy; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})
                exp_run.track(np.array(test_loss),
                              name=f"Loss; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                exp_run.track(np.array(test_loss_partial_reinit),
                              name=f"Loss; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})

                # Record dead neurons
                death_test_batch = next(test_death[idx])
                no_reinit_dead_neurons.append(utl.count_dead_neurons(
                    utl.death_check_given_model(net)(params, death_test_batch)))
                exp_run.track(np.array(no_reinit_dead_neurons[-1]),
                              name=f"Dead neurons; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                exp_run.track(np.array(no_reinit_dead_neurons[-1] / total_neurons),
                              name=f"Dead neurons ratio; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                partial_reinit_dead_neurons.append(utl.count_dead_neurons(
                    utl.death_check_given_model(net)(params_partial_reinit, death_test_batch)))
                exp_run.track(np.array(partial_reinit_dead_neurons[-1]),
                              name=f"Dead neurons; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})
                exp_run.track(np.array(partial_reinit_dead_neurons[-1] / total_neurons),
                              name=f"Dead neurons ratio; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})

            # Training step
            train_batch = next(train[idx])
            params, opt_state = update_fn(params, opt_state, train_batch)
            params_partial_reinit, opt_partial_reinit_state = update_fn(params_partial_reinit,
                                                                        opt_partial_reinit_state, train_batch)

        # Plots
        x = list(range(0, exp_config.total_steps, exp_config.record_freq))
        fig = plt.figure(figsize=(25, 15))
        plt.plot(x, no_reinit_perf, color='red', label="accuracy, no reinitialisation")
        plt.plot(x, partial_reinit_perf, color='green', label="accuracy, with partial reinitialisation")
        plt.plot(x, np.array(no_reinit_dead_neurons) / total_neurons, color='red', linewidth=3, linestyle=':',
                 label="dead neurons, no reinitialisation")
        plt.plot(x, np.array(partial_reinit_dead_neurons) / total_neurons, color='green', linewidth=3, linestyle=':',
                 label="dead neurons, with partial reinitialisation")
        plt.xlabel("Iterations", fontsize=16)
        plt.ylabel("Inactive neurons (ratio)/accuracy", fontsize=16)
        plt.title(f"From {setting[order]} switch experiment"
                  f" on {exp_config.datasets[0]}/{exp_config.datasets[1]}"
                  f", keeping {exp_config.kept_classes[0]}/{exp_config.kept_classes[1]} classes",
                  fontweight='bold', fontsize=20)
        plt.legend(prop={'size': 12})
        fig.savefig(dir_path + f"{setting[order]}.png")
        aim_fig = Figure(fig)
        exp_run.track(aim_fig, name=f"From {setting[order]} experiment", step=0)


if __name__ == "__main__":
    run_exp()
