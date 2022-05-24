"""Experiment where we try to verify if solving on a easier task before switching to an easier one is hurtful on
generalisation and is linked to the amount of dead neurons. Also run on an MLP"""

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
        "total_steps": 20001,
        "report_freq": 500,
        "record_freq": 10,
        "compare_full_reset": True,  # Include the comparison with a complete reset of the parameters
        "lr": 1e-3,
        "optimizer": "adam",
        "datasets": ("fashion mnist", "fashion mnist"),  # Datasets to use, listed from easier to harder
        "kept_classes": (5, None),  # Number of classes to use, listed from easier to harder
        "regularizer": "cdg_l2",
        "reg_param": 1e-4,
    }

    total_neurons = exp_config["size"] + 3 * exp_config["size"]

    assert exp_config["optimizer"] in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config["dataset"] in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config["regularizer"] in regularizer_choice, "Currently supported datasets: " + str(regularizer_choice)

    # Logger config
    exp_name = "easier_harder_switch_experiment"
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = exp_config

    # Load the dataset for the 2 tasks (ez and hard)
    load_data_easier = dataset_choice[exp_config["datasets"][0]]
    if exp_config["kept_classes"][0]:
        indices = np.random.choice(10, exp_config["kept_classes"][0], replace=False)
    else:
        indices = None
    train_easier = load_data_easier(is_training=True, batch_size=50, subset=indices)
    train_eval_easier = load_data_easier(is_training=True, batch_size=250, subset=indices)
    test_eval_easier = load_data_easier(is_training=False, batch_size=250, subset=indices)
    test_death_easier = load_data_easier(is_training=True, batch_size=1000, subset=indices)

    load_data_harder = dataset_choice[exp_config["datasets"][1]]
    if exp_config["kept_classes"][1]:
        indices = np.random.choice(10, exp_config["kept_classes"][1], replace=False)
    else:
        indices = None
    train_harder = load_data_harder(is_training=True, batch_size=50, subset=indices)
    train_eval_harder = load_data_harder(is_training=True, batch_size=250, subset=indices)
    test_eval_harder = load_data_harder(is_training=False, batch_size=250, subset=indices)
    test_death_harder = load_data_harder(is_training=True, batch_size=1000, subset=indices)

    train = [train_easier, train_harder]
    train_eval = [train_eval_easier, train_eval_harder]
    test_eval = [test_eval_easier, test_eval_harder]
    test_death = [test_death_easier, test_death_harder]

    # Create network/optimizer and initialize params
    architecture = lenet_var_size(exp_config["size"], exp_config["kept_classes"])
    net = build_models(architecture)
    opt = optimizer_choice[exp_config["optimizer"]](exp_config["lr"])

    params = net.init(jax.random.PRNGKey(42 - 1), next(train_easier))
    opt_state = opt.init(params)
    params_partial_reinit = deepcopy(params)
    opt_partial_reinit_state = opt.init(params_partial_reinit)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config["regularizer"], reg_param=exp_config["reg_param"],
                                   classes=exp_config["kept_classes"][0])
    accuracy_fn = utl.accuracy_given_model(net)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt)

    # First prng key
    key = jax.random.PRNGKey(0)

    for order in [0, -1]:
        # Monitoring:
        no_reinit_perf = []
        no_reinit_dead_neurons = []
        partial_reinit_perf = []
        partial_reinit_dead_neurons = []
        idx = order

        for step in range(exp_config["total_steps"]):
            if (step-1)//2 == 0:  # Switch task at mid training
                # reinitialize dead neurons
                _, key = jax.random.split(key)
                new_params = net.init(key, next(train[idx]))
                dead_neurons = utl.death_check_given_model(net)(params_partial_reinit, next(test_death[idx]))
                params_partial_reinit = utl.reinitialize_dead_neurons(dead_neurons, params_partial_reinit, new_params)
                opt_partial_reinit_state = opt.init(params_partial_reinit)

                # switch task
                idx += -1

            if step % exp_config["report_freq"] == 0:
                # Periodically evaluate classification accuracy on train & test sets.
                train_loss = loss(params, next(train_eval[idx]))
                train_accuracy = accuracy_fn(params, next(train_eval[idx]))
                test_accuracy = accuracy_fn(params, next(test_eval[idx]))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                      f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
