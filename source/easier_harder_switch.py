"""Experiment where we try to verify if solving on an easier task before switching to a harder one is hurtful on
generalisation and is linked to the amount of dead neurons. Also run on an MLP"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from aim import Run, Figure, Image
import os
import time
from dataclasses import dataclass
from typing import Any, Tuple, Union, Optional
from ast import literal_eval
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from jax.tree_util import Partial

import utils.utils as utl
from utils.utils import build_models
from utils.config import activation_choice, architecture_choice, dataset_choice, optimizer_choice, regularizer_choice
from utils.config import pick_architecture, bn_config_choice
from copy import deepcopy

# Experience name -> for aim logger
exp_name = "easier_harder_switch_experiment"


# Configuration
@dataclass
class ExpConfig:
    total_steps: int = 700001
    report_freq: int = 3000
    record_freq: int = 250
    full_checkpoint_freq: int = 5000
    lr: float = 1e-3
    train_batch_size: int = 128
    eval_batch_size: int = 512
    death_batch_size: int = 512
    architecture: str = "mlp_3"
    with_bn: bool = False  # Add batchnorm layers or not in the models
    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
    size: Any = 256  # Number of hidden units in the different layers (tuple for convnet)
    optimizer: str = "adam"
    wd_param: Optional[float] = None
    activation: str = "relu"  # Activation function used throughout the model
    datasets: Any = ("mnist", "fashion mnist")  # Datasets to use, listed from easier to harder
    reverse_order: bool = True  # If true, also train from harder to easier
    kept_classes: Any = (None, None)  # Number of classes to use, listed from easier to harder
    regularizer: Optional[str] = 'None'
    # compare_to_full_reset: bool = True
    compare_to_partial_reset: bool = False
    reinit_state: bool = False  # Reinit optimizer and params state at switch task
    reg_param: float = 1e-4
    perturb_param: float = 0  # Perturbation parameter for rnadam
    init_seed: int = 41
    rng_seed: int = 777
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:

    total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, exp_config.size)
    dataset_total_classes = 10  # TODO: allow compatibility with dataset > 10 classes

    if type(exp_config.datasets) == str:
        exp_config.datasets = utl.add_comma_in_str(exp_config.datasets)
        exp_config.datasets = literal_eval(exp_config.datasets)
    if type(exp_config.kept_classes) == str:
        exp_config.kept_classes = literal_eval(exp_config.kept_classes)

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config.datasets[0] in dataset_choice.keys(), "Currently supported datasets: " + str(
        dataset_choice.keys())
    assert exp_config.datasets[1] in dataset_choice.keys(), "Currently supported datasets: " + str(
        dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(
        architecture_choice.keys())
    assert exp_config.activation in activation_choice.keys(), "Current activation function available: " + str(
        activation_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if exp_config.wd_param == 'None':
        exp_config.wd_param = None
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)

    if 'None' in exp_config.kept_classes:  # TODO: Probably a better way than to accept None as argument...
        kept_classes = list(exp_config.kept_classes)
        for i, item in enumerate(kept_classes):
            if item == 'None':
                kept_classes[i] = None
        exp_config.kept_classes = tuple(kept_classes)

    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())

    net_config = {}
    if exp_config.with_bn:
        assert exp_config.architecture in pick_architecture(
            with_bn=True).keys(), "Current architectures available with batchnorm: " + str(
            pick_architecture(with_bn=True).keys())
        net_config['bn_config'] = bn_config_choice[exp_config.bn_config]

    activation_fn = activation_choice[exp_config.activation]

    # Logger config
    exp_run = Run(repo="./rnadam_easier_harder", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Help function to stack parameters
    if exp_config.compare_to_partial_reset:
        def dict_stack(xx, y, z):
            return jnp.stack([xx, y, z])
    else:
        def dict_stack(xx, y):
            return jnp.stack([xx, y])

    # Helper function for logging dead neurons statistics w/r to whole ds and validation acc
    def log_whole_ds_deaths_and_valid_acc(dead_neurons_count, dead_per_layers, valid_acc, step, context):
        exp_run.track(jax.device_get(dead_neurons_count), name=f"Dead neurons w/r whole ds; {setting[order]}",
                      step=step,
                      context={"reinitialisation": context})
        exp_run.track(jax.device_get(dead_neurons_count/total_neurons),
                      name=f"Dead neurons ratio w/r whole ds; {setting[order]}",
                      step=step,
                      context={"reinitialisation": context})
        exp_run.track(jax.device_get(valid_acc),
                      name=f"Validation full accuracy; {setting[order]}",
                      step=step,
                      context={"reinitialisation": context})
        for i, layer_dead in enumerate(dead_per_layers):
            total_neuron_in_layer = total_per_layer[i]
            exp_run.track(jax.device_get(layer_dead),
                          name=f"Dead neurons in layer {i}; w/r whole ds; {setting[order]}", step=step,
                          context={"reinitialisation": context})
            exp_run.track(jax.device_get(layer_dead/total_neuron_in_layer),
                          name=f"Dead neurons ratio in layer {i}; w/r whole ds; {setting[order]}", step=step,
                          context={"reinitialisation": context})

    # Load the dataset for the 2 tasks (ez and hard)
    load_data_easier = dataset_choice[exp_config.datasets[0]]
    if exp_config.kept_classes[0]:
        indices = np.random.choice(10, exp_config.kept_classes[0], replace=False)
    else:
        indices = None
    train_easier = load_data_easier(split='train', is_training=True, batch_size=exp_config.train_batch_size, subset=indices, transform=False)
    train_eval_easier = load_data_easier(split='train', is_training=False, batch_size=exp_config.eval_batch_size, subset=indices, transform=False)
    test_size, test_eval_easier = load_data_easier(split='test', is_training=False, batch_size=exp_config.eval_batch_size, subset=indices, transform=False, cardinality=True)
    dataset_size, test_death_easier = load_data_easier(split='train', is_training=False, batch_size=exp_config.death_batch_size, subset=indices, transform=False, cardinality=True)

    load_data_harder = dataset_choice[exp_config.datasets[1]]
    if exp_config.kept_classes[1]:
        indices = np.random.choice(10, exp_config.kept_classes[1], replace=False)
    else:
        indices = None
    train_harder = load_data_harder(split='train', is_training=True, batch_size=exp_config.train_batch_size, subset=indices, transform=False)
    train_eval_harder = load_data_harder(split='train', is_training=False, batch_size=exp_config.eval_batch_size, subset=indices, transform=False)
    test_eval_harder = load_data_harder(split='test', is_training=False, batch_size=exp_config.eval_batch_size, subset=indices, transform=False)
    test_death_harder = load_data_harder(split='train', is_training=False, batch_size=exp_config.death_batch_size, subset=indices, transform=False)

    train = [train_easier, train_harder]
    train_eval = [train_eval_easier, train_eval_harder]
    test_eval = [test_eval_easier, test_eval_harder]
    test_death = [test_death_easier, test_death_harder]

    # Create network/optimizer and initialize params
    architecture = pick_architecture(with_bn=exp_config.with_bn)[exp_config.architecture]
    architecture = architecture(exp_config.size, dataset_total_classes, activation_fn=activation_fn, **net_config)
    net, _ = build_models(*architecture)
    ones_init = jnp.ones_like(next(train_easier)[0]), jnp.ones_like(next(train_easier)[1])
    init_fn = utl.get_init_fn(net, ones_init)
    opt = optimizer_choice[exp_config.optimizer]
    if "w" in exp_config.optimizer and exp_config.wd_param:  # Pass reg_param to wd argument of adamw
        opt = Partial(opt, weight_decay=exp_config.wd_param)
    opt = opt(exp_config.lr)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                   classes=dataset_total_classes)
    test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                        classes=dataset_total_classes, is_training=False)
    accuracy_fn = utl.accuracy_given_model(net)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt, perturb=exp_config.perturb_param, init_fn=init_fn)
    scan_len = dataset_size // exp_config.death_batch_size
    scan_death_check_fn = utl.scanned_death_check_fn(utl.death_check_given_model(net), scan_len)
    full_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // exp_config.eval_batch_size)

    # First prng key
    key = jax.random.PRNGKey(exp_config.init_seed)
    rnadam_key = jax.random.PRNGKey(exp_config.rng_seed)

    setting = ["easier to harder", "harder to easier"]
    # dir_path = "./logs/plots/" + exp_name + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    # os.makedirs(dir_path)
    if exp_config.reverse_order:
        tasks_order = [0, -1]
    else:
        tasks_order = [0]

    for order in tasks_order:
        # Initialize params
        params, state = net.init(key, next(train_easier))
        opt_state = opt.init(params)
        # params_partial_reinit = deepcopy(params)
        # state_partial_reinit = deepcopy(state)
        # opt_partial_reinit_state = opt.init(params_partial_reinit)

        # Monitoring:
        no_reinit_perf = []
        no_reinit_dead_neurons = []
        full_reset_perf = []
        full_reset_dead_neurons = []
        partial_reinit_perf = []
        partial_reinit_dead_neurons = []
        idx = order

        after_switch = False

        for step in range(exp_config.total_steps):
            if (step+1) / (exp_config.total_steps//2) == 1:  # Switch task at mid-training
                # switching task
                after_switch = True

                # compare to full reset (no warm startup):
                _, key = jax.random.split(key)
                params_full_reset, state_full_reset = net.init(key, next(train[idx]))
                opt_full_reset_state = opt.init(params_full_reset)
                if exp_config.reinit_state:
                    _, state = net.init(key, next(train[idx]))  # No need to split key
                    opt_state = opt.init(params)
                # reinitialize dead neurons
                if exp_config.compare_to_partial_reset:
                    _, key = jax.random.split(key)
                    new_params, _ = net.init(key, next(train[idx]))
                    dead_neurons = utl.death_check_given_model(net)(params, state, next(test_death[idx]))
                    params_partial_reinit = utl.reinitialize_dead_neurons(dead_neurons, params, new_params)
                    state_partial_reinit = deepcopy(state)
                    opt_partial_reinit_state = opt.init(params_partial_reinit)

                # switch task
                idx += -1

            if step % exp_config.report_freq == 0:
                # Periodically evaluate classification accuracy on train & test sets.
                train_loss = test_loss_fn(params, state, next(train_eval[idx]))
                train_accuracy = accuracy_fn(params, state, next(train_eval[idx]))
                test_accuracy = accuracy_fn(params, state, next(test_eval[idx]))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                      f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")

            if step % exp_config.record_freq == 0:
                # Record accuracies
                test_batch = next(test_eval[idx])
                test_accuracy = accuracy_fn(params, state, test_batch)
                test_loss = test_loss_fn(params, state, test_batch)
                no_reinit_perf.append(test_accuracy)
                exp_run.track(np.array(no_reinit_perf[-1]),
                              name=f"Test accuracy; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                exp_run.track(np.array(test_loss),
                              name=f"Loss; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                if after_switch:
                    test_accuracy_full_reset = accuracy_fn(params_full_reset, state_full_reset, test_batch)
                    test_loss_full_reset = test_loss_fn(params_full_reset, state_full_reset, test_batch)
                    full_reset_perf.append(test_accuracy_full_reset)
                    exp_run.track(np.array(full_reset_perf[-1]),
                                  name=f"Test accuracy; {setting[order]}", step=step,
                                  context={"reinitialisation": 'Full'})
                    exp_run.track(np.array(test_loss_full_reset),
                                  name=f"Loss; {setting[order]}", step=step, context={"reinitialisation": 'Full'})
                    if exp_config.compare_to_partial_reset:
                        test_accuracy_partial_reinit = accuracy_fn(params_partial_reinit, state_partial_reinit, test_batch)
                        test_loss_partial_reinit = test_loss_fn(params_partial_reinit, state_partial_reinit, test_batch)
                        partial_reinit_perf.append(test_accuracy_partial_reinit)
                        exp_run.track(np.array(partial_reinit_perf[-1]),
                                      name=f"Test accuracy; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})
                        exp_run.track(np.array(test_loss_partial_reinit),
                                      name=f"Loss; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})

                # Record dead neurons
                death_test_batch = next(test_death[idx])
                no_reinit_dead_neurons.append(utl.count_dead_neurons(
                    utl.death_check_given_model(net)(params, state, death_test_batch))[0])
                exp_run.track(np.array(no_reinit_dead_neurons[-1]),
                              name=f"Dead neurons; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                exp_run.track(np.array(no_reinit_dead_neurons[-1] / total_neurons),
                              name=f"Dead neurons ratio; {setting[order]}", step=step, context={"reinitialisation": 'None'})
                if after_switch:
                    full_reset_dead_neurons.append(utl.count_dead_neurons(
                        utl.death_check_given_model(net)(params_full_reset, state_full_reset,
                                                         death_test_batch))[0])
                    exp_run.track(np.array(full_reset_dead_neurons[-1]),
                                  name=f"Dead neurons; {setting[order]}", step=step,
                                  context={"reinitialisation": 'Full'})
                    exp_run.track(np.array(full_reset_dead_neurons[-1] / total_neurons),
                                  name=f"Dead neurons ratio; {setting[order]}", step=step,
                                  context={"reinitialisation": 'Full'})
                    if exp_config.compare_to_partial_reset:
                        partial_reinit_dead_neurons.append(utl.count_dead_neurons(
                            utl.death_check_given_model(net)(params_partial_reinit, state_partial_reinit, death_test_batch))[0])
                        exp_run.track(np.array(partial_reinit_dead_neurons[-1]),
                                      name=f"Dead neurons; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})
                        exp_run.track(np.array(partial_reinit_dead_neurons[-1] / total_neurons),
                                      name=f"Dead neurons ratio; {setting[order]}", step=step, context={"reinitialisation": 'Partial'})

            if step % exp_config.full_checkpoint_freq == 0:
                dead_neurons = scan_death_check_fn(params, state, test_death[idx])
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)
                validation_accuracy = full_accuracy_fn(params, state, test_eval[idx])
                log_whole_ds_deaths_and_valid_acc(dead_neurons_count, dead_per_layers, validation_accuracy, step,
                                                  'None')

                if after_switch:
                    full_dead_neurons = scan_death_check_fn(params_full_reset, state_full_reset, test_death[idx])
                    full_dead_neurons_count, full_dead_per_layers = utl.count_dead_neurons(full_dead_neurons)
                    full_reset_validation_accuracy = full_accuracy_fn(params_full_reset, state_full_reset,
                                                                      test_eval[idx])
                    log_whole_ds_deaths_and_valid_acc(full_dead_neurons_count, full_dead_per_layers,
                                                      full_reset_validation_accuracy, step, 'Full')

                    if exp_config.compare_to_partial_reset:
                        partial_dead_neurons = scan_death_check_fn(params_partial_reinit, state_partial_reinit,
                                                                   test_death[idx])
                        partial_dead_neurons_count, partial_dead_per_layers = utl.count_dead_neurons(
                            partial_dead_neurons)
                        partial_validation_accuracy = full_accuracy_fn(params_partial_reinit, state_partial_reinit,
                                                                       test_eval[idx])
                        log_whole_ds_deaths_and_valid_acc(partial_dead_neurons_count, partial_dead_per_layers,
                                                          partial_validation_accuracy, step, 'Partial')

            # Training step
            train_batch = next(train[idx])
            if after_switch:
                # Train in parallel
                if exp_config.compare_to_partial_reset:
                    all_params = jax.tree_map(dict_stack, params, params_full_reset, params_partial_reinit)
                    all_states = jax.tree_map(dict_stack, state, state_full_reset, state_partial_reinit)
                    all_opt_states = jax.tree_map(dict_stack, opt_state, opt_full_reset_state, opt_partial_reinit_state)
                    if exp_config.perturb_param:
                        rnadam_key, temp_keys = jax.random.split(rnadam_key)
                        temp_keys = jax.random.split(temp_keys, 3)
                        all_params, all_states, all_opt_states, _ = jax.vmap(update_fn, in_axes=(
                            utl.vmap_axes_mapping(params), utl.vmap_axes_mapping(state),
                            utl.vmap_axes_mapping(opt_state), None, 0))(
                            all_params,
                            all_states,
                            all_opt_states,
                            train_batch,
                            temp_keys)
                    else:
                        all_params, all_states, all_opt_states = jax.vmap(update_fn, in_axes=(
                            utl.vmap_axes_mapping(params), utl.vmap_axes_mapping(state),
                            utl.vmap_axes_mapping(opt_state), None))(
                            all_params,
                            all_states,
                            all_opt_states,
                            train_batch)
                    params, params_full_reset, params_partial_reinit = utl.dict_split(all_params)
                    state, state_full_reset, state_partial_reinit = utl.dict_split(all_states, _len=3)
                    opt_state, opt_full_reset_state, opt_partial_reinit_state = utl.dict_split(all_opt_states)

                else:
                    all_params = jax.tree_map(dict_stack, params, params_full_reset)
                    all_states = jax.tree_map(dict_stack, state, state_full_reset)
                    all_opt_states = jax.tree_map(dict_stack, opt_state, opt_full_reset_state)
                    if exp_config.perturb_param:
                        rnadam_key, temp_keys = jax.random.split(rnadam_key)
                        temp_keys = jax.random.split(temp_keys, 2)
                        all_params, all_states, all_opt_states, _ = jax.vmap(update_fn, in_axes=(
                            utl.vmap_axes_mapping(params), utl.vmap_axes_mapping(state),
                            utl.vmap_axes_mapping(opt_state), None, 0))(
                            all_params,
                            all_states,
                            all_opt_states,
                            train_batch,
                            temp_keys)
                    else:
                        all_params, all_states, all_opt_states = jax.vmap(update_fn, in_axes=(
                            utl.vmap_axes_mapping(params), utl.vmap_axes_mapping(state),
                            utl.vmap_axes_mapping(opt_state), None))(
                            all_params,
                            all_states,
                            all_opt_states,
                            train_batch)
                    params, params_full_reset = utl.dict_split(all_params)
                    state, state_full_reset = utl.dict_split(all_states)
                    opt_state, opt_full_reset_state = utl.dict_split(all_opt_states)

            else:
                if exp_config.perturb_param:
                    params, state, opt_state, rnadam_key = update_fn(params, state, opt_state, train_batch, rnadam_key)
                else:
                    params, state, opt_state = update_fn(params, state, opt_state, train_batch)

            # Train sequentially instead
            # params, state, opt_state = update_fn(params, opt_state, train_batch)
            # params_partial_reinit, state_partial_reinit, opt_partial_reinit_state = update_fn(params_partial_reinit,
            #                                                             opt_partial_reinit_state, train_batch)

        # # Plots
        # x = list(range(0, exp_config.total_steps, exp_config.record_freq))
        # xx = list(range(exp_config.total_steps // 2,
        #                 exp_config.total_steps // 2 + exp_config.record_freq * len(full_reset_perf),
        #                 exp_config.record_freq))
        # fig = plt.figure(figsize=(25, 15))
        # plt.plot(x, no_reinit_perf, color='red', label="accuracy, no reinitialisation")
        # plt.plot(xx, full_reset_perf, color='blue', label="accuracy, with full reinitialisation")
        # plt.plot(x, np.array(no_reinit_dead_neurons) / total_neurons, color='red', linewidth=3, linestyle=':',
        #          label="dead neurons, no reinitialisation")
        # plt.plot(xx, np.array(full_reset_dead_neurons) / total_neurons, color='blue', linewidth=3, linestyle=':',
        #          label="dead neurons, with full reinitialisation")
        # if exp_config.compare_to_partial_reset:
        #     plt.plot(xx, partial_reinit_perf, color='green', label="accuracy, with partial reinitialisation")
        #     plt.plot(xx, np.array(partial_reinit_dead_neurons) / total_neurons, color='green', linewidth=3, linestyle=':',
        #              label="dead neurons, with partial reinitialisation")
        # plt.xlabel("Iterations", fontsize=16)
        # plt.ylabel("Inactive neurons (ratio)/accuracy", fontsize=16)
        # plt.title(f"From {setting[order]} switch experiment"
        #           f" on {exp_config.datasets[0]}/{exp_config.datasets[1]}"
        #           f", keeping {exp_config.kept_classes[0]}/{exp_config.kept_classes[1]} classes",
        #           fontweight='bold', fontsize=20)
        # plt.legend(prop={'size': 12})
        # fig.savefig(dir_path + f"{setting[order]}.png")
        # # aim_fig = Figure(fig)
        # aim_img = Image(fig)
        # # exp_run.track(aim_fig, name=f"From {setting[order]} experiment", step=0)
        # exp_run.track(aim_img, name=f"From {setting[order]} experiment; img", step=0)


if __name__ == "__main__":
    run_exp()
