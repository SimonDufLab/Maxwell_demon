""" Experiment trying to measure the effective capacity in overfitting regime with
the final number of live neurons. Knowing that if left to train for a long period neurons keep dying until a plateau is
reached and that accuracy behave in the same manner (convergence/overfitting regime) the set up is as follow:
For a given dataset and a given architecture, vary the width (to increase capacity) and measure the number of live
neurons after reaching the overfitting regime and the plateau. Empirical observation: Even with increased capacity, the
number of live neurons at the end eventually also reaches a plateau."""

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
from utils.config import optimizer_choice, dataset_choice, regularizer_choice, architecture_choice

# Experience name -> for aim logger
exp_name = "asymptotic_live_neurons"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    record_freq: int = 100
    pruning_freq: int = 2000
    live_freq: int = 20000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
    lr: float = 1e-3
    train_batch_size: int = 128
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    dataset: str = "mnist"
    classes: int = 10  # Number of classes in the training dataset
    architecture: str = "mlp_3"
    sizes: Any = (50, 100, 250, 500, 750, 1000, 1250, 1500, 2000)
    regularizer: Optional[str] = "cdg_l2"
    reg_param: float = 1e-4
    epsilon_close: float = 0.0  # Relaxing criterion for dead neurons, epsilon-close to relu gate
    init_seed: int = 41
    dynamic_pruning: bool = False

    # def __post_init__(self):
    #     if type(self.sizes) == str:
    #         self.sizes = literal_eval(self.sizes)


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
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(architecture_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.sizes) == str:
        exp_config.sizes = literal_eval(exp_config.sizes)

    if exp_config.dynamic_pruning:
        exp_name_ = exp_name+"_with_dynamic_pruning"
    else:
        exp_name_ = exp_name

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name_)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Create pickle directory
    pickle_dir_path = "./logs/metadata/" + exp_name_ + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
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
    # final_test_eval = load_data(split="test", is_training=False, batch_size=10000)

    death_minibatch_size = exp_config.death_batch_size
    dataset_size, test_death = load_data(split="train", is_training=False,
                                         batch_size=death_minibatch_size, cardinality=True)
    # final_test_death = load_data(split="train", is_training=False, batch_size=dataset_size)

    # Recording over all widths
    live_neurons = []
    avg_live_neurons = []
    std_live_neurons = []
    size_arr = []
    f_acc = []

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

    for size in exp_config.sizes:  # Vary the NN width
        # Time the subrun for the different sizes
        subrun_start_time = time.time()

        # Make the network and optimiser
        architecture = architecture_choice[exp_config.architecture](size, exp_config.classes)
        net = build_models(architecture)

        opt = optimizer_choice[exp_config.optimizer](exp_config.lr)
        accuracies_log = []

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param)
        accuracy_fn = utl.accuracy_given_model(net)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt)
        death_check_fn = utl.death_check_given_model(net)
        scan_len = dataset_size // death_minibatch_size
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
            utl.death_check_given_model(net, with_activations=True), scan_len, True)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_size // eval_size)

        params = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
        opt_state = opt.init(params)

        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, size)

        for step in range(exp_config.training_steps):
            if step % exp_config.record_freq == 0:
                train_loss = loss(params, next(train_eval))
                train_accuracy = accuracy_fn(params, next(train_eval))
                test_accuracy = accuracy_fn(params, next(test_eval))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                # Periodically print classification accuracy on train & test sets.
                if step % exp_config.report_freq == 0:
                    print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                          f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
                dead_neurons = death_check_fn(params, next(test_death))
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                accuracies_log.append(test_accuracy)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                              context={"net size": utl.size_to_string(size)})

            if step % exp_config.pruning_freq == 0:
                dead_neurons = scan_death_check_fn(params, test_death)
                # dead_neurons = jax.tree_map(utl.logical_and_sum, dead_neurons)
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)
                if exp_config.dynamic_pruning:
                    # Pruning the network
                    params, opt_state, new_sizes = utl.remove_dead_neurons_weights(params, dead_neurons, opt_state)
                    architecture = architecture_choice[exp_config.architecture](new_sizes, exp_config.classes)
                    net = build_models(architecture)

                    # Recompile training/monitoring functions
                    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param)
                    accuracy_fn = utl.accuracy_given_model(net)
                    update_fn = utl.update_given_loss_and_optimizer(loss, opt)
                    death_check_fn = utl.death_check_given_model(net)
                    scan_len = dataset_size // death_minibatch_size
                    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
                    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
                        utl.death_check_given_model(net, with_activations=True), scan_len, True)
                    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
                    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_size // eval_size)
                del dead_neurons  # Freeing memory
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset", step=step,
                              context={"net size": utl.size_to_string(size)})
                for i, layer_dead in enumerate(dead_per_layers):
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                                  context={"net size": utl.size_to_string(size)})
                del dead_per_layers
                train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, train_eval))
                exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                              step=step,
                              context={"net size": utl.size_to_string(size)})

            if ((step+1) % exp_config.live_freq == 0) and (step+2 < exp_config.training_steps):
                current_dead_neurons = scan_death_check_fn(params, test_death)
                current_dead_neurons_count, _ = utl.count_dead_neurons(current_dead_neurons)
                del current_dead_neurons
                del _
                exp_run.track(jax.device_get(total_neurons - current_dead_neurons_count),
                              name=f"Live neurons at training step {step+1}", step=total_neurons)

            # Train step over single batch
            params, opt_state = update_fn(params, opt_state, next(train))

        # final_accuracy = jax.device_get(accuracy_fn(params, next(final_test_eval)))
        final_accuracy = jax.device_get(final_accuracy_fn(params, test_eval))
        size_arr.append(total_neurons)

        activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, test_death)
        # final_dead_neurons = scan_death_check_fn(params, test_death)

        # final_dead_neurons = jax.tree_map(utl.logical_and_sum, batched_dead_neurons)
        final_dead_neurons_count, final_dead_per_layer = utl.count_dead_neurons(final_dead_neurons)
        del final_dead_neurons  # Freeing memory

        activations_max, activations_mean, activations_count = activations_data
        activations_meta.maximum.append(activations_max)
        activations_meta.mean.append(activations_mean)
        activations_meta.count.append(activations_count)
        activations_max, _ = ravel_pytree(activations_max)
        activations_max = jax.device_get(activations_max)
        activations_mean, _ = ravel_pytree(activations_mean)
        activations_mean = jax.device_get(activations_mean)
        activations_count, _ = ravel_pytree(activations_count)
        activations_count = jax.device_get(activations_count)

        # Additionally, track an 'on average' number of death neurons within a batch
        # def scan_f(_, __):
        #     _, batch_dead_neurons = utl.death_check_given_model(net, with_activations=True)(params, next(test_death))
        #     return None, total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0]
        # _, batches_final_live_neurons = jax.lax.scan(scan_f, None, None, scan_len)
        batch_dead_neurons = death_check_fn(params, next(test_death))
        batches_final_live_neurons = [total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0]]
        for i in range(scan_len-1):
            batch_dead_neurons = death_check_fn(params, next(test_death))
            batches_final_live_neurons.append(total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0])
        batches_final_live_neurons = jnp.stack(batches_final_live_neurons)

        avg_final_live_neurons = jnp.mean(batches_final_live_neurons, axis=0)
        std_final_live_neurons = jnp.std(batches_final_live_neurons, axis=0)

        exp_run.track(jax.device_get(avg_final_live_neurons),
                      name="On average, live neurons after convergence w/r total neurons", step=total_neurons)
        exp_run.track(jax.device_get(total_neurons - final_dead_neurons_count),
                      name="Live neurons after convergence w/r total neurons", step=total_neurons)
        for i, layer_dead in enumerate(final_dead_per_layer):
            total_neuron_in_layer = total_per_layer[i]
            exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                          name=f"Live neurons in layer {i} after convergence w/r total neurons",
                          step=total_neurons)  # Either total_neuron_in_layer/total_neurons
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r total neurons", step=total_neurons)
        activations_max_dist = Distribution(activations_max, bin_count=100)
        exp_run.track(activations_max_dist, name='Maximum activation distribution after convergence', step=0,
                      context={"net size": utl.size_to_string(size)})
        activations_mean_dist = Distribution(activations_mean, bin_count=100)
        exp_run.track(activations_mean_dist, name='Mean activation distribution after convergence', step=0,
                      context={"net size": utl.size_to_string(size)})
        activations_count_dist = Distribution(activations_count, bin_count=50)
        exp_run.track(activations_count_dist, name='Activation count per neuron after convergence', step=0,
                      context={"net size": utl.size_to_string(size)})

        live_neurons.append(total_neurons - final_dead_neurons_count)
        avg_live_neurons.append(avg_final_live_neurons)
        std_live_neurons.append(std_final_live_neurons)
        f_acc.append(final_accuracy)

        # Making sure compiled fn cache was cleared
        loss._clear_cache()
        accuracy_fn._clear_cache()
        update_fn._clear_cache()
        death_check_fn._clear_cache()
        # scan_death_check_fn._clear_cache()  # No more cache
        # scan_death_check_fn_with_activations._clear_cache()  # No more cache
        # final_accuracy_fn._clear_cache()  # No more cache

        # Pickling activations for later epsilon-close investigation in a .ipynb
        with open(pickle_dir_path+'activations_meta.p', 'wb') as fp:
            pickle.dump(asdict(activations_meta), fp)  # Update by overwrite

        # Pickling the final parameters value as well
        params_meta.parameters.append(params)
        with open(pickle_dir_path + 'params_meta.p', 'wb') as fp:
            pickle.dump(asdict(params_meta), fp)  # Update by overwrite

        # Print running time
        print()
        print(f"Running time for size {size}: " + str(timedelta(seconds=time.time()-subrun_start_time)))
        print("----------------------------------------------")
        print()

    # Plots
    dir_path = "./logs/plots/" + exp_name_ + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    os.makedirs(dir_path)

    fig1 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, live_neurons, label="Live neurons", linewidth=4)
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Live neurons at end of training", fontsize=16)
    plt.title("Effective capacity, "+exp_config.architecture+" on "+exp_config.dataset, fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    fig1.savefig(dir_path+"effective_capacity.png")
    aim_fig1 = Figure(fig1)
    aim_img1 = Image(fig1)
    exp_run.track(aim_fig1, name="Effective capacity", step=0)
    exp_run.track(aim_img1, name="Effective capacity; img", step=0)

    fig1_5 = plt.figure(figsize=(15, 10))
    plt.errorbar(size_arr, avg_live_neurons, std_live_neurons, label="Live neurons", linewidth=4)
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Average live neurons at end of training", fontsize=16)
    plt.title((f"Effective capacity averaged on minibatch of size={death_minibatch_size}, "+exp_config.architecture+" on "
               + exp_config.dataset), fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    fig1_5.savefig(dir_path+"avg_effective_capacity.png")
    aim_img1_5 = Image(fig1_5)
    exp_run.track(aim_img1_5, name="Average effective capacity; img", step=0)

    fig2 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, jnp.array(live_neurons) / jnp.array(size_arr), label="alive ratio")
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Ratio of live neurons at end of training", fontsize=16)
    plt.title(exp_config.architecture+" effective capacity on "+exp_config.dataset, fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    fig2.savefig(dir_path+"live_neurons_ratio.png")
    aim_fig2 = Figure(fig2)
    aim_img2 = Image(fig2)
    exp_run.track(aim_fig2, name="Live neurons ratio", step=0)
    exp_run.track(aim_img2, name="Live neurons ratio; img", step=0)

    fig3 = plt.figure(figsize=(15, 10))
    plt.plot(size_arr, f_acc, label="accuracy", linewidth=4)
    plt.xlabel("Number of neurons in NN", fontsize=16)
    plt.ylabel("Final accuracy", fontsize=16)
    plt.title("Performance at convergence, "+exp_config.architecture+" on "+exp_config.dataset, fontweight='bold', fontsize=20)
    plt.legend(prop={'size': 16})
    fig3.savefig(dir_path+"performance_at_convergence.png")
    aim_fig3 = Figure(fig3)
    aim_img3 = Image(fig3)
    exp_run.track(aim_fig3, name="Performance at convergence", step=0)
    exp_run.track(aim_img3, name="Performance at convergence; img", step=0)

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
