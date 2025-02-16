""" Experiment trying to measure the effective capacity in overfitting regime with
the final number of live neurons. Knowing that if left to train for a long period neurons keep dying until a plateau is
reached and that accuracy behave in the same manner (convergence/overfitting regime) the set up is as follow:
For a given dataset and a given architecture, vary the width (to increase capacity) and measure the number of live
neurons after reaching the overfitting regime and the plateau. Empirical observation: Even with increased capacity, the
number of live neurons at the end eventually also reaches a plateau."""

import copy
import optax
import jax
import jax.numpy as jnp
import haiku as hk
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
import omegaconf.listconfig

from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree

import utils.utils as utl
from utils.utils import build_models
from utils.config import activation_choice, optimizer_choice, dataset_choice, dataset_target_cardinality
from utils.config import regularizer_choice, architecture_choice, lr_scheduler_choice, bn_config_choice
from utils.config import pick_architecture


# Experience name -> for aim logger
exp_name = "asymptotic_live_neurons"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 120001
    report_freq: int = 3000
    record_freq: int = 100
    pruning_freq: int = 2000
    # live_freq: int = 20000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
    lr: float = 1e-3
    gradient_clipping: bool = False
    lr_schedule: str = "None"
    final_lr: float = 1e-6
    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    lr_decay_scaling_factor: float = 0.1  # scaling factor for lr decay
    train_batch_size: int = 128
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    activation: str = "relu"  # Activation function used throughout the model
    dataset: str = "mnist"
    normalize_inputs: bool = False
    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
    kept_classes_seed: int = 42  # Control variability when randomly selecting classes
    noisy_label: float = 0  # ratio (between [0,1]) of labels to randomly (uniformly) flip
    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
    architecture: str = "mlp_3"
    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
    with_bn: bool = False  # Add batchnorm layers or not in the models
    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
    sizes: Any = (50, 100, 250, 500, 750, 1000, 1250, 1500, 2000)
    regularizer: Optional[str] = "None"
    reg_param: float = 1e-4
    wd_param: Optional[float] = None
    reg_param_decay_cycles: int = 1  # number of cycles -1 inside a switching_period that reg_param is divided by 10
    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
    var_check: bool = False  # Using the variance definition for detecting dead neurons
    record_var_dist: bool = False  # Record variance distribution at pruning_freq is set to True
    init_seed: int = 41
    dynamic_pruning: bool = False
    add_noise: bool = False  # Add Gaussian noise to the gradient signal
    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones
    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
    noise_eta: float = 0.01
    noise_gamma: float = 0.0
    noise_seed: int = 1
    dropout_rate: float = 0
    with_rng_seed: int = 428
    linear_switch: bool = False  # Whether to switch mid-training steps to linear activations
    measure_linear_perf: bool = False  # Measure performance over the linear network without changing activation
    record_distribution_data: bool = False
    save_wanda: bool = False  # Whether to save weights and activations value or not
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

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

    if "imagenet" in exp_config.dataset:
        dataset_dir = exp_config.dataset
        exp_config.dataset = "imagenet"

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(
        optimizer_choice.keys())
    assert exp_config.dataset in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(
        architecture_choice.keys())
    assert exp_config.activation in activation_choice.keys(), "Current activation function available: " + str(
        activation_choice.keys())
    assert exp_config.lr_schedule in lr_scheduler_choice.keys(), "Current lr scheduler function available: " + str(
        lr_scheduler_choice.keys())
    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if exp_config.wd_param == 'None':
        exp_config.wd_param = None
    assert (not (("adamw" in exp_config.optimizer) and bool(
        exp_config.regularizer))) or bool(exp_config.wd_param), "Set wd_param if adamw is used with a regularization loss"
    if type(exp_config.sizes) == str:
        exp_config.sizes = literal_eval(exp_config.sizes)
    if type(exp_config.noise_imp) == str:
        exp_config.noise_imp = literal_eval(exp_config.noise_imp)
    if type(exp_config.epsilon_close) == str:
        exp_config.epsilon_close = literal_eval(exp_config.epsilon_close)

    if exp_config.dynamic_pruning:
        exp_name_ = exp_name+"_with_dynamic_pruning"
    else:
        exp_name_ = exp_name

    activation_fn = activation_choice[exp_config.activation]

    # Path for logs
    log_path = "./asymptotic_exps"
    if exp_config.dataset == "imagenet":
        log_path = "./asymptotic_imagenet_exps"
    # Logger config
    exp_run = Run(repo=log_path, experiment=exp_name_)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    if exp_config.save_wanda:
        # Create pickle directory
        pickle_dir_path = log_path + "/metadata/" + exp_name_ + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
        os.makedirs(pickle_dir_path)
        # Dump config file in it as well
        with open(pickle_dir_path+'config.json', 'w') as fp:
            json.dump(OmegaConf.to_container(exp_config), fp, indent=4)

    # Experiments with dropout
    with_dropout = exp_config.dropout_rate > 0
    if with_dropout:
        dropout_key = jax.random.PRNGKey(exp_config.with_rng_seed)
        assert exp_config.architecture in pick_architecture(
            with_dropout=True).keys(), "Current architectures available with dropout: " + str(
            pick_architecture(with_dropout=True).keys())
        net_config = {"dropout_rate": exp_config.dropout_rate}
    else:
        net_config = {}

    if not exp_config.with_bias:
        net_config['with_bias'] = exp_config.with_bias

    if exp_config.with_bn:
        assert exp_config.architecture in pick_architecture(
            with_bn=True).keys(), "Current architectures available with batchnorm: " + str(
            pick_architecture(with_bn=True).keys())
        net_config['bn_config'] = bn_config_choice[exp_config.bn_config]

    # Load the different dataset
    if exp_config.kept_classes:
        assert exp_config.kept_classes <= dataset_target_cardinality[
            exp_config.dataset], "subset must be smaller or equal to total number of classes in ds"
        kpt_key = jax.random.PRNGKey(exp_config.kept_classes_seed)
        kept_indices = jax.random.choice(kpt_key, dataset_target_cardinality[exp_config.dataset], (exp_config.kept_classes,),
                                         replace=False)
        kept_indices = jax.device_get(kept_indices)  # Back in numpy
    else:
        kept_indices = None
    load_data = dataset_choice[exp_config.dataset]
    eval_size = exp_config.eval_batch_size
    death_minibatch_size = exp_config.death_batch_size
    train_ds_size, train, train_eval, test_death = load_data(split="train", is_training=True,
                                                             batch_size=exp_config.train_batch_size,
                                                             other_bs=[eval_size, death_minibatch_size],
                                                             subset=kept_indices,
                                                             cardinality=True,
                                                             noisy_label=exp_config.noisy_label,
                                                             permuted_img_ratio=exp_config.permuted_img_ratio,
                                                             gaussian_img_ratio=exp_config.gaussian_img_ratio,
                                                             augment_dataset=exp_config.augment_dataset,
                                                             normalize=exp_config.normalize_inputs)
    # train_eval = train
    # test_death = train
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size, subset=kept_indices,
                                     cardinality=True, augment_dataset=exp_config.augment_dataset,
                                     normalize=exp_config.normalize_inputs)
    steps_per_epoch = train_ds_size // exp_config.train_batch_size
    if exp_config.dataset == 'imagenet':
        partial_train_ds_size = train_ds_size / 1000  # .1% of dataset used for evaluation on train
        test_death = train_eval  # Don't want to prefetch too many ds
    else:
        partial_train_ds_size = train_ds_size

    # Recording over all widths
    # live_neurons = []
    # avg_live_neurons = []
    # std_live_neurons = []
    # size_arr = []
    # f_acc = []

    if exp_config.save_wanda:
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

    decay_cycles = exp_config.reg_param_decay_cycles + int(exp_config.zero_end_reg_param)
    if decay_cycles == 2:
        reg_param_decay_period = int(0.8 * exp_config.training_steps)
    else:
        reg_param_decay_period = exp_config.training_steps // decay_cycles

    for size in exp_config.sizes:  # Vary the NN width
        # Time the subrun for the different sizes
        subrun_start_time = time.time()

        # Reset decay param
        decaying_reg_param = exp_config.reg_param

        # Make the network and optimiser
        architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
        if not exp_config.kept_classes:
            classes = dataset_target_cardinality[exp_config.dataset]   # Retrieving the number of classes in dataset
        else:
            classes = exp_config.kept_classes
        architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
        net, raw_net = build_models(*architecture, with_dropout=with_dropout)

        if exp_config.measure_linear_perf:
            lin_act_fn = activation_choice["linear"]
            lin_architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
            lin_architecture = lin_architecture(size, classes, activation_fn=lin_act_fn, **net_config)
            lin_net, raw_net = build_models(*lin_architecture, with_dropout=with_dropout)

        optimizer = optimizer_choice[exp_config.optimizer]
        opt_chain = []
        if "w" in exp_config.optimizer:  # Pass reg_param to wd argument of adamw # TODO: dangerous condition
            if exp_config.wd_param:  # wd_param overwrite reg_param when specified
                optimizer = Partial(optimizer, weight_decay=exp_config.wd_param)
            else:
                optimizer = Partial(optimizer, weight_decay=exp_config.reg_param)
        elif exp_config.wd_param:  # TODO: Maybe exclude adamw?
            opt_chain.append(optax.add_decayed_weights(weight_decay=exp_config.wd_param))
        if exp_config.optimizer == "adam_to_momentum":  # Setting transition steps to total # of steps
            optimizer = Partial(optimizer, transition_steps=exp_config.training_steps)
        if exp_config.gradient_clipping:
            opt_chain.append(optax.clip(10))

        if 'noisy' in exp_config.optimizer:
            opt_chain.append(optimizer(exp_config.lr, eta=exp_config.noise_eta,
                                       gamma=exp_config.noise_gamma))
        else:
            if isinstance(exp_config.lr_decay_steps, omegaconf.listconfig.ListConfig):  # TODO: This is dirty...
                decay_boundaries = [steps_per_epoch * lr_decay_step for lr_decay_step in exp_config.lr_decay_steps]
            else:
                decay_boundaries = [steps_per_epoch * exp_config.lr_decay_steps * (i + 1) for i in
                                    range((exp_config.training_steps // steps_per_epoch) // exp_config.lr_decay_steps)]
            lr_schedule = lr_scheduler_choice[exp_config.lr_schedule](exp_config.training_steps, exp_config.lr,
                                                                      exp_config.final_lr,
                                                                      decay_boundaries,
                                                                      exp_config.lr_decay_scaling_factor)
            opt_chain.append(optimizer(lr_schedule))
        opt = optax.chain(*opt_chain)

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                       classes=classes, with_dropout=with_dropout)
        test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                            classes=classes, is_training=False, with_dropout=with_dropout)
        accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
        death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                                                     avg=exp_config.avg_for_eps)
        var_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                                                         avg=exp_config.avg_for_eps, var=exp_config.var_check)
        scan_len = int(partial_train_ds_size // death_minibatch_size)
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        var_scan_death_check_fn = utl.scanned_death_check_fn(var_death_check_fn, scan_len)
        scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
            utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout,
                                        avg=exp_config.avg_for_eps), scan_len, True)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

        if exp_config.measure_linear_perf:
            lin_accuracy_fn = utl.accuracy_given_model(lin_net, with_dropout=with_dropout)
            lin_full_accuracy_fn = utl.create_full_accuracy_fn(lin_accuracy_fn, test_size // eval_size)

        params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
        # initial_params = copy.deepcopy(params)  # Keep a copy of the initial params for relative change metric
        opt_state = opt.init(params)
        activation_layer_order = list(state.keys())
        neuron_states = utl.NeuronStates(activation_layer_order)
        acti_map = utl.get_activation_mapping(raw_net, next(train))
        del raw_net
        update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                        exp_config.noise_live_only, with_dropout=with_dropout,
                                                        acti_map=acti_map)

        noise_key = jax.random.PRNGKey(exp_config.noise_seed)

        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, size)
        init_total_neurons = copy.copy(total_neurons)
        init_total_per_layer = copy.copy(total_per_layer)

        for step in range(exp_config.training_steps):
            if (decay_cycles > 1) and (step % reg_param_decay_period == 0) and \
                    (not (step % (exp_config.training_steps - 1) == 0)):  # and (not exp_config.reg_param_schedule):
                decaying_reg_param = decaying_reg_param / 10
                if (step >= ((decay_cycles - 1) * reg_param_decay_period)) and exp_config.zero_end_reg_param:
                    decaying_reg_param = 0
                print("decaying reg param:")
                print(decaying_reg_param)
                print()
                loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=decaying_reg_param,
                                               classes=classes, with_dropout=with_dropout)
                test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                    reg_param=decaying_reg_param,
                                                    classes=classes, is_training=False, with_dropout=with_dropout)
                update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                                exp_config.noise_live_only, with_dropout=with_dropout,
                                                                acti_map=acti_map)

            if step % exp_config.record_freq == 0:
                train_loss = test_loss_fn(params, state, next(train_eval))
                train_accuracy = accuracy_fn(params, state, next(train_eval))
                test_accuracy = accuracy_fn(params, state, next(test_eval))
                test_loss = test_loss_fn(params, state, next(test_eval))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                # Periodically print classification accuracy on train & test sets.
                if step % exp_config.report_freq == 0:
                    print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                          f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
                test_death_batch = next(test_death)
                dead_neurons = death_check_fn(params, state, test_death_batch)
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step,
                              context={"net size": utl.size_to_string(size)})
                if exp_config.epsilon_close:
                    for eps in exp_config.epsilon_close:
                        eps_dead_neurons = death_check_fn(params, state, test_death_batch, eps)
                        eps_dead_neurons_count, _ = utl.count_dead_neurons(eps_dead_neurons)
                        exp_run.track(jax.device_get(eps_dead_neurons_count),
                                      name="Quasi-dead neurons", step=step,
                                      context={"net size": utl.size_to_string(size), "epsilon": eps})
                        exp_run.track(jax.device_get(total_neurons - eps_dead_neurons_count),
                                      name="Quasi-live neurons", step=step,
                                      context={"net size": utl.size_to_string(size), "epsilon": eps})
                if exp_config.var_check:
                    var_dead_neurons = var_death_check_fn(params, state, test_death_batch)
                    var_dead_neurons_count, _ = utl.count_dead_neurons(var_dead_neurons)
                    exp_run.track(jax.device_get(var_dead_neurons_count),
                                  name="Quasi-dead neurons", step=step,
                                  context={"net size": utl.size_to_string(size), "variance check": True})
                    exp_run.track(jax.device_get(total_neurons - var_dead_neurons_count),
                                  name="Quasi-live neurons", step=step,
                                  context={"net size": utl.size_to_string(size), "variance check": True})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(jax.device_get(test_loss), name="Test loss", step=step,
                              context={"net size": utl.size_to_string(size)})

            if step % exp_config.pruning_freq == 0:
                if exp_config.record_var_dist:
                    dist_data, dead_neurons = scan_death_check_fn_with_activations_data(params, state,
                                                                                        test_death)
                else:
                    dead_neurons = scan_death_check_fn(params, state, test_death)
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                              step=step,
                              context={"net size": utl.size_to_string(size)})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                              name="Live neurons; whole training dataset",
                              step=step,
                              context={"net size": utl.size_to_string(size)})
                if exp_config.epsilon_close:
                    for eps in exp_config.epsilon_close:
                        eps_dead_neurons = scan_death_check_fn(params, state, test_death, eps)
                        eps_dead_neurons_count, eps_dead_per_layers = utl.count_dead_neurons(eps_dead_neurons)
                        exp_run.track(jax.device_get(eps_dead_neurons_count),
                                      name="Quasi-dead neurons; whole training dataset",
                                      step=step,
                                      context={"net size": utl.size_to_string(size), "epsilon": eps})
                        exp_run.track(jax.device_get(total_neurons - eps_dead_neurons_count),
                                      name="Quasi-live neurons; whole training dataset",
                                      step=step,
                                      context={"net size": utl.size_to_string(size), "epsilon": eps})
                if exp_config.var_check:
                    var_dead_neurons = var_scan_death_check_fn(params, state, test_death)
                    var_dead_neurons_count, eps_dead_per_layers = utl.count_dead_neurons(var_dead_neurons)
                    exp_run.track(jax.device_get(var_dead_neurons_count),
                                  name="Quasi-dead neurons; whole training dataset",
                                  step=step,
                                  context={"net size": utl.size_to_string(size), "variance check": True})
                    exp_run.track(jax.device_get(total_neurons - var_dead_neurons_count),
                                  name="Quasi-live neurons; whole training dataset",
                                  step=step,
                                  context={"net size": utl.size_to_string(size), "variance check": True})
                    if exp_config.record_var_dist:
                        _, _, _, var_dist = dist_data
                        var_dist, _ = ravel_pytree(var_dist)
                        var_dist = jax.device_get(var_dist)
                        aim_var_dist = Distribution(var_dist, bin_count=100)
                        exp_run.track(aim_var_dist, name='Activation variance',
                                      step=step,
                                      context={"net size": utl.size_to_string(size)})
                        (_, batch_act), _ = net.apply(params, state, x=next(test_death), return_activations=True, is_training=False)
                        batch_var = jax.tree_map(Partial(jnp.var, axis=0), batch_act)
                        for layer, layer_batch_var in enumerate(batch_var):
                            layer_batch_var = jax.device_get(layer_batch_var)
                            batch_var_dist = Distribution(layer_batch_var, bin_count=100)
                            exp_run.track(batch_var_dist, name='Batch activation variance',
                                          step=step,
                                          context={"net size": utl.size_to_string(size), "layer": str(layer)})
                            exp_run.track(jax.device_get(jnp.mean(layer_batch_var)),
                                          name="Batch variance mean",
                                          step=step,
                                          context={"net size": utl.size_to_string(size), "layer": str(layer)})

                for i, layer_dead in enumerate(dead_per_layers):
                    total_neuron_in_layer = total_per_layer[i]
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                                  context={"net size": utl.size_to_string(size)})
                    exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                                  name=f"Live neurons in layer {i}; whole training dataset", step=step,
                                  context={"net size": utl.size_to_string(size)})
                del dead_per_layers

                if exp_config.measure_linear_perf:
                    # Record performance over full validation set of the NN for relu and linear activations
                    relu_perf = jax.device_get(final_accuracy_fn(params, state, test_eval))
                    exp_run.track(relu_perf,
                                  name="Total accuracy for relu NN", step=step,
                                  context={"net size": utl.size_to_string(size)})
                    lin_perf = jax.device_get(lin_full_accuracy_fn(params, state, test_eval))
                    exp_run.track(lin_perf,
                                  name="Total accuracy for linear NN", step=step,
                                  context={"net size": utl.size_to_string(size)})

                if exp_config.dynamic_pruning:
                    # Pruning the network
                    neuron_states.update_from_ordered_list(dead_neurons)
                    params, opt_state, state, new_sizes = utl.prune_params_state_optstate(params, acti_map,
                                                                                          neuron_states, opt_state,
                                                                                          state)

                    architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                        exp_config.architecture]
                    architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
                    net = build_models(*architecture)[0]
                    total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes)

                    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                   reg_param=decaying_reg_param, classes=classes,
                                                   with_dropout=with_dropout)
                    test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                           reg_param=decaying_reg_param,
                                                           classes=classes,
                                                           is_training=False, with_dropout=with_dropout)
                    accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
                    update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise,
                                                                    exp_config.noise_imp, exp_config.noise_live_only,
                                                                    with_dropout=with_dropout,
                                                                    acti_map=acti_map)
                    death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
                    # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                    #                                                  epsilon=exp_config.epsilon_close,
                    #                                                  avg=exp_config.avg_for_eps)
                    # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
                    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
                    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
                    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn,
                                                                    int(partial_train_ds_size // eval_size))

                del dead_neurons  # Freeing memory
                train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, state, train_eval))
                exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                              step=step,
                              context={"net size": utl.size_to_string(size)})

            # if ((step+1) % exp_config.live_freq == 0) and (step+2 < exp_config.training_steps):
            #     current_dead_neurons = scan_death_check_fn(params, state, test_death)
            #     current_dead_neurons_count, _ = utl.count_dead_neurons(current_dead_neurons)
            #     del current_dead_neurons
            #     del _
            #     exp_run.track(jax.device_get(total_neurons - current_dead_neurons_count),
            #                   name=f"Live neurons at training step {step+1}", step=starting_neurons)

            if (((step+1) % (exp_config.training_steps//2)) == 0) and exp_config.linear_switch:
                activation_fn = activation_choice["linear"]
                architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
                architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
                net, raw_net = build_models(*architecture, with_dropout=with_dropout)

                # Reset training/monitoring functions
                loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                               classes=classes, with_dropout=with_dropout)
                test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                    reg_param=exp_config.reg_param,
                                                    classes=classes, is_training=False, with_dropout=with_dropout)
                accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
                update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                                exp_config.noise_live_only, with_dropout=with_dropout,
                                                                acti_map=acti_map)
                death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                                                             avg=exp_config.avg_for_eps)
                var_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                                                                 avg=exp_config.avg_for_eps, var=exp_config.var_check)
                scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
                var_scan_death_check_fn = utl.scanned_death_check_fn(var_death_check_fn, scan_len)
                scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
                    utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout,
                                                avg=exp_config.avg_for_eps), scan_len, True)
                final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
                full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

            # Train step over single batch
            if with_dropout:
                params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train), dropout_key)
            else:
                if not exp_config.add_noise:
                    params, state, opt_state = update_fn(params, state, opt_state, next(train))
                else:
                    noise_var = exp_config.noise_eta / ((1 + step) ** exp_config.noise_gamma)
                    noise_var = exp_config.lr * noise_var  # Apply lr for consistency with update size
                    params, state, opt_state, noise_key = update_fn(params, state, opt_state, next(train), noise_var,
                                                                    noise_key)

        # # final_accuracy = jax.device_get(accuracy_fn(params, next(final_test_eval)))
        # final_accuracy = jax.device_get(final_accuracy_fn(params, state, test_eval))
        # final_train_acc = jax.device_get(full_train_acc_fn(params, state, train_eval))
        # size_arr.append(starting_neurons)

        if exp_config.record_distribution_data:
            scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
                utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
            activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state,
                                                                                             test_death)
        else:
            final_dead_neurons = scan_death_check_fn(params, state, test_death)

        neuron_states.update_from_ordered_list(final_dead_neurons)
        params, opt_state, state, new_sizes = utl.prune_params_state_optstate(params, acti_map,
                                                                              neuron_states, opt_state,
                                                                              state)  # Final pruning before eval
        final_params_count = utl.count_params(params)

        # final_dead_neurons_count, final_dead_per_layer = utl.count_dead_neurons(final_dead_neurons)
        del final_dead_neurons  # Freeing memory
        architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
            exp_config.architecture]
        architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
        net = build_models(*architecture)[0]
        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes)

        accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
        death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
        # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
        #                                                  epsilon=exp_config.epsilon_close,
        #                                                  avg=exp_config.avg_for_eps)
        # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

        final_accuracy = final_accuracy_fn(params, state, test_eval)
        final_train_acc = full_train_acc_fn(params, state, train_eval)

        # Additionally, track an 'on average' number of death neurons within a batch
        # def scan_f(_, __):
        #     _, batch_dead_neurons = utl.death_check_given_model(net, with_activations=True)(params, next(test_death))
        #     return None, total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0]
        # _, batches_final_live_neurons = jax.lax.scan(scan_f, None, None, scan_len)
        batch_dead_neurons = death_check_fn(params, state, next(test_death))
        batches_final_live_neurons = [total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0]]
        for i in range(scan_len - 1):
            batch_dead_neurons = death_check_fn(params, state, next(test_death))
            batches_final_live_neurons.append(total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0])
        batches_final_live_neurons = jnp.stack(batches_final_live_neurons)

        avg_final_live_neurons = jnp.mean(batches_final_live_neurons, axis=0)
        # std_final_live_neurons = jnp.std(batches_final_live_neurons, axis=0)

        exp_run.track(jax.device_get(avg_final_live_neurons),
                      name="On average, live neurons after convergence w/r total neurons", step=init_total_neurons)
        exp_run.track(jax.device_get(avg_final_live_neurons / init_total_neurons),
                      name="Average live neurons ratio after convergence w/r total neurons", step=init_total_neurons)
        total_live_neurons = total_neurons
        exp_run.track(jax.device_get(total_live_neurons),
                      name="Live neurons after convergence w/r total neurons", step=init_total_neurons)
        exp_run.track(jax.device_get(total_live_neurons/init_total_neurons),
                      name="Live neurons ratio after convergence w/r total neurons", step=init_total_neurons)
        if exp_config.epsilon_close:
            for eps in exp_config.epsilon_close:
                eps_final_dead_neurons = scan_death_check_fn(params, state, test_death, eps)
                eps_final_dead_neurons_count, _ = utl.count_dead_neurons(eps_final_dead_neurons)
                del eps_final_dead_neurons
                eps_batch_dead_neurons = death_check_fn(params, state, next(test_death), eps)
                eps_batches_final_live_neurons = [total_neurons - utl.count_dead_neurons(eps_batch_dead_neurons)[0]]
                for i in range(scan_len - 1):
                    eps_batch_dead_neurons = death_check_fn(params, state, next(test_death), eps)
                    eps_batches_final_live_neurons.append(total_neurons - utl.count_dead_neurons(eps_batch_dead_neurons)[0])
                eps_batches_final_live_neurons = jnp.stack(eps_batches_final_live_neurons)
                eps_avg_final_live_neurons = jnp.mean(eps_batches_final_live_neurons, axis=0)

                exp_run.track(jax.device_get(eps_avg_final_live_neurons),
                              name="On average, quasi-live neurons after convergence w/r total neurons",
                              step=init_total_neurons, context={"epsilon": eps})
                exp_run.track(jax.device_get(eps_avg_final_live_neurons / init_total_neurons),
                              name="Average quasi-live neurons ratio after convergence w/r total neurons",
                              step=init_total_neurons, context={"epsilon": eps})
                eps_total_live_neurons = total_neurons - eps_final_dead_neurons_count
                exp_run.track(jax.device_get(eps_total_live_neurons),
                              name="Quasi-live neurons after convergence w/r total neurons", step=init_total_neurons,
                              context={"epsilon": eps})
                exp_run.track(jax.device_get(eps_total_live_neurons / init_total_neurons),
                              name="Quasi-live neurons ratio after convergence w/r total neurons",
                              step=init_total_neurons, context={"epsilon": eps})

        for i, live_in_layer in enumerate(total_per_layer):
            total_neuron_in_layer = init_total_per_layer[i]
            # live_in_layer = total_neuron_in_layer - layer_dead
            exp_run.track(jax.device_get(live_in_layer),
                          name=f"Live neurons in layer {i} after convergence w/r total neurons",
                          step=init_total_neurons)
            exp_run.track(jax.device_get(live_in_layer/total_neuron_in_layer),
                          name=f"Live neurons ratio in layer {i} after convergence w/r total neurons",
                          step=init_total_neurons)
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r total neurons", step=init_total_neurons)
        exp_run.track(final_train_acc,
                      name="Train accuracy after convergence w/r total neurons", step=init_total_neurons)
        # if not exp_config.dynamic_pruning:  # Cannot take norm between initial and pruned params
        #     params_vec, _ = ravel_pytree(params)
        #     initial_params_vec, _ = ravel_pytree(initial_params)
        #     exp_run.track(
        #         jax.device_get(jnp.linalg.norm(params_vec - initial_params_vec) / jnp.linalg.norm(initial_params_vec)),
        #         name="Relative change in norm of weights from init after convergence w/r total neurons",
        #         step=starting_neurons)
        if exp_config.record_distribution_data:
            activations_max, activations_mean, activations_count, _ = activations_data
            if exp_config.save_wanda:
                activations_meta.maximum[size] = activations_max
                activations_meta.mean[size] = activations_mean
                activations_meta.count[size] = activations_count
            activations_max, _ = ravel_pytree(activations_max)
            # activations_max = jax.device_get(activations_max)
            activations_mean, _ = ravel_pytree(activations_mean)
            # activations_mean = jax.device_get(activations_mean)
            activations_count, _ = ravel_pytree(activations_count)
            # activations_count = jax.device_get(activations_count)
            activations_max_dist = Distribution(activations_max, bin_count=100)
            exp_run.track(activations_max_dist, name='Maximum activation distribution after convergence', step=0,
                          context={"size": utl.size_to_string(size)})
            activations_mean_dist = Distribution(activations_mean, bin_count=100)
            exp_run.track(activations_mean_dist, name='Mean activation distribution after convergence', step=0,
                          context={"size": utl.size_to_string(size)})
            activations_count_dist = Distribution(activations_count, bin_count=50)
            exp_run.track(activations_count_dist, name='Activation count per neuron after convergence', step=0,
                          context={"size": utl.size_to_string(size)})

        # live_neurons.append(total_neurons - final_dead_neurons_count)
        # avg_live_neurons.append(avg_final_live_neurons)
        # std_live_neurons.append(std_final_live_neurons)
        # f_acc.append(final_accuracy)

        # Making sure compiled fn cache was cleared
        # loss.clear_cache()
        # test_loss_fn.clear_cache()
        # accuracy_fn.clear_cache()
        # update_fn.clear_cache()
        # death_check_fn.clear_cache()
        # eps_death_check_fn.clear_cache()
        # scan_death_check_fn._clear_cache()  # No more cache
        # scan_death_check_fn_with_activations._clear_cache()  # No more cache
        # final_accuracy_fn._clear_cache()  # No more cache

        if exp_config.save_wanda:
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

    # # Plots
    # dir_path = "./logs/plots/" + exp_name_ + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
    # os.makedirs(dir_path)
    #
    # fig1 = plt.figure(figsize=(15, 10))
    # plt.plot(size_arr, live_neurons, label="Live neurons", linewidth=4)
    # plt.xlabel("Number of neurons in NN", fontsize=16)
    # plt.ylabel("Live neurons at end of training", fontsize=16)
    # plt.title("Effective capacity, "+exp_config.architecture+" on "+exp_config.dataset, fontweight='bold', fontsize=20)
    # plt.legend(prop={'size': 16})
    # fig1.savefig(dir_path+"effective_capacity.png")
    # # aim_fig1 = Figure(fig1)
    # # aim_img1 = Image(fig1)
    # # exp_run.track(aim_fig1, name="Effective capacity", step=0)
    # # exp_run.track(aim_img1, name="Effective capacity; img", step=0)
    #
    # fig1_5 = plt.figure(figsize=(15, 10))
    # plt.errorbar(size_arr, avg_live_neurons, std_live_neurons, label="Live neurons", linewidth=4)
    # plt.xlabel("Number of neurons in NN", fontsize=16)
    # plt.ylabel("Average live neurons at end of training", fontsize=16)
    # plt.title((f"Effective capacity averaged on minibatch of size={death_minibatch_size}, "+exp_config.architecture+" on "
    #            + exp_config.dataset), fontweight='bold', fontsize=20)
    # plt.legend(prop={'size': 16})
    # fig1_5.savefig(dir_path+"avg_effective_capacity.png")
    # # aim_img1_5 = Image(fig1_5)
    # # exp_run.track(aim_img1_5, name="Average effective capacity; img", step=0)
    #
    # fig2 = plt.figure(figsize=(15, 10))
    # plt.plot(size_arr, jnp.array(live_neurons) / jnp.array(size_arr), label="alive ratio")
    # plt.xlabel("Number of neurons in NN", fontsize=16)
    # plt.ylabel("Ratio of live neurons at end of training", fontsize=16)
    # plt.title(exp_config.architecture+" effective capacity on "+exp_config.dataset, fontweight='bold', fontsize=20)
    # plt.legend(prop={'size': 16})
    # fig2.savefig(dir_path+"live_neurons_ratio.png")
    # # aim_fig2 = Figure(fig2)
    # # aim_img2 = Image(fig2)
    # # # exp_run.track(aim_fig2, name="Live neurons ratio", step=0)
    # # exp_run.track(aim_img2, name="Live neurons ratio; img", step=0)
    #
    # fig3 = plt.figure(figsize=(15, 10))
    # plt.plot(size_arr, f_acc, label="accuracy", linewidth=4)
    # plt.xlabel("Number of neurons in NN", fontsize=16)
    # plt.ylabel("Final accuracy", fontsize=16)
    # plt.title("Performance at convergence, "+exp_config.architecture+" on "+exp_config.dataset, fontweight='bold', fontsize=20)
    # plt.legend(prop={'size': 16})
    # fig3.savefig(dir_path+"performance_at_convergence.png")
    # # aim_fig3 = Figure(fig3)
    # # aim_img3 = Image(fig3)
    # # # exp_run.track(aim_fig3, name="Performance at convergence", step=0)
    # # exp_run.track(aim_img3, name="Performance at convergence; img", step=0)

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
