""" Experiment to run to compare pruning performance to baseline (pruning perform in controlling_overfitting.py)"""

import copy
import optax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")
from aim import Run, Distribution
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
from pathlib import Path
import omegaconf.listconfig
import functools
import jaxpruner

from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

import utils.utils as utl
from utils.utils import build_models
from utils.config import activation_choice, optimizer_choice, dataset_choice, dataset_target_cardinality
from utils.config import regularizer_choice, architecture_choice, lr_scheduler_choice, bn_config_choice
from utils.config import pick_architecture, baseline_pruning_method_choice, reg_param_scheduler_choice


# Experience name -> for aim logger
exp_name = "pruning_baseline"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 250001
    report_freq: int = 2500
    record_freq: int = 250
    pruning_freq: int = 1000
    # live_freq: int = 25000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
    lr: float = 1e-3
    gradient_clipping: bool = False
    lr_schedule: str = "constant"
    final_lr: float = 1e-6
    lr_decay_steps: Any = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
    lr_decay_scaling_factor: float = 0.1  # scaling factor for lr decay
    train_batch_size: int = 512
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adamw"
    activation: str = "relu"  # Activation function used throughout the model
    dataset: str = "mnist"
    normalize_inputs: bool = False  # Substract mean across channels from inputs and divide by variance
    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
    label_smoothing: float = 0.0  # Level of smoothing applied during the loss calculation, 0.0 -> no smoothing
    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
    noisy_label: float = 0.0  # ratio (between [0,1]) of labels to randomly (uniformly) flip
    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
    architecture: str = "mlp_3"
    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
    with_bn: bool = False  # Add batchnorm layers or not in the models
    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
    size: Any = 50  # Can also be a tuple for convnets
    regularizer: Optional[str] = "None"
    reg_param: float = 5e-4
    masked_reg: Optional[str] = None  # If "all" exclude all bias and bn params, if "scale" only exclude scale param/ also offset_only, scale_only and bn_params_only
    wd_param: Optional[float] = None
    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
    reg_param_schedule: Optional[str] = None  # Schedule for reg_param, priority over reg_param_decay_cycles flag
    reg_param_span: Optional[int] = None  # More general than zero_end_reg_param, allow to decide when reg_param schedule falls to 0
    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
    init_seed: int = 41
    dynamic_pruning: bool = False
    prune_after: int = 0  # Option: only start pruning after <prune_after> step has been reached
    spar_levels: Any = (0.5, 0.8)
    sparsity_distribution: str = "uniform"  # uniform or erk : available distribution in jaxpruner
    pruning_method: str = "WMP"  # See config.py for option
    update_start_step: float = 0.32  # When to start pruning during training
    update_end_step: float = 0.8  # When to end pruning during training
    drop_fraction: float = 0.1  # Jaxpruner parameter
    add_noise: bool = False  # Add Gaussian noise to the gradient signal
    asymmetric_noise: bool = True  # Use an asymmetric noise addition, not applied to all neurons' weights
    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones. reverse if False
    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
    noise_eta: float = 0.01  # Variance of added noise; can only be used with a reg_param_schedule that it will match
    noise_seed: int = 1
    dropout_rate: float = 0
    with_rng_seed: int = 428
    preempt_handling: bool = False  # Frequent checkpointing to handle SLURM preemption
    jobid: Optional[str] = None  # Manually restart previous job from checkpoint
    checkpoint_freq: int = 5  # in epochs
    save_wanda: bool = False  # Whether to save weights and activations value or not
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:
    run_start_time = time.time()

    if "imagenet" in exp_config.dataset:
        dataset_dir = exp_config.dataset
        if "vit" in exp_config.architecture:
            exp_config.dataset = "imagenet_vit"
        else:
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
    assert exp_config.sparsity_distribution in (
    "uniform", "erk"), "Implemented sparsity distribution in jaxpruner are uniform or erk rn"
    assert exp_config.pruning_method in baseline_pruning_method_choice.keys(), "Supporting only the following baseline pruner" + str(
        baseline_pruning_method_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if exp_config.wd_param == 'None':
        exp_config.wd_param = None
    assert (not (("adamw" in exp_config.optimizer) and bool(
        exp_config.regularizer))) or bool(
        exp_config.wd_param), "Set wd_param if adamw is used with a regularization loss"
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.epsilon_close) == str:
        exp_config.epsilon_close = literal_eval(exp_config.epsilon_close)
    if type(exp_config.spar_levels) == str:
        exp_config.spar_levels = literal_eval(exp_config.spar_levels)
    if type(exp_config.noise_imp) == str:
        exp_config.noise_imp = literal_eval(exp_config.noise_imp)
    if type(exp_config.lr_decay_steps) == str:
        exp_config.lr_decay_steps = literal_eval(exp_config.lr_decay_steps)

    if exp_config.dynamic_pruning:
        exp_name_ = exp_name+"_with_dynamic_pruning"
    else:
        exp_name_ = exp_name

    activation_fn = activation_choice[exp_config.activation]

    # Check for checkpoints
    load_from_preexisting_model_state = False
    if exp_config.preempt_handling:
        SCRATCH = Path(os.environ["SCRATCH"])
        if exp_config.jobid:
            SLURM_JOBID = exp_config.jobid
        else:
            SLURM_JOBID = os.environ["SLURM_JOBID"]
            exp_config.jobid = SLURM_JOBID
        saving_dir = SCRATCH / exp_name_ / SLURM_JOBID

        # Create the directory if it does not exist
        os.makedirs(saving_dir, exist_ok=True)

        # Check for previous checkpoints
        run_state = utl.load_run_state(saving_dir)
        if run_state:
            load_from_preexisting_model_state = True
        else:  # Initialize the run_state
            run_state = utl.JaxPrunerRunState(epoch=0, training_step=0, model_dir=saving_dir,
                                              aim_hash=None, slurm_jobid=SLURM_JOBID, exp_name=exp_name_,
                                              curr_pruning_density=exp_config.spar_levels[0],
                                              dropout_key=jax.random.PRNGKey(exp_config.with_rng_seed),
                                              decaying_reg_param=exp_config.reg_param)
            # with open(os.path.join(saving_dir, "checkpoint_run_state.pkl"), "wb") as f:  # Save only if one additional epoch completed
            #     pickle.dump(run_state, f)

            # Dump config file in it as well
            with open(os.path.join(saving_dir, 'config.json'), 'w') as fp:
                json.dump(OmegaConf.to_container(exp_config), fp, indent=4)
        aim_hash = run_state["aim_hash"]
    else:
        aim_hash = None

    # Logger config
    log_path = "./NEURIPS2024_jaxpruner_baseline"
    exp_run = Run(repo=log_path, experiment=exp_name_, run_hash=aim_hash, force_resume=True)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)
    if exp_config.preempt_handling:
        run_state["aim_hash"] = exp_run.hash

    if exp_config.save_wanda:
        # Create pickle directory
        pickle_dir_path = log_path + "/metadata/" + exp_name_ + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
        os.makedirs(pickle_dir_path)
        # Dump config file in it as well
        with open(pickle_dir_path + 'config.json', 'w') as fp:
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
        dropout_key = None
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
        kept_indices = np.random.choice(dataset_target_cardinality[exp_config.dataset], exp_config.kept_classes,
                                        replace=False)
    else:
        kept_indices = None
    load_data = dataset_choice[exp_config.dataset]
    if 'imagenet' in exp_config.dataset:
        load_data = Partial(load_data, dataset_dir)
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
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size, subset=kept_indices,
                                     cardinality=True, augment_dataset=exp_config.augment_dataset,
                                     normalize=exp_config.normalize_inputs)
    steps_per_epoch = train_ds_size // exp_config.train_batch_size
    if 'imagenet' in exp_config.dataset:
        partial_train_ds_size = train_ds_size/1000  # .1% of dataset used for evaluation on train
        test_death = train_eval  # Don't want to prefetch too many ds
    else:
        partial_train_ds_size = train_ds_size / 25

    if exp_config.save_wanda:
        # Recording metadata about activations that will be pickled
        @dataclass
        class ActivationMeta:
            maximum: List[float] = field(default_factory=list)
            mean: List[float] = field(default_factory=list)
            count: List[int] = field(default_factory=list)
        activations_meta = ActivationMeta()

    if exp_config.preempt_handling:
        beginning_index = exp_config.spar_levels.index(run_state["curr_pruning_density"])
        spars_iterate = exp_config.spar_levels[beginning_index:]
    else:
        spars_iterate = exp_config.spar_levels
    for sparsity in spars_iterate:  # Vary the regularizer parameter to measure impact on overfitting

        size = exp_config.size
        # Make the network and optimiser
        architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[exp_config.architecture]
        if not exp_config.kept_classes:
            classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
        else:
            classes = exp_config.kept_classes
        architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
        net, _ = build_models(*architecture, with_dropout=with_dropout)

        optimizer = optimizer_choice[exp_config.optimizer]
        opt_chain = []
        if exp_config.gradient_clipping:
            opt_chain.append(optax.clip(10))
        if "loschi" in exp_config.optimizer:  # Using reg_param parameters to control wd with those optimizers
            if exp_config.reg_param_schedule:
                if exp_config.zero_end_reg_param:
                    sched_end = int(0.9 * exp_config.training_steps)
                else:
                    sched_end = exp_config.training_steps
                if exp_config.wd_param:
                    div_factor = exp_config.reg_param/exp_config.wd_param
                    final_div_factor = 1.0
                else:  # default values
                    div_factor = 25.0
                    final_div_factor = 1e4
                wd_schedule = reg_param_scheduler_choice[exp_config.reg_param_schedule](sched_end, exp_config.reg_param,
                                                                                        div_factor=div_factor,
                                                                                        final_div_factor=final_div_factor)
            else:
                wd_schedule = exp_config.reg_param if exp_config.reg_param > 0.0 else exp_config.wd_param
            optimizer = Partial(optimizer, weight_decay=wd_schedule)
        elif "w" in exp_config.optimizer:  # Pass reg_param to wd argument of adamw # TODO: dangerous condition
            if exp_config.wd_param:  # wd_param overwrite reg_param when specified
                optimizer = Partial(optimizer, weight_decay=exp_config.wd_param)
            else:
                optimizer = Partial(optimizer, weight_decay=exp_config.reg_param)
        elif exp_config.wd_param:  # TODO: Maybe exclude adamw?
            opt_chain.append(optax.add_decayed_weights(weight_decay=exp_config.wd_param))

        if 'noisy' in exp_config.optimizer:
            assert False, "Removed noisy optimizer option for this exp"
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
        accuracies_log = []

        # Configuring the (jax)pruner
        if exp_config.sparsity_distribution == "uniform":
            sparsity_distribution = functools.partial(
                jaxpruner.sparsity_distributions.uniform, sparsity=sparsity)
        elif exp_config.sparsity_distribution == "erk":
            sparsity_distribution = functools.partial(
                jaxpruner.sparsity_distributions.erk, sparsity=sparsity)

        def treemap_sparsity(pytree, _sparsity):
            return jax.tree_map(lambda x: _sparsity, pytree)

        def exclusion_fn(key):  # fn to exclude specific layer from pruner (for example, bn layers)
            to_exclude = ["norm", "bn", 'init']
            return any([_nme in key for _nme in to_exclude])

        def custom_distribution(_params):  # Don't prune normalization layer and init conv, according to litterature
            _sparsity_dict = sparsity_distribution(_params)
            return {key: treemap_sparsity(_sparsity_dict[key], None) if exclusion_fn(key) else _sparsity_dict[key] for
                    key in _sparsity_dict}

        _pruning_method = baseline_pruning_method_choice[exp_config.pruning_method]
        if exp_config.pruning_method in ('RigL', 'Set'):
            updater_config = {"drop_fraction_fn": optax.cosine_decay_schedule(
                exp_config.drop_fraction, int(exp_config.update_end_step * exp_config.training_steps)
            )}
        else:
            updater_config = {}
        pruner = _pruning_method(
            sparsity_distribution_fn=custom_distribution,
            scheduler=jaxpruner.sparsity_schedules.PolynomialSchedule(
                update_freq=1000, update_start_step=int(exp_config.update_start_step * exp_config.training_steps),
                update_end_step=int(exp_config.update_end_step * exp_config.training_steps)),
            **updater_config
        )

        # if exp_config.pruning_method == "WMP":
        #     pruner = jaxpruner.GlobalMagnitudePruning(
        #         sparsity_distribution_fn=custom_distribution,
        #         scheduler=jaxpruner.sparsity_schedules.PolynomialSchedule(
        #             update_freq=500, update_start_step=int(.30*exp_config.training_steps), update_end_step=int(.80*exp_config.training_steps))
        #     )
        # elif exp_config.pruning_method == "LMP":
        #     pruner = utl.LayerMagnitudePruning(
        #         sparsity_distribution_fn=custom_distribution,
        #         scheduler=jaxpruner.sparsity_schedules.PolynomialSchedule(
        #             update_freq=500, update_start_step=int(.30 * exp_config.training_steps),
        #             update_end_step=int(.80 * exp_config.training_steps))
        #     )
        opt = pruner.wrap_optax(opt)

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                       classes=classes, with_dropout=with_dropout,
                                       exclude_bias_bn_from_reg=exp_config.masked_reg,
                                       label_smoothing=exp_config.label_smoothing)
        test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param,
                                               classes=classes, is_training=False, with_dropout=with_dropout,
                                               exclude_bias_bn_from_reg=exp_config.masked_reg)
        accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                        exp_config.asymmetric_noise,
                                                        live_only=exp_config.noise_live_only,
                                                        with_dropout=with_dropout)
        noise_key = jax.random.PRNGKey(exp_config.noise_seed)
        death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
        # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
        #                                                  epsilon=exp_config.epsilon_close, avg=exp_config.avg_for_eps)
        scan_len = int(partial_train_ds_size // death_minibatch_size)
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
        scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
            utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

        if load_from_preexisting_model_state:
            params, state, opt_state = utl.restore_all_pytree_states(run_state["model_dir"])
        else:
            params, state = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
            opt_state = opt.init(params)
        # initial_params = copy.deepcopy(params)  # Keep a copy of the initial params for relative change metric
        # init_state = copy.deepcopy(state)
        frozen_layer_lists = utl.extract_layer_lists(params)
        # ordered_layers = utl.extract_ordered_layers(params)
        # print(jax.tree_map(jnp.shape, params))

        starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, size)
        total_neurons, total_per_layer = starting_neurons, starting_per_layer
        init_total_neurons = copy.copy(total_neurons)
        init_total_per_layer = copy.copy(total_per_layer)

        initial_params_count = utl.count_params(params)

        reg_param = exp_config.reg_param
        decaying_reg_param = copy.deepcopy(exp_config.reg_param)
        decay_cycles = exp_config.reg_param_decay_cycles + int(exp_config.zero_end_reg_param)
        if decay_cycles == 2:
            reg_param_decay_period = int(0.8 * exp_config.training_steps)
        else:
            reg_param_decay_period = exp_config.training_steps // decay_cycles

        if exp_config.reg_param_schedule:
            if exp_config.reg_param_span:
                sched_end = exp_config.reg_param_span
            elif exp_config.zero_end_reg_param:
                sched_end = int(0.9 * exp_config.training_steps)
            else:
                sched_end = exp_config.training_steps
            reg_sched = reg_param_scheduler_choice[exp_config.reg_param_schedule](sched_end, reg_param)
            if exp_config.add_noise:
                noise_sched = reg_param_scheduler_choice[exp_config.reg_param_schedule](sched_end, exp_config.noise_eta)

        subrun_start_time = time.time()

        if load_from_preexisting_model_state:
            starting_step = run_state["training_step"]
            decaying_reg_param = run_state["decaying_reg_param"]
            dropout_key = run_state["dropout_key"]
            load_from_preexisting_model_state = False
        else:
            starting_step = 0
        print(f"Continuing training from step {starting_step} and sparsity_level {sparsity}")
        for step in range(starting_step, exp_config.training_steps):
            if (decay_cycles > 1) and (step % reg_param_decay_period == 0) and \
                    (not (step % (exp_config.training_steps - 1) == 0)) and (not exp_config.reg_param_schedule):
                decaying_reg_param = decaying_reg_param / 10
                if (exp_config.training_steps // reg_param_decay_period) == decay_cycles:
                    decaying_reg_param = 0
                loss.clear_cache()
                test_loss_fn.clear_cache()
                update_fn.clear_cache()
                loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=decaying_reg_param,
                                               classes=classes, with_dropout=with_dropout,
                                               exclude_bias_bn_from_reg=exp_config.masked_reg,
                                               label_smoothing=exp_config.label_smoothing)
                test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                       reg_param=decaying_reg_param,
                                                       classes=classes, is_training=False, with_dropout=with_dropout,
                                                       exclude_bias_bn_from_reg=exp_config.masked_reg)
                utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                    exp_config.asymmetric_noise,
                                                    live_only=exp_config.noise_live_only,
                                                    with_dropout=with_dropout)
            if exp_config.reg_param_schedule and step < exp_config.training_steps:
                decaying_reg_param = reg_sched(step)
            if (step > 0) and exp_config.preempt_handling and (step % (exp_config.checkpoint_freq * steps_per_epoch) == 0):
                print(
                    f"Elapsed time in current run at step {step}: {timedelta(seconds=time.time() - subrun_start_time)}")
                chckpt_init_time = time.time()
                utl.jaxpruner_checkpoint_exp(run_state, params, state, opt_state, curr_epoch=step//steps_per_epoch,
                                             curr_step=step, curr_pruning_density=sparsity, dropout_key=dropout_key,
                                             decaying_reg_param=decaying_reg_param)
                print(
                    f"Checkpointing performed in: {timedelta(seconds=time.time() - chckpt_init_time)}")

            if step % exp_config.record_freq == 0:
                train_loss = test_loss_fn(params, state, next(train_eval), _reg_param=decaying_reg_param)
                train_accuracy = accuracy_fn(params, state, next(train_eval))
                test_accuracy = accuracy_fn(params, state, next(test_eval))
                test_loss = test_loss_fn(params, state, next(test_eval), _reg_param=decaying_reg_param)
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                # Periodically print classification accuracy on train & test sets.
                if step % exp_config.report_freq == 0:
                    print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                          f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
                test_death_batch = next(test_death)
                dead_neurons = death_check_fn(params, state, test_death_batch)
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                accuracies_log.append(test_accuracy)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                              context={"sparsity level": str(sparsity)})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step,
                              context={"sparsity level": str(sparsity)})
                if exp_config.epsilon_close:
                    for eps in exp_config.epsilon_close:
                        eps_dead_neurons = death_check_fn(params, state, test_death_batch, eps)
                        eps_dead_neurons_count, _ = utl.count_dead_neurons(eps_dead_neurons)
                        exp_run.track(jax.device_get(eps_dead_neurons_count),
                                      name="Quasi-dead neurons", step=step,
                                      context={"sparsity level": str(sparsity), "epsilon": eps})
                        exp_run.track(jax.device_get(total_neurons - eps_dead_neurons_count),
                                      name="Quasi-live neurons", step=step,
                                      context={"sparsity level": str(sparsity), "epsilon": eps})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"sparsity level": str(sparsity)})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={"sparsity level": str(sparsity)})
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                              context={"sparsity level": str(sparsity)})
                exp_run.track(jax.device_get(test_loss), name="Test loss", step=step,
                              context={"sparsity level": str(sparsity)})

            if step % exp_config.pruning_freq == 0:
                dead_neurons = scan_death_check_fn(params, state, test_death)
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                              step=step,
                              context={"sparsity level": str(sparsity)})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                              name="Live neurons; whole training dataset",
                              step=step,
                              context={"sparsity level": str(sparsity)})
                if exp_config.epsilon_close:
                    for eps in exp_config.epsilon_close:
                        eps_dead_neurons = scan_death_check_fn(params, state, test_death, eps)
                        eps_dead_neurons_count, eps_dead_per_layers = utl.count_dead_neurons(eps_dead_neurons)
                        exp_run.track(jax.device_get(eps_dead_neurons_count),
                                      name="Quasi-dead neurons; whole training dataset",
                                      step=step,
                                      context={"sparsity level": str(sparsity), "epsilon": eps})
                        exp_run.track(jax.device_get(total_neurons - eps_dead_neurons_count),
                                      name="Quasi-live neurons; whole training dataset",
                                      step=step,
                                      context={"sparsity level": str(sparsity), "epsilon": eps})
                for i, layer_dead in enumerate(dead_per_layers):
                    total_neuron_in_layer = total_per_layer[i]
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                                  context={"sparsity level": str(sparsity)})
                    exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                                  name=f"Live neurons in layer {i}; whole training dataset", step=step,
                                  context={"sparsity level": str(sparsity)})
                del dead_per_layers

                if exp_config.dynamic_pruning and step >= exp_config.prune_after:
                    # Pruning the network
                    params, opt_state, state, new_sizes = utl.remove_dead_neurons_weights(params, dead_neurons,
                                                                                          frozen_layer_lists, opt_state.inner_state,
                                                                                          state)
                    architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                        exp_config.architecture]
                    architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
                    net, _ = build_models(*architecture)
                    total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes)

                    # Clear previous cache
                    loss.clear_cache()
                    test_loss_fn.clear_cache()
                    accuracy_fn.clear_cache()
                    update_fn.clear_cache()
                    death_check_fn.clear_cache()
                    # scan_death_check_fn.clear_cache()
                    # eps_death_check_fn.clear_cache()  # No more cache
                    # eps_scan_death_check_fn.clear_cache()  # No more cache
                    # Recompile training/monitoring functions
                    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                   reg_param=decaying_reg_param, classes=classes,
                                                   with_dropout=with_dropout,
                                                   exclude_bias_bn_from_reg=exp_config.masked_reg,
                                                   label_smoothing=exp_config.label_smoothing)
                    test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                           reg_param=decaying_reg_param,
                                                           classes=classes,
                                                           is_training=False, with_dropout=with_dropout,
                                                           exclude_bias_bn_from_reg=exp_config.masked_reg)
                    accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
                    utl.update_given_loss_and_optimizer(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                        exp_config.asymmetric_noise,
                                                        live_only=exp_config.noise_live_only,
                                                        with_dropout=with_dropout)
                    death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
                    # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                    #                                                  epsilon=exp_config.epsilon_close,
                    #                                                  avg=exp_config.avg_for_eps)
                    scan_len = int(partial_train_ds_size // death_minibatch_size)
                    # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
                    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
                    scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
                        utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
                    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
                    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

                del dead_neurons  # Freeing memory
                if "imagenet" not in exp_config.dataset:
                    train_acc_whole_ds = full_train_acc_fn(params, state, train_eval)
                    exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                                  step=step,
                                  context={"sparsity level": str(sparsity)})

            # if ((step+1) % exp_config.live_freq == 0) and (step+2 < exp_config.training_steps):
            #     current_dead_neurons = scan_death_check_fn(params, state, test_death)
            #     current_dead_neurons_count, _ = utl.count_dead_neurons(current_dead_neurons)
            #     del current_dead_neurons
            #     del _
            #     exp_run.track(jax.device_get(total_neurons - current_dead_neurons_count),
            #                   name=f"Live neurons at training step {step+1}", step=starting_neurons)

            # Train step over single batch
            if with_dropout:
                params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next(train), dropout_key, _reg_param=decaying_reg_param)
                params = pruner.post_gradient_update(params, opt_state)
            else:
                if not exp_config.add_noise:
                    params, state, opt_state = update_fn(params, state, opt_state, next(train), _reg_param=decaying_reg_param)
                else:
                    noise_var = noise_sched(step)
                    params, state, opt_state, noise_key = update_fn(params, state, opt_state, next(train),
                                                                    noise_var,
                                                                    noise_key, _reg_param=decaying_reg_param)
                params = pruner.post_gradient_update(params, opt_state)

        final_accuracy = jax.device_get(final_accuracy_fn(params, state, test_eval))
        final_train_acc = jax.device_get(full_train_acc_fn(params, state, train_eval))

        activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state, test_death)
        # final_dead_neurons = scan_death_check_fn(params, test_death)

        # final_dead_neurons = jax.tree_map(utl.logical_and_sum, batched_dead_neurons)
        final_dead_neurons_count, final_dead_per_layer = utl.count_dead_neurons(final_dead_neurons)
            # pruned_params = utl.remove_dead_neurons_weights(params, final_dead_neurons,
        #                                                          frozen_layer_lists, opt_state.inner_state,
        #                                                          state)[0]
        # # final_params_count = utl.count_params(pruned_params)
        # final_params_count = utl.count_non_zero_params(pruned_params)
        final_params_count = int((1-sparsity) * initial_params_count)
        del final_dead_neurons  # Freeing memory

        activations_max, activations_mean, activations_count, _ = activations_data
        if exp_config.save_wanda:
            activations_meta.maximum.append(activations_max)
            activations_meta.mean.append(activations_mean)
            activations_meta.count.append(activations_count)
        activations_max, _ = ravel_pytree(activations_max)
        activations_max = jax.device_get(activations_max)
        activations_mean, _ = ravel_pytree(activations_mean)
        activations_mean = jax.device_get(activations_mean)
        activations_count, _ = ravel_pytree(activations_count)
        activations_count = jax.device_get(activations_count)

        batch_dead_neurons = death_check_fn(params, state, next(test_death))
        batches_final_live_neurons = [total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0]]
        for i in range(scan_len - 1):
            batch_dead_neurons = death_check_fn(params, state, next(test_death))
            batches_final_live_neurons.append(total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0])
        batches_final_live_neurons = jnp.stack(batches_final_live_neurons)

        avg_final_live_neurons = jnp.mean(batches_final_live_neurons, axis=0)
        std_final_live_neurons = jnp.std(batches_final_live_neurons, axis=0)

        log_step = sparsity * 100

        exp_run.track(jax.device_get(avg_final_live_neurons),
                      name="On average, live neurons after convergence w/r sparsity", step=log_step)
        exp_run.track(jax.device_get(avg_final_live_neurons / total_neurons),
                      name="Average live neurons ratio after convergence w/r sparsity", step=log_step)
        total_live_neurons = total_neurons - final_dead_neurons_count
        exp_run.track(jax.device_get(total_live_neurons),
                      name="Live neurons after convergence w/r sparsity", step=log_step)
        exp_run.track(jax.device_get(total_live_neurons / total_neurons),
                      name="Live neurons ratio after convergence w/r sparsity", step=log_step)
        exp_run.track(jax.device_get(sparsity),  # Logging true reg_param value to display with aim metrics
                      name="sparsity w/r sparsity", step=log_step)
        if exp_config.epsilon_close:
            for eps in exp_config.epsilon_close:
                eps_final_dead_neurons = scan_death_check_fn(params, state, test_death, eps)
                eps_final_dead_neurons_count, _ = utl.count_dead_neurons(eps_final_dead_neurons)
                del eps_final_dead_neurons
                eps_batch_dead_neurons = death_check_fn(params, state, next(test_death), eps)
                eps_batches_final_live_neurons = [total_neurons - utl.count_dead_neurons(eps_batch_dead_neurons)[0]]
                for i in range(scan_len - 1):
                    eps_batch_dead_neurons = death_check_fn(params, state, next(test_death), eps)
                    eps_batches_final_live_neurons.append(
                        total_neurons - utl.count_dead_neurons(eps_batch_dead_neurons)[0])
                eps_batches_final_live_neurons = jnp.stack(eps_batches_final_live_neurons)
                eps_avg_final_live_neurons = jnp.mean(eps_batches_final_live_neurons, axis=0)

                exp_run.track(jax.device_get(eps_avg_final_live_neurons),
                              name="On average, quasi-live neurons after convergence w/r sparsity",
                              step=log_step, context={"epsilon": eps})
                exp_run.track(jax.device_get(eps_avg_final_live_neurons / total_neurons),
                              name="Average quasi-live neurons ratio after convergence w/r sparsity",
                              step=log_step, context={"epsilon": eps})
                eps_total_live_neurons = total_neurons - eps_final_dead_neurons_count
                exp_run.track(jax.device_get(eps_total_live_neurons),
                              name="Quasi-live neurons after convergence w/r sparsity", step=log_step,
                              context={"epsilon": eps})
                exp_run.track(jax.device_get(eps_total_live_neurons / total_neurons),
                              name="Quasi-live neurons ratio after convergence w/r sparsity",
                              step=log_step, context={"epsilon": eps})

        for i, layer_dead in enumerate(final_dead_per_layer):
            total_neuron_in_layer = init_total_per_layer[i]
            live_in_layer = total_neuron_in_layer - layer_dead
            exp_run.track(jax.device_get(live_in_layer),
                          name=f"Live neurons in layer {i} after convergence w/r sparsity",
                          step=log_step)
            exp_run.track(jax.device_get(live_in_layer / total_neuron_in_layer),
                          name=f"Live neurons ratio in layer {i} after convergence w/r sparsity",
                          step=log_step)
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r sparsity", step=log_step)
        exp_run.track(final_train_acc,
                      name="Train accuracy after convergence w/r sparsity", step=log_step)
        log_sparsity_step = jax.device_get(total_live_neurons / init_total_neurons) * 1000
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r percent*10 of neurons remaining", step=log_sparsity_step)
        log_params_sparsity_step = final_params_count / initial_params_count * 1000
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r percent*10 of params remaining",
                      step=log_params_sparsity_step)
        # if not exp_config.dynamic_pruning:  # Cannot take norm between initial and pruned params
        #     params_vec, _ = ravel_pytree(params)
        #     initial_params_vec, _ = ravel_pytree(initial_params)
        #     exp_run.track(
        #         jax.device_get(jnp.linalg.norm(params_vec - initial_params_vec) / jnp.linalg.norm(initial_params_vec)),
        #         name="Relative change in norm of weights from init after convergence w/r sparsity",
        #         step=log_step)
        # activations_max_dist = Distribution(activations_max, bin_count=100)
        # exp_run.track(activations_max_dist, name='Maximum activation distribution after convergence', step=0,
        #               context={"sparsity level": str(sparsity)})
        # activations_mean_dist = Distribution(activations_mean, bin_count=100)
        # exp_run.track(activations_mean_dist, name='Mean activation distribution after convergence', step=0,
        #               context={"sparsity level": str(sparsity)})
        # activations_count_dist = Distribution(activations_count, bin_count=50)
        # exp_run.track(activations_count_dist, name='Activation count per neuron after convergence', step=0,
        #               context={"sparsity level": str(sparsity)})

        # Making sure compiled fn cache was cleared
        loss.clear_cache()
        test_loss_fn.clear_cache()
        accuracy_fn.clear_cache()
        update_fn.clear_cache()
        death_check_fn.clear_cache()

        # Print running time
        print()
        print(f"Running time for sparsity {sparsity}: " + str(timedelta(seconds=time.time() - subrun_start_time)))
        print("----------------------------------------------")
        print()

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
