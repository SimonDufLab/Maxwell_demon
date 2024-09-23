""" Experiment trying to verify if we can control overfitting (i.e. reduce memorization) in presence of noisy_label
by applying either cdg_l1 or cdg_l2 loss"""

import copy
import dataclasses

import omegaconf.listconfig
import optax
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import tensorflow as tf

import utils.grok_utils

tf.config.experimental.set_visible_devices([], "GPU")
from aim import Run, Distribution
import os
import time
from datetime import timedelta
import pickle
import json
import gc
import signal
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, List, Dict, Union
from ast import literal_eval
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path

from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

import utils.utils as utl
import utils.scores as scr
from utils.utils import build_models
from utils.config import activation_choice, optimizer_choice, dataset_choice, dataset_target_cardinality
from utils.config import regularizer_choice, architecture_choice, bn_architecture_choice, lr_scheduler_choice, bn_config_choice
from utils.config import reg_param_scheduler_choice
from utils.config import pick_architecture
from datasets.grok_datasets import vocab_size_mapping


# Experience name -> for aim logger
exp_name = "controlling_overfitting"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 250001
    report_freq: int = 2500
    record_freq: int = 250
    pruning_freq: int = 1000
    # live_freq: int = 25000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
    record_gate_grad_stat: bool = False  # Record in logger info about gradient magnitude per layer throughout training
    mod_via_gate_grad: bool = False  # Use gate gradients to rescale weight gradients if True -> shitty, don't use
    lr: float = 1e-3
    gradient_clipping: bool = False
    lr_schedule: str = "constant"
    warmup_ratio: float = 0.05  # ratio of total steps used for warming up lr, when applicable
    final_lr: float = 1e-6
    lr_decay_steps: Any = 5  # Number of epochs after which lr is decayed
    lr_decay_scaling_factor: float = 0.1  # scaling factor for lr decay
    train_batch_size: int = 512
    eval_batch_size: int = 512
    death_batch_size: int = 512
    accumulate_batches: int = 1  # Make effective batch size for training: train_batch_size x accumulate_batches
    optimizer: str = "adam"
    alpha_decay: float = 5.0  # Param controlling transition speed from adam to momentum in adam_to_momentum optimizers
    activation: str = "relu"  # Activation function used throughout the model
    shifted_relu: float = 0.0  # Shift value (b) applied on output before activation. To promote dead neurons
    srelu_sched: str = "constant"  # Schedule for shifted ReLU, same choices as reg_param_schedule
    dataset: str = "mnist"
    reduced_ds_size: Optional[int] = None  # Limit the size of ds to fix amounts for training
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
    grok_depth: int = 2  # Depth control for grokking models
    regularizer: Optional[str] = "cdg_l2"
    reg_params: Any = (0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
    masked_reg: Optional[str] = None  # If "all" exclude all bias and bn params, if "scale" only exclude scale param/ also offset_only, scale_only and bn_params_only
    wd_param: Optional[float] = None
    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
    reg_param_schedule: Optional[str] = None  # Schedule for reg_param, priority over reg_param_decay_cycles flag
    reg_param_span: Optional[int] = None # More general than zero_end_reg_param, allow to decide when reg_param schedule falls to 0
    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
    init_seed: int = 41
    dynamic_pruning: bool = False
    exclude_layer: Optional[str] = None
    prune_after: int = 0  # Option: only start pruning after <prune_after> step has been reached
    prune_at_end: Any = None  # If prune after training, tuple like (reg_param, lr, additional_steps)
    pretrain: int = 0  # Train only the normalization parameters for <pretrain> steps
    pretrain_targets: Any = "all"  # Parameters to pretrain, all normalization + head layers by default
    reset_during_pretrain: bool = False  # Reset dead neurons during pretraining instead of pruning them
    srelu_during_reset: float = 0.0  # Shift the ReLU to identify low-activation neurons, to target more for resetting
    sigm_pretrain: bool = False  # Apply sigmoid transformation to scale params, bounding effective values
    tanh_pretrain: bool = False  # Apply tanh transformation to scale params, bounding effective values
    clip_norm: Any = None  # Set as (scale_min_val, scale_max_val, offset_min_val, offset_max_val) for pretrain
    temperature: Optional[float] = None  # Temperature parameter for the temperature-adjusted softmax fc layer
    resize_old: Optional[int] = None  # Resize a previous run for reset -- architectural search
    old_run: Optional[str] = None  #  Previous run ID to use for above.
    record_pretrain_distribution: bool = False  # Monitor bn params distribution or not
    pruning_reg: Optional[str] = "cdg_l2"
    pruning_opt: str = "momentum9"  # Optimizer for pruning part after initial training
    add_noise: bool = False  # Add Gaussian noise to the gradient signal
    asymmetric_noise: bool = True  # Use an asymmetric noise addition, not applied to all neurons' weights
    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones. reverse if False
    noise_offset_only: bool = False  # Special option to only add noise to offset parameters of normalization layers
    positive_offset: bool = False  # Force the noise on offset to be solely positive (increasing revival rate)
    going_wild: bool = False  # This is not ok...
    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
    noise_eta: float = 0.01  # Variance of added noise; can only be used with a reg_param_schedule that it will match
    noise_gamma: float = 0.0
    noise_seed: int = 1
    dropout_rate: float = 0
    perturb_param: float = 0  # Perturbation parameter for rnadam
    with_rng_seed: int = 428
    # linear_switch: bool = False  # Whether to switch mid-training steps to linear activations
    measure_linear_perf: bool = False  # Measure performance over the linear network without changing activation
    record_distribution_data: bool = False  # Whether to record distribution at end of training -- high memory usage
    preempt_handling: bool = False  # Frequent checkpointing to handle SLURM preemption
    jobid: Optional[str] = None  # Manually restart previous job from checkpoint
    checkpoint_freq: int = 1  # in epochs
    save_wanda: bool = False  # Whether to save weights and activations value or not
    save_act_only: bool = True  # Only saving distributions with wanda, not the weights
    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

    # def __post_init__(self):
    #     if type(self.sizes) == str:
    #         self.sizes = literal_eval(self.sizes)


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)

# Using tf on CPU for data loading
# tf.config.experimental.set_visible_devices([], "GPU") # Set earlier


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
    assert exp_config.architecture in architecture_choice.keys() or exp_config.architecture in bn_architecture_choice.keys(), "Current architectures available: " + str(
        architecture_choice.keys())
    assert exp_config.activation in activation_choice.keys(), "Current activation function available: " + str(
        activation_choice.keys())
    assert exp_config.lr_schedule in lr_scheduler_choice.keys(), "Current lr scheduler function available: " + str(
        lr_scheduler_choice.keys())
    assert exp_config.bn_config in bn_config_choice.keys(), "Current batchnorm configurations available: " + str(
        bn_config_choice.keys())
    # assert exp_config.reg_param_schedule in reg_param_scheduler_choice.keys(), "Current reg param scheduler available: " + str(
    #     reg_param_scheduler_choice.keys())
    if exp_config.record_distribution_data:
        assert not exp_config.dynamic_pruning, "Dynamic pruning must be disabled to record meaningful distribution data"

    utl.reformat_dict_config(exp_config)
    # if exp_config.regularizer == 'None':
    #     exp_config.regularizer = None
    # if exp_config.wd_param == 'None':
    #     exp_config.wd_param = None
    # if exp_config.reg_param_schedule == 'None':
    #     exp_config.reg_param_schedule = None
    # if exp_config.prune_at_end == 'None':
    #     exp_config.prune_at_end = None
    assert (not (("adamw" in exp_config.optimizer) and bool(
        exp_config.regularizer))) or bool(exp_config.wd_param), "Set wd_param if adamw is used with a regularization loss"
    assert type(exp_config.pretrain_targets) is str, "The targeted layers for pretraining need to be specified as a single string, separated by commas (,)"
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.reg_params) == str:
        exp_config.reg_params = literal_eval(exp_config.reg_params)
    if type(exp_config.noise_imp) == str:
        exp_config.noise_imp = literal_eval(exp_config.noise_imp)
    if type(exp_config.epsilon_close) == str:
        exp_config.epsilon_close = literal_eval(exp_config.epsilon_close)
    if type(exp_config.prune_at_end) == str:
        exp_config.prune_at_end = literal_eval(exp_config.prune_at_end)
    if type(exp_config.lr_decay_steps) == str:
        exp_config.lr_decay_steps = literal_eval(exp_config.lr_decay_steps)
    # if exp_config.add_noise:
    #     exp_config.regularizer = None  # Disable regularizer when noise is used to promote neurons death
    if type(exp_config.clip_norm) == str:
        exp_config.clip_norm = literal_eval(exp_config.clip_norm)
    exp_config.pretrain_targets = tuple(exp_config.pretrain_targets.split(','))

    if exp_config.dynamic_pruning:
        exp_name_ = exp_name+"_with_dynamic_pruning"
    elif "imagenet" in exp_config.dataset:
        exp_name_ = "imgnet_"+exp_name
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
            if exp_config.old_run and exp_config.resize_old:
                load_from_preexisting_model_state = True
                load_old_dir = SCRATCH / exp_name_ / exp_config.old_run
                run_state = utl.load_run_state(load_old_dir)
                run_state['epoch'] = 0
                run_state['training_step'] = 0
                run_state["aim_hash"] = None
                run_state['exp_name'] = exp_name_
                if type(run_state['curr_starting_size']) is int:
                    run_state['curr_arch_sizes'] = list(utl.get_total_neurons(exp_config.architecture, run_state['curr_arch_sizes'])[1])
                run_state['curr_starting_size'] = [exp_config.resize_old * curr_layer_size for curr_layer_size in
                                                   run_state['curr_arch_sizes']]
                run_state['curr_arch_sizes'] = run_state['curr_starting_size']
                SLURM_JOBID = os.environ["SLURM_JOBID"]
                exp_config.jobid = SLURM_JOBID
                run_state['slurm_jobid'] = SLURM_JOBID
                run_state['curr_reg_param'] = exp_config.reg_params[0]
                run_state['decaying_reg_param'] = exp_config.reg_params[0]
                run_state['training_time'] = 0.0
            else:
                run_state = utl.RunState(epoch=0, training_step=0, model_dir=saving_dir, curr_arch_sizes=exp_config.size,
                                         aim_hash=None, slurm_jobid=SLURM_JOBID, exp_name=exp_name_,
                                         curr_starting_size=exp_config.size, curr_reg_param=exp_config.reg_params[0],
                                         dropout_key=jax.random.PRNGKey(exp_config.with_rng_seed),
                                         decaying_reg_param=exp_config.reg_params[0],
                                         best_accuracy=0.0, best_params_count=None, best_total_neurons=None,
                                         training_time=0.0, reset_counter=None, reset_tracker=None,
                                         cumulative_dead_neurons=None)
                # with open(os.path.join(saving_dir, "checkpoint_run_state.pkl"), "wb") as f:  # Save only if one additional epoch completed
                #     pickle.dump(run_state, f)

                # Dump config file in it as well
                with open(os.path.join(saving_dir, 'config.json'), 'w') as fp:
                    json.dump(OmegaConf.to_container(exp_config), fp, indent=4)

        aim_hash = run_state["aim_hash"]
    else:
        aim_hash = None
    
    # Path for logs
    if exp_config.perturb_param:
        log_path = "./ICLR2023_rnadam"
    elif exp_config.bn_config == "deactivate_small_units":
        log_path = "./exploration__deactivate_small_units"
    else:
        log_path = "./ICML2024_rebuttal_main"  # "./preempt_test"  #
    if "imagenet" in exp_config.dataset:
        log_path = "./imagenet_exps_post_ICML"
    if exp_config.pretrain > 0:
        log_path = "./boosted_initialization_exps"
    if "grok" in exp_config.architecture:  # Override pretrain option
        log_path = "./grok_exps"
    if "color_mnist" in exp_config.dataset:  # Override pretrain and grok
        log_path = "./spurious_experiments"
    # Logger config
    exp_run = Run(repo=log_path, experiment=exp_name_, run_hash=aim_hash, force_resume=True)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)
    if exp_config.preempt_handling:
        run_state["aim_hash"] = exp_run.hash

    if exp_config.save_wanda:
        # Create pickle directory
        pickle_dir_path = log_path + "/metadata/" + exp_name_ + time.strftime("/%Y-%m-%d---%B %d---%H:%M:%S/")
        os.makedirs(pickle_dir_path)
        # Dump config file in it as well
        with open(pickle_dir_path+'config.json', 'w') as fp:
            json.dump(OmegaConf.to_container(exp_config), fp, indent=4)

    # Batch accumulation
    if exp_config.accumulate_batches > 1:
        get_updater = Partial(utl.update_with_accumulated_grads, accumulated_grads=exp_config.accumulate_batches)
    elif exp_config.pretrain:
        get_updater = utl.get_mask_update_fn
    else:
        get_updater = utl.update_given_loss_and_optimizer

    # Experiments with dropout
    with_dropout = exp_config.dropout_rate > 0
    if with_dropout:
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
        if exp_config.sigm_pretrain and exp_config.pretrain > 0:
            net_config['bn_config']['sigm_scale'] = True
        if exp_config.tanh_pretrain and exp_config.pretrain > 0:
            net_config['bn_config']['tanh_scale'] = True

    if 'grok' in exp_config.architecture:
        net_config['vocab_size'] = vocab_size_mapping[exp_config.dataset]
        net_config['depth'] = exp_config.grok_depth

    if exp_config.temperature:
        net_config['temperature'] = exp_config.temperature

    # warmup:
    if 'warmup' in exp_config.lr_schedule:
        sched_config = {"warmup_ratio": exp_config.warmup_ratio}
    else:
        sched_config = {}

    # Load the different dataset
    if exp_config.kept_classes:
        assert exp_config.kept_classes <= dataset_target_cardinality[
            exp_config.dataset], "subset must be smaller or equal to total number of classes in ds"
        kept_indices = np.random.choice(dataset_target_cardinality[exp_config.dataset], exp_config.kept_classes,
                                        replace=False)
    else:
        kept_indices = None
    load_data = Partial(dataset_choice[exp_config.dataset], reduced_ds_size=exp_config.reduced_ds_size)
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
    steps_per_epoch = train_ds_size // (exp_config.train_batch_size*exp_config.accumulate_batches)
    if 'imagenet' in exp_config.dataset:
        partial_train_ds_size = train_ds_size/1000  # .1% of dataset used for evaluation on train
        test_death = train_eval  # Don't want to prefetch too many ds
    else:
        partial_train_ds_size = train_ds_size / 25

    # # Recording over all widths
    # live_neurons = []
    # avg_live_neurons = []
    # std_live_neurons = []
    # size_arr = []
    # f_acc = []

    if exp_config.save_wanda:
        # Recording metadata about activations that will be pickled
        @dataclass
        class ActivationMeta:
            maximum: Dict[float, List[float]] = field(default_factory=dict)
            mean: Dict[float, List[float]] = field(default_factory=dict)
            count: Dict[float, List[int]] = field(default_factory=dict)
        activations_meta = ActivationMeta()

        if not exp_config.save_act_only:
            # Recording params value at the end of the training as well
            @dataclass
            class FinalParamsMeta:
                parameters: List[float] = field(default_factory=list)
            params_meta = FinalParamsMeta()

    size = exp_config.size

    signal.signal(signal.SIGTERM, utl.signal_handler)  # Before getting pre-empted and requeued.
    signal.signal(signal.SIGUSR1, utl.signal_handler)  # Before reaching the end of the time limit.

    def train_run(reg_param):
        nonlocal load_from_preexisting_model_state

        def record_metrics_and_prune(step, reg_param, activation_fn, decaying_reg_param, net, new_sizes, params, state, opt_state, opt,
                                     total_neurons, total_per_layer, loss, test_loss_fn, accuracy_fn, death_check_fn,
                                     scan_death_check_fn, full_train_acc_fn, final_accuracy_fn, update_fn,
                                     dead_neurons_union, pretrain_mask, init_fn, init_key, reset_counter, reset_tracker):
            """ Inside a function to make sure variables in function scope are cleared from memory"""
            if step == exp_config.training_steps and bool(add_steps_end):
                print("Entered pruning phase")
                #  Reset optimizer:
                optimizer = optimizer_choice[exp_config.pruning_opt]
                opt_chain = []
                if "w" in exp_config.pruning_opt:  # Pass reg_param to wd argument of adamw
                    if exp_config.wd_param:  # wd_param overwrite reg_param when specified
                        optimizer = Partial(optimizer, weight_decay=exp_config.wd_param)
                    else:
                        optimizer = Partial(optimizer, weight_decay=reg_param)
                elif exp_config.wd_param:  # TODO: Maybe exclude adamw?
                    opt_chain.append(optax.add_decayed_weights(weight_decay=exp_config.wd_param))
                # if exp_config.gradient_clipping:
                #     opt_chain.append(optax.clip(0.1))
                lr_schedule = lr_scheduler_choice["one_cycle"](add_steps_end, pruning_lr, None, None)  # TODO: fixed schd...
                opt_chain.append(optimizer(lr_schedule))
                opt = optax.chain(*opt_chain)
                opt_state = opt.init(params)  # TODO: resetting the step counter impact lr schedule, be wary
                state = init_state  # Reset state as well
                # Reset losses etc.
                # utl.clear_caches()
                # loss.clear_cache()
                # test_loss_fn.clear_cache()
                # update_fn.clear_cache()
                decaying_reg_param = pruning_reg_param
                loss = utl.ce_loss_given_model(net, regularizer=exp_config.pruning_reg, reg_param=pruning_reg_param,
                                               classes=classes, with_dropout=with_dropout,
                                               exclude_bias_bn_from_reg=exp_config.masked_reg,
                                               label_smoothing=exp_config.label_smoothing)
                test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.pruning_reg,
                                                       reg_param=pruning_reg_param,
                                                       classes=classes, is_training=False, with_dropout=with_dropout,
                                                       exclude_bias_bn_from_reg=exp_config.masked_reg)
                update_fn = get_updater(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                                exp_config.asymmetric_noise,
                                                                going_wild=exp_config.going_wild,
                                                                live_only=exp_config.noise_live_only,
                                                                noise_offset_only=exp_config.noise_offset_only,
                                                                positive_offset=exp_config.positive_offset,
                                                                with_dropout=with_dropout,
                                                                modulate_via_gate_grad=exp_config.mod_via_gate_grad,
                                                                acti_map=acti_map, perturb=exp_config.perturb_param,
                                                                init_fn=init_fn)

            if (decay_cycles > 1) and (step % reg_param_decay_period == 0) and \
                    (not (step % (exp_config.training_steps - 1) == 0)) and (not exp_config.reg_param_schedule):
                decaying_reg_param = decaying_reg_param / 10
                if (step >= ((decay_cycles - 1) * reg_param_decay_period)) and exp_config.zero_end_reg_param:
                    decaying_reg_param = 0
                print("decaying reg param:")
                print(decaying_reg_param)
                print()
                # utl.clear_caches()
                # loss.clear_cache()
                # test_loss_fn.clear_cache()
                # update_fn.clear_cache()
                loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=decaying_reg_param,
                                               classes=classes, with_dropout=with_dropout,
                                               exclude_bias_bn_from_reg=exp_config.masked_reg,
                                               label_smoothing=exp_config.label_smoothing)
                test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
                                                       reg_param=decaying_reg_param,
                                                       classes=classes, is_training=False, with_dropout=with_dropout,
                                                       exclude_bias_bn_from_reg=exp_config.masked_reg)
                update_fn = get_updater(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                                exp_config.asymmetric_noise,
                                                                going_wild=exp_config.going_wild,
                                                                live_only=exp_config.noise_live_only,
                                                                noise_offset_only=exp_config.noise_offset_only,
                                                                positive_offset=exp_config.positive_offset,
                                                                with_dropout=with_dropout,
                                                                modulate_via_gate_grad=exp_config.mod_via_gate_grad,
                                                                acti_map=acti_map, perturb=exp_config.perturb_param,
                                                                init_fn=init_fn)

            if step % exp_config.record_freq == 0:
                train_loss = test_loss_fn(params, state, next(train_eval), _reg_param=decaying_reg_param)
                train_accuracy = accuracy_fn(params, state, next(train_eval))
                test_accuracy = accuracy_fn(params, state, next(test_eval))
                test_loss = test_loss_fn(params, state, next(test_eval), _reg_param=decaying_reg_param)
                # train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                # Periodically print classification accuracy on train & test sets.
                if step % exp_config.report_freq == 0:
                    print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                          f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
                test_death_batch = next(test_death)
                dead_neurons = death_check_fn(params, state, test_death_batch)
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                exp_run.track(dead_neurons_count, name="Dead neurons", step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                exp_run.track(total_neurons - dead_neurons_count, name="Live neurons", step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                if exp_config.epsilon_close:
                    for eps in exp_config.epsilon_close:
                        eps_dead_neurons = death_check_fn(params, state, test_death_batch, eps)
                        eps_dead_neurons_count, _ = utl.count_dead_neurons(eps_dead_neurons)
                        exp_run.track(eps_dead_neurons_count,
                                      name="Quasi-dead neurons", step=step,
                                      context={"reg param": utl.size_to_string(reg_param), "epsilon": eps})
                        exp_run.track(total_neurons - eps_dead_neurons_count,
                                      name="Quasi-live neurons", step=step,
                                      context={"reg param": utl.size_to_string(reg_param), "epsilon": eps})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                exp_run.track(train_loss, name="Train loss", step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                exp_run.track(test_loss, name="Test loss", step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                # for layer in params.keys():
                #     if "offset" in params[layer].keys():
                #         exp_run.track(jnp.mean(params[layer]["offset"]), name="BN offset trough training", step=step,
                #                       context={"reg param": utl.size_to_string(reg_param), 'layer': layer})
                #     if "scale" in params[layer].keys():
                #         exp_run.track(jnp.mean(params[layer]["scale"]), name="BN scale trough training", step=step,
                #                       context={"reg param": utl.size_to_string(reg_param), 'layer': layer})
                if 'grok' in exp_config.architecture:
                    _params = utils.grok_utils.mask_ff_init_layer(params, 'wb')
                    masked_accuracy = accuracy_fn(_params, state, next(test_eval))
                    exp_run.track(masked_accuracy,
                                  name="Test accuracy, with masking on w and b of init layer", step=step)
                    masked_accuracy = accuracy_fn(_params, state, next(train_eval))
                    exp_run.track(masked_accuracy,
                                  name="Train accuracy, with masking on w and b of init layer", step=step)
                    _params = utils.grok_utils.mask_ff_init_layer(params, 'w')
                    masked_accuracy = accuracy_fn(_params, state, next(test_eval))
                    exp_run.track(masked_accuracy,
                                  name="Test accuracy, with masking on w only of init layer", step=step)
                    masked_accuracy = accuracy_fn(_params, state, next(train_eval))
                    exp_run.track(masked_accuracy,
                                  name="Train accuracy, with masking on w only of init layer", step=step)
                    _params = utils.grok_utils.mask_ff_last_layer(params, 'wb')
                    masked_accuracy = accuracy_fn(_params, state, next(test_eval))
                    exp_run.track(masked_accuracy,
                                  name="Test accuracy, with masking on w and b of second layer", step=step)
                    masked_accuracy = accuracy_fn(_params, state, next(train_eval))
                    exp_run.track(masked_accuracy,
                                  name="Train accuracy, with masking on w and b of second layer", step=step)
                    _params = utils.grok_utils.mask_ff_last_layer(params, 'w')
                    masked_accuracy = accuracy_fn(_params, state, next(test_eval))
                    exp_run.track(masked_accuracy,
                                  name="Test accuracy, with masking on w only of second layer", step=step)
                    masked_accuracy = accuracy_fn(_params, state, next(train_eval))
                    exp_run.track(masked_accuracy,
                                  name="Train accuracy, with masking on w only of second layer", step=step)

            if step % exp_config.pruning_freq == 0:
                dead_neurons = scan_death_check_fn(params, state, test_death)
                if not exp_config.dynamic_pruning:
                    overlap = jax.tree_map(jnp.logical_and, dead_neurons_union, dead_neurons)
                    overlap = sum(jax.tree_map(jnp.sum, overlap)) / sum(jax.tree_map(jnp.sum, dead_neurons_union))
                    dead_neurons_union = jax.tree_map(jnp.logical_or, dead_neurons_union, dead_neurons)
                    exp_run.track(overlap, name="Cumulative overlap of dead neurons",
                                  step=step,
                                  context={"reg param": utl.size_to_string(reg_param)})
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)
                exp_run.track(dead_neurons_count, name="Dead neurons; whole training dataset",
                              step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                exp_run.track(total_neurons - dead_neurons_count,
                              name="Live neurons; whole training dataset",
                              step=step,
                              context={"reg param": utl.size_to_string(reg_param)})
                if exp_config.record_pretrain_distribution:
                    batch_activations, _ = utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout, avg=True)(params, state, next(test_death))
                    for layer_number in range(len(batch_activations)):
                        layer_activations = jnp.mean(batch_activations[layer_number], axis=0)
                        activations_dist = Distribution(layer_activations, bin_count=100)
                        exp_run.track(activations_dist, name='Activation distribution in layer {}'.format(layer_number),
                                      step=step,
                                      context={"reg param": utl.size_to_string(reg_param)})
                    for layer_name, layer_params in params.items():
                        if ('norm' in layer_name) or ('bn' in layer_name):
                            for subkey, subval in layer_params.items():
                                norm_dist = jnp.squeeze(subval)
                                norm_param_dist = Distribution(norm_dist, bin_count=100)
                                exp_run.track(norm_param_dist, name='{} param distribution'.format(subkey),
                                              step=step,
                                              context={"reg param": utl.size_to_string(reg_param),
                                                       "layer": layer_name})
                    if reset_counter is not None:
                        reset_count_dist = Distribution(jnp.concatenate(jax.tree_util.tree_leaves(reset_counter)), bin_count=100)
                        exp_run.track(reset_count_dist, name='All layers reset counter', step=step, context={"reg param": utl.size_to_string(reg_param)})
                        for acti_layer_name, layer_counter in reset_counter.items():
                            reset_count_dist = Distribution(layer_counter, bin_count=100)
                            exp_run.track(reset_count_dist, name='Layerwise reset counter', step=step,
                                          context={"reg param": utl.size_to_string(reg_param), "layer": acti_layer_name})
                    if reset_tracker is not None:
                        reset_tracker_dist = Distribution(jnp.concatenate(jax.tree_util.tree_leaves(reset_tracker)), bin_count=100)
                        exp_run.track(reset_tracker_dist, name='Reset step per neuron', step=step,
                                      context={"reg param": utl.size_to_string(reg_param)})

                if exp_config.epsilon_close:
                    for eps in exp_config.epsilon_close:
                        eps_dead_neurons = scan_death_check_fn(params, state, test_death, eps)
                        eps_dead_neurons_count, eps_dead_per_layers = utl.count_dead_neurons(eps_dead_neurons)
                        exp_run.track(eps_dead_neurons_count,
                                      name="Quasi-dead neurons; whole training dataset",
                                      step=step,
                                      context={"reg param": utl.size_to_string(reg_param), "epsilon": eps})
                        exp_run.track(total_neurons - eps_dead_neurons_count,
                                      name="Quasi-live neurons; whole training dataset",
                                      step=step,
                                      context={"reg param": utl.size_to_string(reg_param), "epsilon": eps})
                for i, layer_dead in enumerate(dead_per_layers):
                    total_neuron_in_layer = total_per_layer[i]
                    exp_run.track(layer_dead,
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                                  context={"reg param": utl.size_to_string(reg_param)})
                    exp_run.track(total_neuron_in_layer - layer_dead,
                                  name=f"Live neurons in layer {i}; whole training dataset", step=step,
                                  context={"reg param": utl.size_to_string(reg_param)})
                del dead_per_layers
                # if decay_cycles > 1:  # Don't record, aim metric don't support y value smaller than 0.001 ...
                #     exp_run.track(decaying_reg_param, name="Current reg_param value", step=step,
                #                   context={"reg param": utl.size_to_string(reg_param)})

                if exp_config.shifted_relu:
                    __flag = True
                    __count = 0
                    state_layers = list(state.keys())
                    while __flag:
                        if "shift_constant" in state[state_layers[__count]]:
                            print(f"Current negative shift pre-relu: {state[state_layers[__count]]['shift_constant']}")
                            __flag = False
                        else:
                            __count += 1


                if exp_config.record_gate_grad_stat:
                    snap_score = scr.snap_score(params, state, test_loss_fn, train_eval, 5)  # Avg on 5 minibatches
                    gate_grad.update(snap_score)
                    for i, layer_gate_grad in enumerate(gate_grad.values()):  # Ordered dict retrieves layers in order
                        exp_run.track(jnp.mean(layer_gate_grad),
                                      name=f"Average gate gradients magnitude in layer {i}; whole training dataset",
                                      step=step,
                                      context={"reg param": utl.size_to_string(reg_param)})

                if exp_config.measure_linear_perf:
                    # Record performance over full validation set of the NN for relu and decaying_reg_paramlinear activations
                    relu_perf = final_accuracy_fn(params, state, test_eval)
                    exp_run.track(relu_perf,
                                  name="Total accuracy for relu NN", step=step,
                                  context={"reg param": utl.size_to_string(reg_param)})
                    lin_perf = lin_full_accuracy_fn(params, state, test_eval)
                    exp_run.track(lin_perf,
                                  name="Total accuracy for linear NN", step=step,
                                  context={"reg param": utl.size_to_string(reg_param)})

                if exp_config.dynamic_pruning and step >= exp_config.prune_after:
                    neuron_states.update_from_ordered_list(dead_neurons)
                    if exp_config.reset_during_pretrain and step < exp_config.pretrain:
                        if exp_config.srelu_during_reset:
                            _state = copy.deepcopy(state)
                            _state = utl.update_gate_constant(_state, exp_config.srelu_during_reset)
                            _dead_neurons = scan_death_check_fn(params, _state, test_death)
                            _neuron_states = utl.NeuronStates(activation_layer_order)
                            _neuron_states.update_from_ordered_list(_dead_neurons)
                        else:
                            _neuron_states = neuron_states
                        if step == 0:
                            reset_counter = jax.tree_map(Partial(jnp.ones_like, dtype=int), _neuron_states.state())
                            reset_tracker = jax.tree_map(Partial(jnp.zeros_like, dtype=int), _neuron_states.state())
                        else:
                            reset_counter = jax.tree_map(jnp.add, reset_counter,
                                                         jax.tree_map(Partial(jnp.asarray, dtype=int),
                                                                      _neuron_states.state()))
                            reset_tracker = jax.tree_map(Partial(utils.utils.map_decision_with_bool_array, potential_leaf=step),
                                                         _neuron_states.invert_state(), reset_tracker)
                        # reinitialize dead neurons
                        init_key, _key = jax.random.split(init_key)
                        new_params, new_state = net.init(_key, next(train))
                        params = utl.reinitialize_dead_neurons(acti_map, _neuron_states, params,
                                                                              new_params)
                        # state = new_state  # TODO: Revise after testing: could be useful to swing reinit neurons with adam
                        # opt_state = opt.init(params)
                    else:
                        # Pruning the network
                        params, opt_state, state, new_sizes = utl.prune_params_state_optstate(params, acti_map,
                                                                                              neuron_states, opt_state,
                                                                                              state)

                        architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                            exp_config.architecture]
                        architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
                        net = build_models(*architecture)[0]
                        init_fn = utl.get_init_fn(net, ones_init)
                        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes, exp_config.grok_depth)

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
                        update_fn = get_updater(loss, opt, exp_config.add_noise,
                                                                        exp_config.noise_imp, exp_config.asymmetric_noise,
                                                                        going_wild=exp_config.going_wild,
                                                                        live_only=exp_config.noise_live_only,
                                                                        noise_offset_only=exp_config.noise_offset_only,
                                                                        positive_offset=exp_config.positive_offset,
                                                                        with_dropout=with_dropout,
                                                                        modulate_via_gate_grad=exp_config.mod_via_gate_grad,
                                                                        acti_map=acti_map, perturb=exp_config.perturb_param,
                                                                        init_fn=init_fn)
                        death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
                        # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                        #                                                  epsilon=exp_config.epsilon_close,
                        #                                                  avg=exp_config.avg_for_eps)
                        # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
                        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
                        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
                        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))
                        if exp_config.pretrain and (step <= exp_config.pretrain):
                            pretrain_mask = {'_mask': utils.grok_utils.mask_untargeted_weights(params, exp_config.pretrain_targets)}

                if "imagenet" not in exp_config.dataset:
                    train_acc_whole_ds = full_train_acc_fn(params, state, train_eval)
                    exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                                  step=step,
                                  context={"reg param": utl.size_to_string(reg_param)})

            # if ((step+1) % exp_config.live_freq == 0) and (step+2 < exp_config.training_steps):
            #     current_dead_neurons = scan_death_check_fn(params, state, test_death)
            #     current_dead_neurons_count, _ = utl.count_dead_neurons(current_dead_neurons)
            #     del current_dead_neurons
            #     del _
            #     exp_run.track(jax.device_get(total_neurons - current_dead_neurons_count),
            #                   name=f"Live neurons at training step {step+1}", step=starting_neurons)

            # if (((step + 1) % (exp_config.training_steps // 2)) == 0) and exp_config.linear_switch:
            #     activation_fn = activation_choice["linear"]
            #     architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
            #         exp_config.architecture]
            #     architecture = architecture(size, classes, activation_fn=activation_fn, **net_config)
            #     net = build_models(*architecture, with_dropout=with_dropout)[0]
            #     init_fn = utl.get_init_fn(net, ones_init)
            #
            #     # Reset training/monitoring functions
            #     # utl.clear_caches()
            #     loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=decaying_reg_param,
            #                                    classes=classes, with_dropout=with_dropout,
            #                                    exclude_bias_bn_from_reg=exp_config.masked_reg,
            #                                    label_smoothing=exp_config.label_smoothing)
            #     test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer,
            #                                            reg_param=decaying_reg_param,
            #                                            classes=classes, is_training=False, with_dropout=with_dropout,
            #                                            exclude_bias_bn_from_reg=exp_config.masked_reg)
            #     accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
            #     update_fn = get_updater(loss, opt, exp_config.add_noise, exp_config.noise_imp,
            #                                                     exp_config.asymmetric_noise,
            #                                                     going_wild=exp_config.going_wild,
            #                                                     live_only=exp_config.noise_live_only,
            #                                                     noise_offset_only=exp_config.noise_offset_only,
            #                                                     positive_offset=exp_config.positive_offset,
            #                                                     with_dropout=with_dropout,
            #                                                     modulate_via_gate_grad=exp_config.mod_via_gate_grad,
            #                                                     acti_map=acti_map, perturb=exp_config.perturb_param,
            #                                                     init_fn=init_fn)
            #     death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
            #     # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
            #     #                                                  epsilon=exp_config.epsilon_close,
            #     #                                                  avg=exp_config.avg_for_eps)
            #     scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
            #     # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
            #     final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
            #     full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

            return (decaying_reg_param, net, new_sizes, params, state, opt_state, opt, total_neurons, total_per_layer,
                    loss, test_loss_fn, accuracy_fn, death_check_fn, scan_death_check_fn, full_train_acc_fn,
                    final_accuracy_fn, update_fn, dead_neurons_union, pretrain_mask, init_fn, init_key,
                    reset_counter, reset_tracker)

        # Make the network and optimiser
        architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
            exp_config.architecture]
        if not exp_config.kept_classes:
            classes = dataset_target_cardinality[exp_config.dataset]  # Retrieving the number of classes in dataset
        else:
            classes = exp_config.kept_classes
        if load_from_preexisting_model_state:
            new_sizes = run_state["curr_arch_sizes"]
        else:
            new_sizes = size
        architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
        net, raw_net = build_models(*architecture, with_dropout=with_dropout)

        ones_init = jnp.ones_like(next(train)[0]), jnp.ones_like(next(train)[1])

        # def init_fn(rdm_key):
        #     return net.init(rdm_key, ones_init)[0]
        init_fn = utl.get_init_fn(net, ones_init)

        dropout_key = jax.random.PRNGKey(exp_config.with_rng_seed)

        if exp_config.measure_linear_perf:
            lin_act_fn = activation_choice["linear"]
            lin_architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                exp_config.architecture]
            lin_architecture = lin_architecture(size, classes, activation_fn=lin_act_fn, **net_config)
            lin_net, raw_net = build_models(*lin_architecture, with_dropout=with_dropout)

        optimizer = optimizer_choice[exp_config.optimizer]
        if "new_adam" in exp_config.optimizer:
            optimizer = Partial(optimizer, t_f=exp_config.training_steps, alpha=exp_config.alpha_decay)
        opt_chain = []
        if "loschi" in exp_config.optimizer:  # Using reg_param parameters to control wd with those optimizers
            if exp_config.reg_param_schedule:
                if exp_config.reg_param_span:
                    sched_end = exp_config.reg_param_span
                elif exp_config.zero_end_reg_param:
                    sched_end = int(0.9 * exp_config.training_steps)
                else:
                    sched_end = exp_config.training_steps
                if exp_config.wd_param:
                    div_factor = reg_param/exp_config.wd_param
                    final_div_factor = 1.0
                else:  # default values
                    div_factor = 25.0
                    final_div_factor = 1e4
                wd_schedule = reg_param_scheduler_choice[exp_config.reg_param_schedule](sched_end, reg_param,
                                                                                        div_factor=div_factor,
                                                                                        final_div_factor=final_div_factor)
            else:
                wd_schedule = reg_param if reg_param > 0.0 else exp_config.wd_param
                sched_end = exp_config.training_steps
            optimizer = Partial(optimizer, weight_decay=wd_schedule, sched_end=sched_end)
        elif "w" in exp_config.optimizer:  # Pass reg_param to wd argument of adamw #TODO: dangerous condition...
            if exp_config.wd_param:  # wd_param overwrite reg_param when specified
                optimizer = Partial(optimizer, weight_decay=exp_config.wd_param)
            else:
                optimizer = Partial(optimizer, weight_decay=reg_param)
        elif exp_config.wd_param:  # TODO: Maybe exclude adamw?
            opt_chain.append(optax.add_decayed_weights(weight_decay=exp_config.wd_param))  # !! Decayed weights are added before adam transformation --> equivalent to reg loss?
        if exp_config.optimizer == "adam_to_momentum":  # Setting transition steps to total # of steps
            optimizer = Partial(optimizer, transition_steps=exp_config.training_steps)
        if exp_config.gradient_clipping:
            opt_chain.append(optax.clip_by_global_norm(1.0))

        if 'noisy' in exp_config.optimizer:  # TODO: kill this
            opt_chain.append(optimizer(exp_config.lr, eta=exp_config.noise_eta,
                                       gamma=exp_config.noise_gamma))
        else:
            if isinstance(exp_config.lr_decay_steps, omegaconf.listconfig.ListConfig):  # TODO: This is dirty...
                decay_boundaries = [steps_per_epoch * lr_decay_step for lr_decay_step in exp_config.lr_decay_steps]
            else:
                total_steps = exp_config.training_steps+exp_config.pretrain
                decay_boundaries = [steps_per_epoch * exp_config.lr_decay_steps * (i+1) for i in range((total_steps//steps_per_epoch)//exp_config.lr_decay_steps)]
            lr_schedule = lr_scheduler_choice[exp_config.lr_schedule](exp_config.training_steps, exp_config.lr,
                                                                      exp_config.final_lr,
                                                                      decay_boundaries,
                                                                      exp_config.lr_decay_scaling_factor,
                                                                      **sched_config)
            opt_chain.append(optimizer(lr_schedule))
        opt = optax.chain(*opt_chain)

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=reg_param,
                                       classes=classes, with_dropout=with_dropout,
                                       exclude_bias_bn_from_reg=exp_config.masked_reg,
                                       label_smoothing=exp_config.label_smoothing)
        test_loss_fn = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=reg_param,
                                               classes=classes, is_training=False, with_dropout=with_dropout,
                                               exclude_bias_bn_from_reg=exp_config.masked_reg)
        accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
        death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
        # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
        #                                                  epsilon=exp_config.epsilon_close, avg=exp_config.avg_for_eps)
        scan_len = int(partial_train_ds_size // death_minibatch_size)
        scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
        # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
        final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
        full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, int(partial_train_ds_size // eval_size))

        if exp_config.measure_linear_perf:
            lin_accuracy_fn = utl.accuracy_given_model(lin_net, with_dropout=with_dropout)
            lin_full_accuracy_fn = utl.create_full_accuracy_fn(lin_accuracy_fn, test_size // eval_size)

        init_key = jax.random.PRNGKey(exp_config.init_seed)
        if load_from_preexisting_model_state:
            _architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
            exp_config.architecture]
            _architecture = _architecture(run_state['curr_starting_size'], classes, activation_fn=activation_fn, **net_config)
            _net, raw_net = build_models(*_architecture, with_dropout=with_dropout)
            params, state = _net.init(init_key, next(train))
            del _architecture
            del _net
        else:
            params, state = net.init(init_key, next(train))
        activation_layer_order = list(state.keys())
        # initial_params = utl.jax_deep_copy(params)  # Keep a copy of the initial params for relative change metric
        init_state = utl.jax_deep_copy(state)
        opt_state = opt.init(params)
        initial_params_count = utl.count_params(params)
        if load_from_preexisting_model_state:
            if exp_config.old_run and exp_config.resize_old:
                params, _, _ = utl.restore_all_pytree_states(run_state["model_dir"])
                params = utl.grow_neurons(params, exp_config.resize_old)
                run_state['model_dir'] = saving_dir
                load_from_preexisting_model_state = False
            else:
                params, state, opt_state = utl.restore_all_pytree_states(run_state["model_dir"])
        # frozen_layer_lists = utl.extract_layer_lists(params)
        neuron_states = utl.NeuronStates(activation_layer_order)
        reset_counter = None
        reset_tracker = None
        acti_map = utl.get_activation_mapping(raw_net, next(train))
        if exp_config.exclude_layer:
            for d_key, d_val in acti_map.items():
                if exp_config.exclude_layer in d_key:
                    acti_map[d_key]['preceding'] = None
                    acti_map[d_key]['following'] = None

        # print(acti_map)
        # raise SystemExit
        del raw_net
        if exp_config.record_gate_grad_stat:
            gate_grad = utl.NeuronStates(activation_layer_order)
        update_fn = get_updater(loss, opt, exp_config.add_noise, exp_config.noise_imp,
                                                        exp_config.asymmetric_noise,
                                                        live_only=exp_config.noise_live_only,
                                                        going_wild=exp_config.going_wild,
                                                        noise_offset_only=exp_config.noise_offset_only,
                                                        positive_offset=exp_config.positive_offset,
                                                        with_dropout=with_dropout,
                                                        modulate_via_gate_grad=exp_config.mod_via_gate_grad,
                                                        acti_map=acti_map, perturb=exp_config.perturb_param,
                                                        init_fn=init_fn)

        noise_key = jax.random.PRNGKey(exp_config.noise_seed)

        starting_neurons, starting_per_layer = utl.get_total_neurons(exp_config.architecture, size, exp_config.grok_depth)
        # total_neurons, total_per_layer = starting_neurons, starting_per_layer
        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes, exp_config.grok_depth)
        init_total_neurons = copy.copy(starting_neurons)
        init_total_per_layer = copy.copy(starting_per_layer)

        # Visualize NN with tabulate
        # print(hk.experimental.tabulate(net.init)(next(train)))
        # raise SystemExit

        decaying_reg_param = reg_param
        decay_cycles = exp_config.reg_param_decay_cycles + int(exp_config.zero_end_reg_param)
        if decay_cycles == 2:
            reg_param_decay_period = int(0.8 * exp_config.training_steps)
        else:
            reg_param_decay_period = exp_config.training_steps // decay_cycles

        if exp_config.reg_param_schedule:
            if exp_config.reg_param_span:
                sched_end = exp_config.reg_param_span
            elif exp_config.zero_end_reg_param:
                sched_end = int(0.9*exp_config.training_steps)
            else:
                sched_end = exp_config.training_steps
            reg_sched = reg_param_scheduler_choice[exp_config.reg_param_schedule](sched_end, reg_param)
            if exp_config.add_noise:
                noise_sched = reg_param_scheduler_choice[exp_config.reg_param_schedule](sched_end, exp_config.noise_eta)
        if exp_config.shifted_relu:
            # shift_relu_sched = utl.linear_warmup(exp_config.reg_param_span, exp_config.shifted_relu)
            shift_relu_sched = reg_param_scheduler_choice[exp_config.srelu_sched](exp_config.reg_param_span,
                                                                                  exp_config.shifted_relu)

        if exp_config.prune_at_end:
            pruning_reg_param, pruning_lr, add_steps_end = exp_config.prune_at_end
        else:
            add_steps_end = 0
        if exp_config.pretrain:
            add_steps_start = exp_config.pretrain
            pretrain_mask = {'_mask': utils.grok_utils.mask_untargeted_weights(params, exp_config.pretrain_targets)}
        else:
            add_steps_start = 0
            pretrain_mask = {}

        if load_from_preexisting_model_state:
            starting_step = run_state["training_step"]
            decaying_reg_param = run_state["decaying_reg_param"]
            dropout_key = run_state["dropout_key"]
            reset_counter = run_state["reset_counter"]
            reset_tracker = run_state["reset_tracker"]
            try:  # TODO Remove after completing all currently running from previous version
                best_acc = run_state["best_accuracy"]
                best_params_count = run_state["best_params_count"]
                best_total_neurons = run_state["best_total_neurons"]
                training_time = run_state["training_time"]
                dead_neurons_union = run_state["cumulative_dead_neurons"]
            except KeyError:
                best_acc = 0
                best_params_count = initial_params_count
                best_total_neurons = init_total_neurons
                training_time = 0
                dead_neurons_union = None
            load_from_preexisting_model_state = False
            if starting_step > exp_config.pretrain:
                pretrain_mask = {}
        else:
            starting_step = 0
            best_acc = 0
            best_params_count = initial_params_count
            best_total_neurons = init_total_neurons
            training_time = 0
            if not exp_config.dynamic_pruning:
                dead_neurons_union = death_check_fn(params, state, next(test_death))
            else:
                dead_neurons_union = None

        print(f"Continuing training from step {starting_step} and reg_param {reg_param}")
        training_timer = time.time()
        for step in range(starting_step, exp_config.training_steps + add_steps_end + add_steps_start):
            if step == add_steps_start:
                pretrain_mask = {}
                # Reset state and opt_state
                _, state = net.init(init_key, next(train))
                opt_state = ((opt.init(params)[0][:-1]+(opt_state[0][-1],)),)  # Reinitializing optimizer state, but keeping lr_scheduler counter
                if exp_config.sigm_pretrain or exp_config.tanh_pretrain:
                    _, sigm_params = utils.grok_utils.split_norm_layers(params)
                    if exp_config.sigm_pretrain:
                        sigm_params = jax.tree_map(jax.nn.sigmoid, sigm_params)
                    elif exp_config.tanh_pretrain:
                        sigm_params = jax.tree_map(jax.nn.tanh, sigm_params)
                    params.update(sigm_params)
                    net_config['bn_config']['sigm_scale'] = False
                    architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                        exp_config.architecture]
                    architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
                    net = build_models(*architecture)[0]
                    init_fn = utl.get_init_fn(net, ones_init)
                    total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes,
                                                                           exp_config.grok_depth)

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
                    update_fn = get_updater(loss, opt, exp_config.add_noise,
                                            exp_config.noise_imp, exp_config.asymmetric_noise,
                                            going_wild=exp_config.going_wild,
                                            live_only=exp_config.noise_live_only,
                                            noise_offset_only=exp_config.noise_offset_only,
                                            positive_offset=exp_config.positive_offset,
                                            with_dropout=with_dropout,
                                            modulate_via_gate_grad=exp_config.mod_via_gate_grad,
                                            acti_map=acti_map, perturb=exp_config.perturb_param,
                                            init_fn=init_fn)
                    death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout)
                    # eps_death_check_fn = utl.death_check_given_model(net, with_dropout=with_dropout,
                    #                                                  epsilon=exp_config.epsilon_close,
                    #                                                  avg=exp_config.avg_for_eps)
                    # eps_scan_death_check_fn = utl.scanned_death_check_fn(eps_death_check_fn, scan_len)
                    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
                    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))
                    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn,
                                                                    int(partial_train_ds_size // eval_size))
            if (step > 0) and (step % steps_per_epoch == 0):  # Keep track of the best accuracy along training
                architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[  # TODO: those line should not be necessary ...
                    exp_config.architecture]
                architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
                net = build_models(*architecture)[0]
                accuracy_fn = utl.accuracy_given_model(net, with_dropout=with_dropout)
                final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, int(test_size // eval_size))

                curr_acc = final_accuracy_fn(params, state, test_eval)
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    best_params_count = utl.count_params(params)
                    best_total_neurons = utl.get_total_neurons(exp_config.architecture, new_sizes, exp_config.grok_depth)[0]
            # Update decaying reg_param if needed:
            if exp_config.reg_param_schedule and step < exp_config.training_steps:
                decaying_reg_param = reg_sched(step)
            # Checkpoint
            if (step > 0) and exp_config.preempt_handling and (step % (exp_config.checkpoint_freq * steps_per_epoch) == 0):
                print(
                    f"Elapsed time in current run at step {step}: {timedelta(seconds=time.time() - subrun_start_time)}")
                chckpt_init_time = time.time()
                training_time += time.time() - training_timer
                utl.checkpoint_exp(run_state, params, state, opt_state, curr_epoch=step//steps_per_epoch,
                                   curr_step=step, curr_arch_sizes=new_sizes, curr_starting_size=size,
                                   curr_reg_param=reg_param, dropout_key=dropout_key,
                                   decaying_reg_param=decaying_reg_param, best_acc=best_acc,
                                   best_params_count=best_params_count, best_total_neurons=best_total_neurons,
                                   training_time=training_time, reset_counter=reset_counter,
                                   reset_tracker=reset_tracker, dead_neurons_union=dead_neurons_union)
                training_timer = time.time()
                print(
                    f"Checkpointing performed in: {timedelta(seconds=time.time() - chckpt_init_time)}")
            # Record metrics and prune model if needed:
            (decaying_reg_param, net, new_sizes, params, state, opt_state, opt, total_neurons, total_per_layer, loss,
             test_loss_fn, accuracy_fn, death_check_fn, scan_death_check_fn, full_train_acc_fn, final_accuracy_fn,
             update_fn, dead_neurons_union, pretrain_mask, init_fn, init_key, reset_counter,
             reset_tracker) = record_metrics_and_prune(step, reg_param,
                                                       activation_fn,
                                                       decaying_reg_param, net,
                                                       new_sizes,
                                                       params,
                                                       state, opt_state, opt,
                                                       total_neurons,
                                                       total_per_layer, loss,
                                                       test_loss_fn,
                                                       accuracy_fn, death_check_fn,
                                                       scan_death_check_fn,
                                                       full_train_acc_fn,
                                                       final_accuracy_fn, update_fn,
                                                       dead_neurons_union,
                                                       pretrain_mask,
                                                       init_fn,
                                                       init_key,
                                                       reset_counter,
                                                       reset_tracker)  # Ugly, but cache is cleared
            if (step % exp_config.pruning_freq == 0) and exp_config.dynamic_pruning:
                # jax.clear_backends()
                gc.collect()
            # Train step over single or accumulated batch
            if exp_config.accumulate_batches > 1:
                next_batches = train
            else:
                next_batches = next(train)
            if with_dropout or exp_config.perturb_param:
                params, state, opt_state, dropout_key = update_fn(params, state, opt_state, next_batches,
                                                                  dropout_key,
                                                                  _reg_param=decaying_reg_param,
                                                                  **pretrain_mask)
            else:
                if not exp_config.add_noise:
                    params, state, opt_state = update_fn(params, state, opt_state, next_batches,
                                                         _reg_param=decaying_reg_param,
                                                         **pretrain_mask)
                else:
                    # noise_var = exp_config.noise_eta / ((1 + step) ** exp_config.noise_gamma)
                    # noise_var = exp_config.lr * noise_var  # Apply lr for consistency with update size
                    noise_var = noise_sched(step)
                    params, state, opt_state, noise_key = update_fn(params, state, opt_state, next_batches,
                                                                    noise_var,
                                                                    noise_key, _reg_param=decaying_reg_param,
                                                                    **pretrain_mask)
            if exp_config.shifted_relu:
                state = utl.update_gate_constant(state, shift_relu_sched(step))
            if step <= exp_config.pretrain and exp_config.clip_norm:
                params = utils.grok_utils.clip_norm_params(params, *exp_config.clip_norm)

        if exp_config.record_distribution_data:
            scan_death_check_fn_with_activations_data = utl.scanned_death_check_fn(
                utl.death_check_given_model(net, with_activations=True, with_dropout=with_dropout), scan_len, True)
            activations_data, final_dead_neurons = scan_death_check_fn_with_activations_data(params, state,
                                                                                             test_death)
        else:
            final_dead_neurons = scan_death_check_fn(params, state, test_death)

        neuron_states.update_from_ordered_list(final_dead_neurons)
        _params, _opt_state, _state, new_sizes = utl.prune_params_state_optstate(params, acti_map,
                                                                              neuron_states, opt_state,
                                                                              state)  # Final pruning before eval
        final_params_count = utl.count_params(_params)

        del final_dead_neurons  # Freeing memory
        if exp_config.activation == "relu" or exp_config.dynamic_pruning:  # Only performs end-of-training actual pruning if activation is relu
            params, opt_state, state = _params, _opt_state, _state
            architecture = pick_architecture(with_dropout=with_dropout, with_bn=exp_config.with_bn)[
                exp_config.architecture]
            architecture = architecture(new_sizes, classes, activation_fn=activation_fn, **net_config)
            net = build_models(*architecture)[0]
            total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, new_sizes, exp_config.grok_depth)

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

        log_step = reg_param * 1e8

        exp_run.track(avg_final_live_neurons,
                      name="On average, live neurons after convergence w/r reg param", step=log_step)
        exp_run.track(avg_final_live_neurons / init_total_neurons,
                      name="Average live neurons ratio after convergence w/r reg param", step=log_step)
        total_live_neurons = total_neurons  # - final_dead_neurons_count
        exp_run.track(total_live_neurons,
                      name="Live neurons after convergence w/r reg param", step=log_step)
        exp_run.track(total_live_neurons / init_total_neurons,
                      name="Live neurons ratio after convergence w/r reg param", step=log_step)
        exp_run.track(reg_param,  # Logging true reg_param value to display with aim metrics
                      name="Reg param w/r reg param", step=log_step)
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

                exp_run.track(eps_avg_final_live_neurons,
                              name="On average, quasi-live neurons after convergence w/r reg param",
                              step=log_step, context={"epsilon": eps})
                exp_run.track(eps_avg_final_live_neurons / init_total_neurons,
                              name="Average quasi-live neurons ratio after convergence w/r reg param",
                              step=log_step, context={"epsilon": eps})
                eps_total_live_neurons = total_neurons - eps_final_dead_neurons_count
                exp_run.track(eps_total_live_neurons,
                              name="Quasi-live neurons after convergence w/r reg param", step=log_step,
                              context={"epsilon": eps})
                exp_run.track(eps_total_live_neurons / init_total_neurons,
                              name="Quasi-live neurons ratio after convergence w/r reg param",
                              step=log_step, context={"epsilon": eps})

        for i, live_in_layer in enumerate(total_per_layer):
            total_neuron_in_layer = init_total_per_layer[i]
            # live_in_layer = total_neuron_in_layer - layer_dead
            exp_run.track(live_in_layer,
                          name=f"Live neurons in layer {i} after convergence w/r reg param",
                          step=log_step)
            exp_run.track(live_in_layer / total_neuron_in_layer,
                          name=f"Live neurons ratio in layer {i} after convergence w/r reg param",
                          step=log_step)
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r reg param", step=log_step)
        exp_run.track(final_train_acc,
                      name="Train accuracy after convergence w/r reg param", step=log_step)
        log_sparsity_step = jax.device_get(total_live_neurons / init_total_neurons) * 1000
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r percent*10 of neurons remaining", step=log_sparsity_step)
        training_time += time.time() - training_timer
        exp_run.track(training_time/60,
                      name="Training time (min.) w/r percent*10 of neurons remaining", step=log_sparsity_step)
        log_params_sparsity_step = final_params_count / initial_params_count * 1000
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r percent*10 of params remaining",
                      step=log_params_sparsity_step)
        exp_run.track(training_time/60,
                      name="Training time (min.) w/r percent*10 of params remaining",
                      step=log_params_sparsity_step)
        log_sparsity_step = jax.device_get(best_total_neurons / init_total_neurons) * 1000
        exp_run.track(best_acc,
                      name="Best accuracy after convergence w/r percent*10 of neurons remaining", step=log_sparsity_step)
        log_params_sparsity_step = best_params_count / initial_params_count * 1000
        exp_run.track(best_acc,
                      name="Best accuracy after convergence w/r percent*10 of params remaining",
                      step=log_params_sparsity_step)
        # if not exp_config.dynamic_pruning:  # Cannot take norm between initial and pruned params
        #     params_vec, _ = ravel_pytree(params)
        #     initial_params_vec, _ = ravel_pytree(initial_params)
        #     exp_run.track(
        #         jax.device_get(jnp.linalg.norm(params_vec - initial_params_vec) / jnp.linalg.norm(initial_params_vec)),
        #         name="Relative change in norm of weights from init after convergence w/r reg param",
        #         step=log_step)

        if 'grok' in exp_config.architecture:
            _params = utils.grok_utils.mask_ff_init_layer(params, 'wb')
            masked_accuracy = final_accuracy_fn(_params, state, test_eval)
            exp_run.track(masked_accuracy,
                          name="Accuracy, final, with masking on w and b of init layer", step=0)
            _params = utils.grok_utils.mask_ff_init_layer(params, 'w')
            masked_accuracy = final_accuracy_fn(_params, state, test_eval)
            exp_run.track(masked_accuracy,
                          name="Accuracy, final, with masking on w only of init layer", step=0)
            _params = utils.grok_utils.mask_ff_last_layer(params, 'wb')
            masked_accuracy = final_accuracy_fn(_params, state, test_eval)
            exp_run.track(masked_accuracy,
                          name="Accuracy, final, with masking on w and b of second layer", step=0)
            _params = utils.grok_utils.mask_ff_last_layer(params, 'w')
            masked_accuracy = final_accuracy_fn(_params, state, test_eval)
            exp_run.track(masked_accuracy,
                          name="Accuracy, final, with masking on w only of second layer", step=0)

        if exp_config.record_distribution_data:
            activations_max, activations_mean, activations_count, _ = activations_data
            if exp_config.save_wanda:
                activations_meta.maximum[reg_param] = activations_max
                activations_meta.mean[reg_param] = activations_mean
                activations_meta.count[reg_param] = activations_count
            activations_max, _ = ravel_pytree(activations_max)
            # activations_max = jax.device_get(activations_max)
            activations_mean, _ = ravel_pytree(activations_mean)
            # activations_mean = jax.device_get(activations_mean)
            activations_count, _ = ravel_pytree(activations_count)
            # activations_count = jax.device_get(activations_count)
            activations_max_dist = Distribution(activations_max, bin_count=100)
            exp_run.track(activations_max_dist, name='Maximum activation distribution after convergence', step=0,
                          context={"reg param": utl.size_to_string(reg_param)})
            activations_mean_dist = Distribution(activations_mean, bin_count=100)
            exp_run.track(activations_mean_dist, name='Mean activation distribution after convergence', step=0,
                          context={"reg param": utl.size_to_string(reg_param)})
            activations_count_dist = Distribution(activations_count, bin_count=50)
            exp_run.track(activations_count_dist, name='Activation count per neuron after convergence', step=0,
                          context={"reg param": utl.size_to_string(reg_param)})

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
            with open(pickle_dir_path + 'activations_meta.p', 'wb') as fp:
                pickle.dump(asdict(activations_meta), fp)  # Update by overwrite

            if not exp_config.save_act_only:
                # Pickling the final parameters value as well
                params_meta.parameters.append(params)
                with open(pickle_dir_path + 'params_meta.p', 'wb') as fp:
                    pickle.dump(asdict(params_meta), fp)  # Update by overwrite

    if exp_config.preempt_handling:
        beginning_index = exp_config.reg_params.index(run_state["curr_reg_param"])
        reg_params_iterate = exp_config.reg_params[beginning_index:]
    else:
        reg_params_iterate = exp_config.reg_params
    for reg_param in reg_params_iterate:  # Vary the regularizer parameter to measure impact on overfitting
        # Time the subrun for the different sizes
        subrun_start_time = time.time()
        # jax.clear_backends()
        gc.collect()

        train_run(reg_param)

        # Print running time
        print()
        print(f"Running time for reg_param {reg_param}: " + str(timedelta(seconds=time.time() - subrun_start_time)))
        print("----------------------------------------------")
        print()

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
