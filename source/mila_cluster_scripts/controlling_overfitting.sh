#!/bin/bash

#SBATCH --job-name=controlling_overfitting
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=24G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=4:30:00 #36:00:00 #around 8 for Resnet                                  # The job will run for 2.5 hours
#SBATCH -x 'cn-d[001-004], cn-g[005-012,017-026]'  # Excluding DGX system, will require a jaxlib update

# Make sure we are located in the right directory and on right branch
cd ~/repositories/Maxwell_demon || exit
git checkout exp-config

# Load required modules
module load python/3.8
module load cuda/11.2/cudnn/8.1

# Load venv
source venv/bin/activate

# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Default configuration:
#    training_steps: int = 120001
#    report_freq: int = 3000
#    record_freq: int = 100
#    pruning_freq: int = 2000
#    live_freq: int = 20000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
#    lr: float = 1e-3
#    lr_schedule: str = "None"
#    final_lr: float = 1e-6
#    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
#    train_batch_size: int = 128
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    optimizer: str = "adam"
#    activation: str = "relu"  # Activation function used throughout the model
#    dataset: str = "mnist"
#    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
#    noisy_label: float = 0.25  # ratio (between [0,1]) of labels to randomly (uniformly) flip
#    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
#    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
#    architecture: str = "mlp_3"
#    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
#    with_bn: bool = False  # Add batchnorm layers or not in the models
#    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
#    size: Any = 50  # Can also be a tuple for convnets
#    regularizer: Optional[str] = "cdg_l2"
#    reg_params: Any = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
#    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
#    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
#    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
#    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
#    init_seed: int = 41
#    dynamic_pruning: bool = False
#    add_noise: bool = False  # Add Gaussian noise to the gradient signal
#    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones
#    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
#    noise_eta: float = 0.01
#    noise_gamma: float = 0.0
#    noise_seed: int = 1
#    dropout_rate: float = 0
#    with_rng_seed: int = 428
#    linear_switch: bool = False  # Whether to switch mid-training steps to linear activations
#    measure_linear_perf: bool = False  # Measure performance over the linear network without changing activation
#    save_wanda: bool = False  # Whether to save weights and activations value or not
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

# Run experiments

#python source/controlling_overfitting.py dataset='mnist' architecture='mlp_3' 'size="(300, 100)"' lr_schedule=piecewise_constant info=LeNet_acc_vs_sparsity train_batch_size=64 lr=1e-3 dynamic_pruning=True prune_after=25000 init_seed=72 noisy_label=0.0
#wait $!

#python source/controlling_overfitting.py dataset='mnist' architecture='mlp_3' 'size="(300, 100)"' lr_schedule=piecewise_constant info=LeNet_acc_vs_sparsity 'reg_params="(1.0, 1.2, 1.4, 1.6, 1.8, 2.0)"' train_batch_size=64 lr=1e-3 dynamic_pruning=True prune_after=25000 init_seed=72 noisy_label=0.0
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='conv_4_2' 'size="(128, 64)"' lr_schedule=piecewise_constant reg_param_decay_cycles=4 zero_end_reg_param=True info=control_capacity_l2_conv_4_2 'reg_params="(0.0, 0.001, 0.005, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)"' optimizer=momentum9 lr=0.001 lr_decay_steps=4 init_seed=72 noisy_label=0.25
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='conv_4_2' 'size="(128, 64)"' lr_schedule=piecewise_constant reg_param_decay_cycles=4 zero_end_reg_param=True info=control_capacity_l2_conv_4_2 'reg_params="(0.14, 0.16, 0.20, 0.25, 0.3, 0.4)"' optimizer=adam lr=0.0001 lr_decay_steps=4 init_seed=72 noisy_label=0.25
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=250001 report_freq=1000 record_freq=500 pruning_freq=2000 live_freq=50000 size=64 with_bn=True lr_schedule=piecewise_constant reg_param_decay_cycles=4 zero_end_reg_param=True info=control_capacity_l2_resnet18_vs_lr_bs 'reg_params="(0.0, 0.00001, 0.00005, 0.0001, 0.0005)"' optimizer=adam lr=0.005 lr_decay_steps=4 train_batch_size=64 augment_dataset=True init_seed=72 noisy_label=0.2 regularizer=l2
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=10000 record_freq=1000 pruning_freq=10000 live_freq=5000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Resnet18_acc_vs_sparsity_cdg_l2_short 'reg_params="(0.0, 0.000001, 0.000005,  0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005)"' optimizer=adam lr=0.05 train_batch_size=256 augment_dataset=True gradient_clipping=True init_seed=71 noisy_label=0.0 regularizer=cdg_l2
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=250001 report_freq=10000 record_freq=1000 pruning_freq=10000 live_freq=250000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Resnet18_acc_vs_sparsity_cdg_l2 'reg_params="(0.008, 0.01, 0.03, 0.05, 0.1, 0.5)"' optimizer=adam lr=0.002 train_batch_size=64 augment_dataset=True gradient_clipping=True init_seed=72 noisy_label=0.0 regularizer=cdg_l2
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=30001 report_freq=1000 record_freq=250 pruning_freq=2000 live_freq=20000 size=64 with_bn=True lr_schedule=fix_steps normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_momentum9_dense 'reg_params="(0.0,)"' optimizer=momentum9 wd_param=0.0005 lr=0.01 train_batch_size=128 augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=False "lr_decay_steps='(20000, 25000)'" init_seed=66
#wait $!

# current main
#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=4 info=Resnet19_dyn_pruning_mom_base_new_opt_full_len 'reg_params="(0.0, )"' optimizer=adam_to_momentum wd_param=0.00001 lr=1.0 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True init_seed=62
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=4 info=Resnet19_dyn_pruning_mom_base_new_opt_full_len 'reg_params="(0.00001, 0.00005, 0.0001)"' optimizer=adam_to_momentum wd_param=0.00001 lr=1.0 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True init_seed=62
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=4 info=Resnet19_dyn_pruning_mom_base_new_opt_full_len 'reg_params="(0.0005, 0.001, 0.005)"' optimizer=adam_to_momentum wd_param=0.00001 lr=1.0 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True init_seed=62
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_dyn_pruning_one_cycle_decay_mom_base 'reg_params="(0.0006, 0.0007, 0.0008, 0.0009)"' optimizer=momentum9 wd_param=0.00001 lr=1.0 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=62
#wait $!

# current main
#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=4 info=Resnet19_dyn_pruning_mom_base_new_opt_full_len 'reg_params="(0.01, 0.05, 0.1)"' optimizer=adam_to_momentum wd_param=0.00001 lr=1.0 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True init_seed=62
#wait $!

#########################
#Structured RigL setup:

#python source/controlling_overfitting.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=4 info=Resnet18_srigl_final_adam 'reg_params="(0.0,)"' optimizer=adam wd_param=0.0 lr=0.005 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True preempt_handling=True checkpoint_freq=5 init_seed=61
#wait $!

# Adam only
#python source/controlling_overfitting.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=4 info=Resnet18_srigl_final_adam 'reg_params="(0.0000005, 0.000001, 0.000005)"' optimizer=adam wd_param=0.0 lr=0.005 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=cdg_l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True preempt_handling=True checkpoint_freq=5 init_seed=65 reg_param_schedule=one_cycle
#wait $!

#python source/controlling_overfitting.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=4 info=Resnet18_srigl_final_adam 'reg_params="(0.00001, 0.00005, 0.0001)"' optimizer=adam wd_param=0.0 lr=0.005 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=cdg_l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True preempt_handling=True checkpoint_freq=5 init_seed=65 reg_param_schedule=one_cycle
#wait $!

#python source/controlling_overfitting.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=4 info=Resnet18_srigl_final_adam 'reg_params="(0.0005, 0.001, 0.005)"' optimizer=adam wd_param=0.0 lr=0.005 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=cdg_l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True preempt_handling=True checkpoint_freq=5 init_seed=65 reg_param_schedule=one_cycle
#wait $!

python source/controlling_overfitting.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=4 info=Resnet18_srigl_final_adam 'reg_params="(0.006, 0.007, 0.008)"' optimizer=adam wd_param=0.0 lr=0.005 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=cdg_l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True preempt_handling=True checkpoint_freq=5 init_seed=65 reg_param_schedule=one_cycle
wait $!

#Momentum only
#python source/controlling_overfitting.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=4 info=Resnet18_srig_final_momentum 'reg_params="(0.01, 0.05, 0.1)"' optimizer=momentum9 wd_param=0.0 lr=0.1 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True preempt_handling=True checkpoint_freq=5 init_seed=65 reg_param_schedule=one_cycle
#wait $!

########################

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=64 with_bn=False lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_l2_acti_distribution 'reg_params="(0.0, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)"' optimizer=adam wd_param=0.0005 lr=0.0001 train_batch_size=256 augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=cdg_l2 activation=relu zero_end_reg_param=False save_wanda=True dynamic_pruning=False init_seed=21
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Resnet18_acc_vs_sparsity_testing_res19_leaky 'reg_params="(0.05, 0.08, 0.1, 0.5, 1.0)"' optimizer=adamw lr=0.002 train_batch_size=256 augment_dataset=True gradient_clipping=True init_seed=75 noisy_label=0.0 regularizer=None activation=relu
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Resnet18_acc_vs_sparsity_res19_prune_at_end_momentum 'reg_params="(0.0, 0.0005)"' optimizer=adamw lr=0.002 train_batch_size=256 augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None activation=relu "prune_at_end='(0.0001, 0.002, 15626)'" pruning_reg=cdg_l2 init_seed=96 pruning_opt=adamw
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='resnet18' training_steps=150000 report_freq=10000 record_freq=500 pruning_freq=10000 live_freq=25000 size=64 with_bn=True lr_schedule=piecewise_constant normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Resnet19_sgd_momentum 'reg_params="(0.0,)"' optimizer=sgd augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None activation=relu wd_param=0.0001 lr=0.1 train_batch_size=256 init_seed=26
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='vgg16' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 "size='(64, 4096)'" with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=vgg16_dense 'reg_params="(0.0,)"' optimizer=adam wd_param=0.0005 lr=0.05 train_batch_size=256 augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None activation=relu save_wanda=False dynamic_pruning=False init_seed=61
#wait $!

# vgg mains
#python source/controlling_overfitting.py dataset='cifar10' architecture='vgg16' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 "size='(64, 4096)'" with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=2 zero_end_reg_param=True info=vgg16_l2_dyn 'reg_params="(0.0, 0.000001, 0.000005, 0.00001, 0.00005)"' optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu save_wanda=False dynamic_pruning=True init_seed=79
#wait $!

#python source/controlling_overfitting.py dataset='cifar10' architecture='vgg16' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 "size='(64, 4096)'" with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=2 zero_end_reg_param=True info=vgg16_l2_dyn 'reg_params="(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)"' optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu save_wanda=False dynamic_pruning=True init_seed=79
#wait $!

#python source/controlling_overfitting.py dataset='mnist' architecture='mlp_3' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=100 with_bn=False normalize_inputs=False info=mlp3_l2_new_reg_decay reg_param_decay_cycles=2 zero_end_reg_param=True optimizer=adam train_batch_size=256 noisy_label=0.0 activation=relu 'reg_params="(0.0, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)"' lr=0.01 regularizer=l2  init_seed=41
#wait $!
