#!/bin/bash

#SBATCH --job-name=structured_baseline
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=24G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00 #around 15min with onecycle                             # The job will run for 2.5 hours

# Make sure we are located in the right directory and on right branch
cd ~/repositories/Maxwell_demon || exit
git checkout exp-config

# Load required modules
module load python/3.8
module load cuda/11.2/cudnn/8.1

# Load venv
source venv/bin/activate

# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Default configuration:
#    training_steps: int = 250001
#    report_freq: int = 2500
#    record_freq: int = 250
#    full_ds_eval_freq: int = 1000
#    live_freq: int = 25000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
#    lr: float = 1e-3
#    gradient_clipping: bool = False
#    lr_schedule: str = "None"
#    final_lr: float = 1e-6
#    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
#    train_batch_size: int = 512
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    optimizer: str = "adamw"
#    activation: str = "relu"  # Activation function used throughout the model
#    dataset: str = "mnist"
#    normalize_inputs: bool = False  # Substract mean across channels from inputs and divide by variance
#    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
#    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
#    noisy_label: float = 0.0  # ratio (between [0,1]) of labels to randomly (uniformly) flip
#    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
#    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
#    architecture: str = "mlp_3"
#    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
#    with_bn: bool = False  # Add batchnorm layers or not in the models
#    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
#    size: Any = 50  # Can also be a tuple for convnets
#    regularizer: Optional[str] = "None"
#    reg_param: float = 5e-4
#    wd_param: Optional[float] = None
#    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
#    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
#    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
#    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
#    pruning_criterion: Optional[str] = None
#    pruning_density: float = 0.0
#    pruning_args: Any = None
#    init_seed: int = 41
#    dropout_rate: float = 0
#    with_rng_seed: int = 428
#    save_wanda: bool = False  # Whether to save weights and activations value or not
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

python source/structured_baseline.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 full_ds_eval_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True info=Resnet19_earlycrop_naive_ss_pruning pruning_criterion=earlycrop optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None activation=relu save_wanda=False pruning_density=0.5 init_seed=62
wait $!