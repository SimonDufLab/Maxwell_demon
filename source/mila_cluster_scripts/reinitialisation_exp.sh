#!/bin/bash

#SBATCH --job-name=reinitialisation_exp
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=8:00:00                                  # The job will run for 1 hour

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
#    size: int = 100  # Number of hidden units in first layer; size*3 in second hidden layer
#    total_steps: int = 20001
#    report_freq: int = 500
#    record_freq: int = 10
#    switching_period: int = 2000  # Switch dataset periodically
#    reset_period: int = 500  # After reset_period steps, reinitialize the parameters
#    reset_horizon: float = 1.0  # Set to lower than one if you want to stop resetting before final steps
#    kept_classes: int = 3  # Number of classes in the randomly selected subset
#    compare_full_reset: bool = True  # Include the comparison with a complete reset of the parameters
#    lr: float = 1e-3
#    optimizer: str = "adam"
#    dataset: str = "mnist"
#    regularizer: Optional[str] = "cdg_l2"
#    reg_param: float = 1e-4

# Run experiments
#python source/reinitialisation_exp.py dataset='cifar100' total_steps=400001 compare_to_reset=False kept_classes=10 architecture='conv_3_2' 'size="(32, 100)"' activation='relu' reset_horizon=0.75 norm_grad=False with_bias=False
#wait $!
#python source/reinitialisation_exp.py dataset='cifar100' total_steps=400001 compare_to_reset=False kept_classes=10 activation='relu' with_bias=False
#wait $!

#python source/reinitialisation_exp.py dataset=mnist total_steps=100001 report_freq=1000 switching_period=5000 compare_to_reset=False architecture=conv_4_2_ln activation=threlu with_bias=False reduce_head=False mask_head=True info=funky_CL_with_LN 'size="(512, 768)"'
#wait $!

#python source/reinitialisation_exp.py total_steps=300001 report_freq=3000 switching_period=60000 kept_classes=2 train_batch_size=16 dataset=cifar10 lr=0.01 lr_schedule=piecewise_constant reduce_head=False mask_head=True regularizer=cdg_l2 freeze_and_reinit=True architecture=conv_4_2 'size="(128, 512)"' reg_param=0.01 optimizer=momentum9 sequential_classes=True reg_param_decay_cycles=4 info=freeze_and_reinit_reduce_head_gap_conv_4_2_cifar10 init_seed=41 reduce_head_gap=True tanh_head=False
#wait $!

python source/reinitialisation_exp.py total_steps=300001 report_freq=3000 switching_period=60000 kept_classes=2 train_batch_size=16 dataset=cifar10 lr=0.01 lr_schedule=piecewise_constant reduce_head=False mask_head=True regularizer=cdg_l2 freeze_and_reinit=True architecture=conv_4_2 'size="(128, 512)"' reg_param=0.01 optimizer=momentum9 sequential_classes=True reg_param_decay_cycles=4 info=freeze_and_reinit_tanh_head_conv_4_2_cifar10 init_seed=41 reduce_head_gap=False tanh_head=True
wait $!
