#!/bin/bash

#SBATCH --job-name=prune_reinit
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=15:00:00                                  # The job will run for 24 hours

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
#    training_steps: int = 120001
#    report_freq: int = 3000
#    record_freq: int = 100
#    full_ds_eval_freq: int = 2000
#    pruning_cycles: int = 1
#    lr: float = 1e-3
#    end_lr: float = 1e-5  # lr used during low noise evaluation
#    lr_schedule: str = "None"
#    final_lr: float = 1e-6
#    end_final_lr: float = 1e-6  # final lr used for scheduler during low noise evaluation
#    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
#    train_batch_size: int = 16
#    end_train_batch_size: int = 512  # final training batch size used during low noise evaluation
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    optimizer: str = "adam"
#    activation: str = "relu"  # Activation function used throughout the model
#    dataset: str = "mnist"
#    architecture: str = "mlp_3"
#    size: Any = 50
#    regularizer: Optional[str] = 'None'
#    reg_param: float = 1e-4
#    init_seed: int = 41
#    dynamic_pruning: bool = False
#    dropout_rate: float = 0
#    with_rng_seed: int = 428
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

# Run experiments

#python source/prune_reinit.py training_steps=250001 optimizer=adam architecture=mlp_3 report_freq=2500 record_freq=250 full_ds_eval_freq=1000 lr_schedule=piecewise_constant info=mnist_mlp_decay end_lr=0.001 size=500 train_batch_size=8 pruning_cycles=18 init_seed=13
#wait $!

python source/prune_reinit.py training_steps=250001 optimizer=adam architecture=conv_4_2 dataset='fashion mnist' report_freq=2500 record_freq=250 full_ds_eval_freq=1000 lr_schedule=piecewise_constant info=fmnist_conv_4_2_decay_cycles end_lr=0.001 'size="(128, 512)"' train_batch_size=8 pruning_cycles=18 init_seed=11
wait $!
