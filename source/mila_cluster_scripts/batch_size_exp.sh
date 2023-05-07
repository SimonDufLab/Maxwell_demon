#!/bin/bash

#SBATCH --job-name=batch_size_exp
#SBATCH --partition=long                          # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                                 # The job will run for 2.5 hours

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
#    training_steps: int = 50001
#    report_freq: int = 3000
#    record_freq: int = 100
#    full_ds_sweep_freq: int = 5000
#    lr: float = 1e-3
#    batch_size_seq: Any = (1, 32, 128, 512, 2048, 8192, 32768, "full")
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    optimizer: str = "adam"
#    dataset: str = "mnist"
#    classes: int = 10  # Number of classes in the training dataset
#    architecture: str = "mlp_3"
#    size: Any = 100
#    regularizer: Optional[str] = 'None'
#    reg_param: float = 1e-4
#    init_seed: int = 41

# Run experiments
python source/batch_size_exp.py dataset='mnist' regularizer=None architecture='mlp_3'
wait $!
#python source/batch_size_exp.py dataset='cifar10' regularizer=None architecture='mlp_3'
#wait $!

#python source/batch_size_exp.py dataset='mnist' regularizer=None architecture='conv_3_2' 'size="(32, 128)"'
#wait $!
#python source/batch_size_exp.py dataset='cifar10' regularizer=None architecture='conv_3_2' 'size="(32, 128)"'
#wait $!