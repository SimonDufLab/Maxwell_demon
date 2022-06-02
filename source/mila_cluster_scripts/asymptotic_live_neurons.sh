#!/bin/bash

#SBATCH --job-name=asymptotic_live_neurons
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=12G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:15:00                                   # The job will run for 2.5 hours

# Make sure we are located in the right directory and on right branch
cd ~/repositories/Maxwell_demon || exit
git checkout experiments-logs

# Load required modules
module load python/3.7
module load cuda/11.2/cudnn/8.1

# Load venv
source venv/bin/activate

# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Default configuration:
#    training_steps: int = 120001
#    report_freq: int = 3000
#    lr: float = 1e-3
#    optimizer: str = "adam"
#    dataset: str = "mnist"
#    regularizer: Optional[str] = "cdg_l2"
#    reg_param: float = 1e-4
#    epsilon_close: float = 0.0  # Relaxing criterion for dead neurons, epsilon-close to relu gate

# Run experiments
python source/asymptotic_live_neurons.py dataset='mnist' regularizer='l2'
wait $!
python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer='l2'
wait $!
python source/asymptotic_live_neurons.py dataset='cifar10' regularizer='l2'
wait $!
python source/asymptotic_live_neurons.py dataset='mnist' regularizer='cdg_l2'
wait $!
python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer='cdg_l2'
wait $!
python source/asymptotic_live_neurons.py dataset='cifar10' regularizer='cdg_l2'