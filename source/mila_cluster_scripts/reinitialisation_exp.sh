#!/bin/bash

#SBATCH --job-name=reinitialisation_exp
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=12G                                        # Ask for 10 GB of RAM
#SBATCH --time=1:00:00                                  # The job will run for 1 hour

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
python source/reinitialisation_exp.py dataset='fashion mnist' regularizer=None
wait $!
