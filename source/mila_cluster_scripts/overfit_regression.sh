#!/bin/bash

#SBATCH --job-name=overfit_regression
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=8G                                        # Ask for 10 GB of RAM
#SBATCH --time=0:35:00                                   # The job will run for 2.5 hours

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
#    record_freq: int = 100
#    pruning_freq: int = 2000
#    drawing_freq: int = 20000
#    final_smoothing: int = 0  # Remove noise for n final smoothing steps
#    lr: float = 1e-3
#    lr_schedule: str = "None"
#    final_lr: float = 1e-6
#    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
#    train_batch_size: int = 12
#    eval_batch_size: int = 100
#    death_batch_size: int = 12
#    optimizer: str = "adam"
#    activation: str = "relu"  # Activation function used throughout the model
#    dataset_size: int = 12  # We want to keep it small to allow overfitting
#    eval_dataset_size: int = 100  # Evaluate on more points
#    dataset_seed: int = 1234  # Random seed to vary the training samples picked
#    noise_std: float = 1.0  # std deviation of the normal distribution (mean=0) added to training data
#    architecture: str = "mlp_3_reg"
#    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
#    with_bn: bool = False  # Add batchnorm layers or not in the models
#    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
#    size: Any = 50
#    regularizer: Optional[str] = "None"
#    reg_param: float = 1e-4
#    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
#    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
#    init_seed: int = 41
#    with_rng_seed: int = 428
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

# Run experiments

python source/overfit_regression.py final_smoothing=20000 size=50 noise_std=0.0
wait $!