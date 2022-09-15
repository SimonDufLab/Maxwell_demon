#!/bin/bash

#SBATCH --job-name=batchnorm_exp
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                                   # The job will run for 2.5 hours

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
#    snapshot_freq: int = 20_000
#    lr: float = 1e-3
#    train_batch_size: int = 128
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    optimizer: str = "adam"
#    dataset: str = "mnist"
#    architecture: str = "mlp_3"
#    size: Any = 50
#    regularizer: Optional[str] = "None"
#    reg_param: float = 1e-4
#    epsilon_close: float = 0.0  # Relaxing criterion for dead neurons, epsilon-close to relu gate
#    init_seed: int = 41
#    add_noise: bool = False  # Add Gaussian noise to the gradient signal
#    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones
#    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
#    noise_eta: float = 0.01
#    noise_gamma: float = 0.0
#    noise_seed: int = 1
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

# Run experiments
python source/batchnorm_exp.py dataset='cifar10' regularizer=None training_steps=320001 architecture='mlp_3' size=500
wait $!

#python source/batchnorm_exp.py dataset='cifar10' regularizer=None training_steps=320001 architecture='conv_4_2' 'size="(256, 128)"'
#wait $!