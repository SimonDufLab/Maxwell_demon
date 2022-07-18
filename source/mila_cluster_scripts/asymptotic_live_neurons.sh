#!/bin/bash

#SBATCH --job-name=asymptotic_live_neurons
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=72:00:00                                   # The job will run for 2.5 hours

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
#    classes: int = 10  # Number of classes in the training dataset
#    architecture: str = "mlp_3"
#    sizes: Any = (50, 100, 250, 500, 750, 1000, 1250, 1500, 2000)
#    regularizer: Optional[str] = "cdg_l2"
#    reg_param: float = 1e-4
#    epsilon_close: float = 0.0  # Relaxing criterion for dead neurons, epsilon-close to relu gate

# Run experiments
python source/asymptotic_live_neurons.py dataset='cifar10' regularizer='cdg_l2' training_steps=200001 architecture='conv_6_2' 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (128, 64), (256, 128), (512, 256), (640, 384), (768, 512))"' init_seed=97 dynamic_pruning=True
wait $!
#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=200001 architecture='conv_6_2' 'sizes="((8, 128), (16, 128), (32, 128), (64, 128), (128, 128), (256, 128), (512, 128), (640, 128), (768, 128), (1024, 128))"' init_seed=28
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=200001 architecture='conv_4_2' 'sizes="((1024, 768),)"' dynamic_pruning=True 
#wait $!

#python source/asymptotic_live_neurons.py dataset='mnist' regularizer='None' training_steps=200001 architecture='conv_3_2' 'sizes="((8, 25), (16, 50), (32, 100), (64, 200), (128, 400), (256, 800), (512, 1600), (1024, 1600))"'
#wait $!
#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer='None' training_steps=200001 architecture='conv_3_2' 'sizes="((8, 25), (16, 50), (32, 100), (64, 200), (128, 400), (256, 800), (512, 1600), (1024, 1600))"'
#wait $!
#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer='None' training_steps=200001 architecture='conv_3_2' 'sizes="((8, 25), (16, 50), (32, 100), (64, 200), (128, 400), (256, 800), (512, 1600), (1024, 1600))"'
#wait $!
#python source/asymptotic_live_neurons.py dataset='mnist' regularizer='cdg_l2' training_steps=200001 architecture='conv_3_2' 'sizes="((8, 25), (16, 50), (32, 100), (64, 200), (128, 400), (256, 800), (512, 1600), (1024, 1600))"'
#wait $!
#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer='cdg_l2' training_steps=200001 architecture='conv_3_2' 'sizes="((8, 25), (16, 50), (32, 100), (64, 200), (128, 400), (256, 800), (512, 1600), (1024, 1600))"'
#wait $!
#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer='cdg_l2' training_steps=200001 architecture='conv_3_2' 'sizes="((8, 25), (16, 50), (32, 100), (64, 200), (128, 400), (256, 800), (512, 1600), (1024, 1600))"'
