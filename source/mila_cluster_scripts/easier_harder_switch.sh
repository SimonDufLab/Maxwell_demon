#!/bin/bash

#SBATCH --job-name=easier_harder_switch
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=96:00:00                                   # The job will run for 1 hour

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
#    total_steps: int = 10001
#    report_freq: int = 500
#    record_freq: int = 10
#    lr: float = 1e-3
#    optimizer: str = "adam"
#    datasets: Tuple[str] = ("mnist", "fashion mnist")  # Datasets to use, listed from easier to harder
#    kept_classes: Tuple[Union[int, None]] = (None, None)  # Number of classes to use, listed from easier to harder
#    regularizer: Optional[str] = "cdg_l2"
#    reg_param: float = 1e-4

# Run experiments

#python source/easier_harder_switch.py 'datasets="(cifar10, cifar10)"' 'kept_classes="(5, None)"' compare_to_partial_reset=True info=Ash_and_Adams activation=abs
#wait $!

python source/easier_harder_switch.py 'datasets="(cifar10, cifar10)"' 'kept_classes="(5, None)"' compare_to_partial_reset=False info=Ash_and_Adams architecture=resnet18 size=64 activation=abs
wait $!
