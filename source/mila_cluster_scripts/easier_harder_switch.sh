#!/bin/bash

#SBATCH --job-name=easier_harder_switch
#SBATCH --partition=long                           #  Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=24G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=12:00:00 #36:00:00 #around 8 for Resnet                                  # The job will run for 2.5 hours
#SBATCH --reservation=ubuntu2204

# Make sure we are located in the right directory and on right branch
cd ~/repositories/Maxwell_demon || exit
git checkout exp-config

# Load required modules
#module load python/3.8
#module load cuda/11.2/cudnn/8.1
module load anaconda/3

# Load venv
#source venv/bin/activate
conda activate py38jax_tf
ulimit -n 4096 # Aim hit too many files open while preparing ds


# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=true
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

python source/easier_harder_switch.py 'datasets="(cifar10_srigl, cifar10_srigl)"' 'kept_classes="(5, None)"' compare_to_partial_reset=False info=Ash_and_Adams architecture=srigl_resnet18 total_steps=195311 optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=128 report_freq=2000 record_freq=500 full_checkpoint_freq=5000 size=64 with_bn=True bn_config=default reinit_state=True reverse_order=False perturb_param=0.0 init_seed=42
wait $!
