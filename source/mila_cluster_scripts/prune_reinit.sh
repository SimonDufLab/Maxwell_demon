#!/bin/bash
#SBATCH --job-name=prune_reinit
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=24G                                        # Ask for 10 GB of RAM
#SBATCH --time=4:00:00                                  # The job will run for 24 hours

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

python source/prune_reinit.py training_steps=63001 report_freq=2000 record_freq=500 full_ds_eval_freq=10000 dataset='cifar10' architecture='resnet18' size=64 with_bn=True lr_schedule=fix_steps normalize_inputs=True optimizer=momentum9 train_batch_size=128 augment_dataset=True gradient_clipping=True activation=relu info=Resnet19_compare_LTH_momentum_decay_reg cycling_reg_param=0.0 rdm_reinit=True rewinding=True "lr_decay_steps='(32000, 48000)'" reg_param_decay_cycles=3 zero_end_reg_param=True wd_param=0.0 regularizer=l2 lr=0.1 reg_param=0.005 init_seed=560
wait $!

#python source/prune_reinit.py training_steps=15626 report_freq=1000 record_freq=100 full_ds_eval_freq=1000 dataset='cifar10' architecture='resnet18' size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True optimizer=adam wd_param=0.0005 train_batch_size=256 augment_dataset=True gradient_clipping=True regularizer=cdg_l2 activation=relu info=Resnet19_compare_LTH_param_count cycling_reg_param=0.0 reg_param=0.0001 lr=0.05 init_seed=520
#wait $!

#python source/prune_reinit.py dataset='cifar10' architecture='vgg16' training_steps=15626 report_freq=1000 record_freq=100 full_ds_eval_freq=1000 "size='(64, 4096)'" with_bn=True lr_schedule=one_cycle normalize_inputs=True info=vgg16_compare_LTH_param_count optimizer=adam wd_param=0.0005 train_batch_size=256 augment_dataset=True gradient_clipping=True regularizer=cdg_l2 activation=relu cycling_reg_param=0.0 reg_param=0.005 lr=0.05 init_seed=432
#wait $!

#python source/prune_reinit.py training_steps=250001 optimizer=adam architecture=mlp_3 report_freq=2500 record_freq=250 full_ds_eval_freq=1000 lr_schedule=piecewise_constant info=mnist_mlp_decay end_lr=0.001 size=500 train_batch_size=8 pruning_cycles=18 init_seed=13
#wait $!

#python source/prune_reinit.py training_steps=250001 optimizer=adam architecture=conv_4_2 dataset='fashion mnist' report_freq=2500 record_freq=250 full_ds_eval_freq=1000 lr_schedule=piecewise_constant info=fmnist_conv_4_2_decay_cycles end_lr=0.001 'size="(128, 512)"' train_batch_size=8 pruning_cycles=18 init_seed=11
#wait $!
