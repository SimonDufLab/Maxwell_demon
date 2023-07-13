#!/bin/bash

#SBATCH --job-name=controlling_overfitting
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=36G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=12:00:00 #36:00:00 #around 8 for Resnet                                  # The job will run for 2.5 hours
#SBATCH -x 'cn-d[001-004], cn-g[005-012,017-026]'  # Excluding DGX system, will require a jaxlib update
                                # The job will run for 2.5 hours

# Copying Imagenet
echo "Started copying test data"
mkdir -p $SLURM_TMPDIR/imagenet
cd       $SLURM_TMPDIR/imagenet
cp -r /network/datasets/imagenet/ILSVRC2012_img_val.tar
# tar  -xf /network/datasets/imagenet/ILSVRC2012_img_val.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
echo "Finished copying test data"
echo "Started copying train data"
# mkdir -p $SLURM_TMPDIR/imagenet/train
# cd       $SLURM_TMPDIR/imagenet/train
cp -r /network/datasets/imagenet/ILSVRC2012_img_train.tar
# tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
echo "Finished copying train data"

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
#    pruning_freq: int = 2000
#    live_freq: int = 20000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
#    lr: float = 1e-3
#    lr_schedule: str = "None"
#    final_lr: float = 1e-6
#    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
#    train_batch_size: int = 128
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    optimizer: str = "adam"
#    activation: str = "relu"  # Activation function used throughout the model
#    dataset: str = "mnist"
#    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
#    noisy_label: float = 0.25  # ratio (between [0,1]) of labels to randomly (uniformly) flip
#    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
#    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
#    architecture: str = "mlp_3"
#    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
#    with_bn: bool = False  # Add batchnorm layers or not in the models
#    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
#    size: Any = 50  # Can also be a tuple for convnets
#    regularizer: Optional[str] = "cdg_l2"
#    reg_params: Any = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
#    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
#    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
#    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
#    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
#    init_seed: int = 41
#    dynamic_pruning: bool = False
#    add_noise: bool = False  # Add Gaussian noise to the gradient signal
#    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones
#    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
#    noise_eta: float = 0.01
#    noise_gamma: float = 0.0
#    noise_seed: int = 1
#    dropout_rate: float = 0
#    with_rng_seed: int = 428
#    linear_switch: bool = False  # Whether to switch mid-training steps to linear activations
#    measure_linear_perf: bool = False  # Measure performance over the linear network without changing activation
#    save_wanda: bool = False  # Whether to save weights and activations value or not
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

# Run experiments

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_dyn_pruning_one_cycle_decay 'reg_params="(0.0, 0.000001, 0.000005)"' optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=128 eval_batch_size=128 death_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=62
#wait $!

python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_dyn_pruning_one_cycle_decay 'reg_params="(0.00001, 0.00005, 0.0001)"' optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=128 eval_batch_size=128 death_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=62
wait $!

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_dyn_pruning_one_cycle_decay 'reg_params="(0.0005, 0.001, 0.005)"' optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=128 eval_batch_size=128 death_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=62
#wait $!

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_dyn_pruning_one_cycle_decay 'reg_params="(0.01, 0.05, 0.1)"' optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=128 eval_batch_size=128 death_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=62

