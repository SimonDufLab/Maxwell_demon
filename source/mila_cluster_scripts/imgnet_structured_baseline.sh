#!/bin/bash

#SBATCH --job-name=imgnet_structured_baseline
#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=8 #16                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100l:1                                     # Ask for 1 GPU
#SBATCH --mem=48G #256G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00 #36:00:00 #around 8 for Resnet                                  # The job will run for 2.5 hours
# #SBATCH -x 'cn-b[001-005], cn-d[001-004], cn-g[005-012,017-026], cn-e[002-003], kepler5'  # Excluding DGX system, will require a jaxlib update and kepler 5 that have 16GB GPU memory

# Copying Imagenet
echo "Copying tf imagenet dataset"
cd $SLURM_TMPDIR
tar -xzf $SCRATCH/tf_tar_imagenet/tf_imagenet.tar.gz
echo "Finished copying train+test data"

echo "Dataset located in"
echo $SLURM_TMPDIR/imagenet2012
export TFDS_DATA_DIR=$SLURM_TMPDIR

# Make sure we are located in the right directory and on right branch
cd ~/repositories/Maxwell_demon || exit
git checkout exp-config

# Load required modules
module load anaconda/3

# Load venv
conda activate py38jax_tf
ulimit -n 4096 # Aim hit too many files open while preparing ds

# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export TF_FORCE_GPU_ALLOW_GROWTH=true
export TCMALLOC_RELEASE_RATE=10.0
export LD_PRELOAD="/home/mila/s/simon.dufort-labbe/.conda/envs/py38jax_tf/lib/libtcmalloc_minimal.so.4"


# Default configuration:
#    training_steps: int = 250001
#    report_freq: int = 2500
#    record_freq: int = 250
#    full_ds_eval_freq: int = 1000
#    live_freq: int = 25000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
#    lr: float = 1e-3
#    gradient_clipping: bool = False
#    lr_schedule: str = "None"
#    final_lr: float = 1e-6
#    lr_decay_steps: int = 5  # If applicable, amount of time the lr is decayed (example: piecewise constant schedule)
#    train_batch_size: int = 512
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    optimizer: str = "adamw"
#    activation: str = "relu"  # Activation function used throughout the model
#    dataset: str = "mnist"
#    normalize_inputs: bool = False  # Substract mean across channels from inputs and divide by variance
#    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
#    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
#    noisy_label: float = 0.0  # ratio (between [0,1]) of labels to randomly (uniformly) flip
#    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
#    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
#    architecture: str = "mlp_3"
#    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
#    with_bn: bool = False  # Add batchnorm layers or not in the models
#    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
#    size: Any = 50  # Can also be a tuple for convnets
#    regularizer: Optional[str] = "None"
#    reg_param: float = 5e-4
#    wd_param: Optional[float] = None
#    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
#    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
#    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
#    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
#    pruning_criterion: Optional[str] = None
#    pruning_density: float = 0.0
#    pruning_steps: int = 1  # Number of steps for single shot iterative pruning
#    modulate_target_density: bool = True  # Not in paper but in code, modify the threshold calculation
#    pruning_args: Any = None
#    init_seed: int = 41
#    dropout_rate: float = 0
#    with_rng_seed: int = 428
#    save_wanda: bool = False  # Whether to save weights and activations value or not
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

echo "Checking if tcmalloc was correctly attributed to LD_PRELOAD"
echo $LD_PRELOAD

# pruning density -> neuron sparsity. Typically, run for /rho in [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9
#python source/structured_baseline.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=500456 report_freq=2000 record_freq=500 full_ds_eval_freq=600000 size=64 with_bn=True lr_schedule=warmup_piecewise_decay "lr_decay_steps='(30, 70, 90)'" lr_decay_scaling_factor=0.1 normalize_inputs=True info=Resnet50_structured_imgnet_srigl_momentum pruning_criterion=snap optimizer=momentum9 wd_param=0.0001 lr=0.1 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 label_smoothing=0.1 regularizer=None activation=relu save_wanda=False modulate_target_density=False pruning_steps=5 pruning_density=0.8 preempt_handling=True init_seed=21
#wait $!

# for resnet, ~80% param sparsity --> pruning density = 0.4, 0.45, 0.5
python source/structured_baseline.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=500456 report_freq=2000 record_freq=500 full_ds_eval_freq=600000 size=64 with_bn=True lr_schedule=warmup_piecewise_decay "lr_decay_steps='(30, 70, 90)'" lr_decay_scaling_factor=0.1 normalize_inputs=True info=Resnet50_structured_imgnet_srigl_adam pruning_criterion=earlycrop optimizer=adam wd_param=0.0001 lr=0.001 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 label_smoothing=0.1 regularizer=None activation=relu save_wanda=False modulate_target_density=False pruning_steps=5 pruning_density=0.5 preempt_handling=True init_seed=31
wait $!
