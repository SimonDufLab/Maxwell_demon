#!/bin/bash

#SBATCH --job-name=imgnet_controlling_overfitting
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=8                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100l:1                                     # Ask for 1 GPU
#SBATCH --mem=96G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00 #36:00:00 #around 8 for Resnet                                  # The job will run for 2.5 hours
# #SBATCH -x 'cn-b[001-005], cn-d[001-004], cn-g[005-012,017-026], cn-e[002-003], kepler5'  # Excluding DGX system, will require a jaxlib update and kepler 5 that have 16GB GPU memory and v100 with 32Gb memory
                                # The job will run for 2.5 hours

# Copying Imagenet
#echo "Started copying test data"
#mkdir -p $SLURM_TMPDIR/imagenet2012/manual
#cd       $SLURM_TMPDIR/imagenet2012/manual
#cp /network/datasets/imagenet/ILSVRC2012_img_val.tar .
## tar  -xf /network/datasets/imagenet/ILSVRC2012_img_val.tar # --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
#echo "Finished copying test data"
#echo "Started copying train data"
## mkdir -p $SLURM_TMPDIR/imagenet2012/train
## cd       $SLURM_TMPDIR/imagenet2012/train
#cp /network/datasets/imagenet/ILSVRC2012_img_train.tar .
## tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'
#echo "Finished copying train data"
#echo "Quickly extracting labels as well"
#cd       $SLURM_TMPDIR/imagenet2012
#tar  -xf /network/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz

# Other approach, already prepared tf dataset
echo "Copying tf imagenet dataset"
cd $SLURM_TMPDIR
tar -xzf $SCRATCH/tf_tar_imagenet/tf_imagenet.tar.gz
echo "Finished copying train+test data"


echo "Dataset located in"
echo $SLURM_TMPDIR/imagenet2012
export TFDS_DATA_DIR=$SLURM_TMPDIR
#echo "Dataset structure:"
#find $SLURM_TMPDIR/imagenet2012 -type d -print -exec sh -c "ls -p '{}' | head -5" \;


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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export TF_FORCE_GPU_ALLOW_GROWTH=true
export TCMALLOC_RELEASE_RATE=10.0
# export TCMALLOC_HEAP_LIMIT_MB=24576 # 24GB max heap
#export TCMALLOC_HEAP_LIMIT_MB=31744 # 31GB max heap for mem=32GB
#export TCMALLOC_HEAP_LIMIT_MB=39936  # 39GB max heap for mem=40G
#export TCMALLOC_HEAP_LIMIT_MB=48128 # 47GB max heap for mem=48GB
export LD_PRELOAD="/home/mila/s/simon.dufort-labbe/.conda/envs/py38jax_tf/lib/libtcmalloc_minimal.so.4"


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

echo "Checking if tcmalloc was correctly attributed to LD_PRELOAD"
echo $LD_PRELOAD

# Classic setting; for test
python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=500456 report_freq=1000 record_freq=500 pruning_freq=12500 size=64 with_bn=True lr_schedule=warmup_cosine_decay normalize_inputs=True reg_param_decay_cycles=4 info=Resnet50_momentum_warmup_cosinedecay_new_opt 'reg_params="(0.0008,)"' optimizer=adam_to_momentum wd_param=0.0001 lr=10.0 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=cdg_l2 activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True init_seed=33
wait $!

####################

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=90083 report_freq=1000 record_freq=200 pruning_freq=2500 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet50_one_cycle_decay_chck_archt 'reg_params="(0.0,)"' optimizer=momentum9 wd_param=0.00001 lr=1.0 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=33
#wait $!

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=90083 report_freq=1000 record_freq=200 pruning_freq=2500 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet50_one_cycle_decay_new_archt 'reg_params="(0.000001,)"' optimizer=adamw wd_param=0.00001 lr=0.002 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=33
#wait $!

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=90083 report_freq=1000 record_freq=200 pruning_freq=2500 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet50_one_cycle_decay_new_archt 'reg_params="(0.00005,)"' optimizer=adamw wd_param=0.0001 lr=0.002 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=33
#wait $!

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=90083 report_freq=1000 record_freq=200 pruning_freq=2500 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet50_one_cycle_decay_new_archt 'reg_params="(0.0001,)"' optimizer=adamw wd_param=0.0001 lr=0.002 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=33
#wait $!

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=90083 report_freq=1000 record_freq=200 pruning_freq=2500 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet50_one_cycle_decay 'reg_params="(0.005,)"' optimizer=momentum9 wd_param=0.00001 lr=1.0 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=True init_seed=33
#wait $!

#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=450410 report_freq=1000 record_freq=500 pruning_freq=12500 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet50_one_cycle_decay 'reg_params="(0.000025,)"' optimizer=adamw wd_param=0.0001 lr=0.0003 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=False init_seed=33
#wait $!

# Appeared to be too much reg for imagenet; reuse in extreme compression only
#python source/controlling_overfitting.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet18' training_steps=90083 report_freq=1000 record_freq=200 pruning_freq=2500 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 info=Resnet19_dyn_pruning_one_cycle_decay 'reg_params="(0.01, 0.05, 0.1)"' optimizer=adamw wd_param=0.0001 lr=0.002 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=False init_seed=32
#wait $!
