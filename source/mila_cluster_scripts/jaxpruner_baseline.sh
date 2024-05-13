#!/bin/bash

#SBATCH --job-name=jaxpruner_baseline
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=16                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100l:1                                     # Ask for 1 GPU
#SBATCH --mem=256G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=48:00:00 #48 for Resnet/120 for ViT                                  # The job will run for t hours
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
ulimit -n 16000 # Aim hit too many files open while preparing ds

# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export TF_FORCE_GPU_ALLOW_GROWTH=true
export TCMALLOC_RELEASE_RATE=10.0
# export TCMALLOC_HEAP_LIMIT_MB=24576 # 24GB max heap
#export TCMALLOC_HEAP_LIMIT_MB=31744 # 31GB max heap for mem=32GB
#export TCMALLOC_HEAP_LIMIT_MB=39936  # 39GB max heap for mem=40G
#export TCMALLOC_HEAP_LIMIT_MB=48128 # 47GB max heap for mem=48GB
export LD_PRELOAD="/home/mila/s/simon.dufort-labbe/.conda/envs/py38jax_tf/lib/libtcmalloc_minimal.so.4"

echo "Checking if tcmalloc was correctly attributed to LD_PRELOAD"
echo $LD_PRELOAD

# Default configuration:
#    training_steps: int = 250001
#    report_freq: int = 2500
#    record_freq: int = 250
#    pruning_freq: int = 1000
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
#    regularizer: Optional[str] = None
#    reg_param: float = 5e-4
#    wd_param: Optional[float] = None
#    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
#    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
#    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
#    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
#    init_seed: int = 41
#    dynamic_pruning: bool = False
#    prune_after: int = 0  # Option: only start pruning after <prune_after> step has been reached
#    spar_levels: Any = (0.5, 0.8)
#    dropout_rate: float = 0
#    with_rng_seed: int = 428
#    save_wanda: bool = False  # Whether to save weights and activations value or not
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

# Run experiments

#python source/jaxpruner_baseline.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Res19_jaxpruner_baseline_WMP optimizer=adamw train_batch_size=256 augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None reg_param=0.001 activation=relu 'spar_levels="(0.0, 0.1, 0.2, 0.3, 0.4)"' wd_param=0.0005 sparsity_distribution=uniform pruning_method=WMP lr=0.005 init_seed=95
#wait $!

#python source/jaxpruner_baseline.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Res19_jaxpruner_baseline_WMP optimizer=adamw train_batch_size=256 augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None reg_param=0.001 activation=relu 'spar_levels="(0.5, 0.6, 0.7, 0.75, 0.8)"' wd_param=0.0005 sparsity_distribution=uniform pruning_method=WMP lr=0.005 init_seed=95
#wait $!

#python source/jaxpruner_baseline.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=True reg_param_decay_cycles=1 zero_end_reg_param=False info=Res19_jaxpruner_baseline_WMP optimizer=adamw train_batch_size=256 augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None reg_param=0.001 activation=relu 'spar_levels="(0.85, 0.9, 0.95, 0.97, 0.99)"' wd_param=0.0005 sparsity_distribution=uniform pruning_method=WMP lr=0.005 init_seed=95
#wait $!

#python source/jaxpruner_baseline.py dataset='mnist' architecture='mlp_3' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=100 with_bn=False normalize_inputs=True info=mnist_baselines optimizer=adam train_batch_size=256 noisy_label=0.0 regularizer=None reg_param=0.0 activation=relu 'spar_levels="(0.0, 0.1, 0.2, 0.3, 0.4)"' sparsity_distribution=uniform pruning_method=RigL lr=0.001 init_seed=95
#wait $!

#python source/jaxpruner_baseline.py dataset='mnist' architecture='mlp_3' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=100 with_bn=False normalize_inputs=True info=mnist_baselines optimizer=adam train_batch_size=256 noisy_label=0.0 regularizer=None reg_param=0.0 activation=relu 'spar_levels="(0.5, 0.6, 0.7, 0.75, 0.8)"' sparsity_distribution=uniform pruning_method=RigL lr=0.001 init_seed=95
#wait $!

#python source/jaxpruner_baseline.py dataset='mnist' architecture='mlp_3' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 size=100 with_bn=False normalize_inputs=True info=mnist_baselines optimizer=adam train_batch_size=256 noisy_label=0.0 regularizer=None reg_param=0.0 activation=relu 'spar_levels="(0.85, 0.9, 0.95, 0.97, 0.99)"' sparsity_distribution=uniform pruning_method=RigL lr=0.001 init_seed=95
#wait $!

#python source/jaxpruner_baseline.py dataset='cifar10' architecture='vgg16' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 "size='(64, 4096)'" with_bn=True lr_schedule=one_cycle normalize_inputs=True info=vgg16_jaxpruner_baseline_saliency optimizer=adamw train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None activation=relu 'spar_levels="(0.0, 0.1, 0.2, 0.3, 0.4)"' wd_param=0.0005 sparsity_distribution=uniform pruning_method=saliency lr=0.005 init_seed=91
#wait $!


#python source/jaxpruner_baseline.py dataset='cifar10' architecture='vgg16' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 "size='(64, 4096)'" with_bn=True lr_schedule=one_cycle normalize_inputs=True info=vgg16_jaxpruner_baseline_saliency optimizer=adamw train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None activation=relu 'spar_levels="(0.5, 0.6, 0.7, 0.75, 0.8)"' wd_param=0.0005 sparsity_distribution=uniform pruning_method=saliency lr=0.005 init_seed=91
#wait $!

#python source/jaxpruner_baseline.py dataset='cifar10' architecture='vgg16' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 "size='(64, 4096)'" with_bn=True lr_schedule=one_cycle normalize_inputs=True info=vgg16_jaxpruner_baseline_saliency optimizer=adamw train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None activation=relu 'spar_levels="(0.85, 0.9, 0.95, 0.97, 0.99)"' wd_param=0.0005 sparsity_distribution=uniform pruning_method=saliency lr=0.005 init_seed=91
#wait $!

#############################
# SriGL setup
# ResNet-18 -- ADAM
#python source/jaxpruner_baseline.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=1 zero_end_reg_param=False info=Res18_srigl_adam optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None reg_param=0.00 activation=relu 'spar_levels="(0.25, 0.5, 0.7, 0.8)"' preempt_handling=True sparsity_distribution=uniform pruning_method=SET init_seed=97
#wait $!

#python source/jaxpruner_baseline.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=1 zero_end_reg_param=False info=Res18_srigl_adam optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None reg_param=0.00 activation=relu 'spar_levels="(0.9, 0.925, 0.95, 0.97, 0.99)"' preempt_handling=True sparsity_distribution=uniform pruning_method=SET init_seed=97
#wait $!

# Exploring with one-cycle for quicker iteration
#python source/jaxpruner_baseline.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=15626 report_freq=1000 record_freq=200 pruning_freq=1000 size=64 with_bn=True lr_schedule=one_cycle normalize_inputs=False reg_param_decay_cycles=4 info=Resnet18_test_become_structured 'spar_levels="(0.8, 0.85, 0.9, 0.95, 0.99)"' optimizer=adamw wd_param=0.0005 lr=0.005 train_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=l2 activation=relu zero_end_reg_param=True reg_param_schedule=one_cycle save_wanda=False preempt_handling=True sparsity_distribution=uniform pruning_method=saliency reg_param=0.0 init_seed=96

# ResNet-18 -- SGDM
#python source/jaxpruner_baseline.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=1 zero_end_reg_param=False info=Res18_jaxpruner_sgdm optimizer=momentum9w wd_param=0.0005 lr=0.1 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None reg_param=0.00 activation=relu 'spar_levels="(0.25, 0.5, 0.7, 0.8)"' preempt_handling=True sparsity_distribution=uniform pruning_method=saliency init_seed=91
#wait $!

#python source/jaxpruner_baseline.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=97656 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=fix_steps lr_decay_steps=77 lr_decay_scaling_factor=0.2 normalize_inputs=False reg_param_decay_cycles=1 zero_end_reg_param=False info=Res18_jaxpruner_sgdm optimizer=momentum9w wd_param=0.0005 lr=0.1 train_batch_size=128 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None reg_param=0.00 activation=relu 'spar_levels="(0.9, 0.925, 0.95, 0.97, 0.99)"' preempt_handling=True sparsity_distribution=uniform pruning_method=saliency init_seed=91
#wait $!

################################
# SRigL Resnet-50 setup -- RigL + DemP
python source/jaxpruner_baseline.py dataset=$SLURM_TMPDIR/imagenet2012 architecture='resnet50' training_steps=500456 report_freq=2000 record_freq=500 pruning_freq=5000 size=64 with_bn=True lr_schedule=warmup_piecewise_decay "lr_decay_steps='(30, 70, 90)'" lr_decay_scaling_factor=0.1 normalize_inputs=True reg_param_decay_cycles=1 reg_param_schedule=one_cycle info=Resnet50_RigLDemP_momentum reg_param=0.0003 optimizer=momentum9w wd_param=0.0001 lr=0.1 train_batch_size=256 eval_batch_size=256 death_batch_size=256 augment_dataset=True gradient_clipping=False noisy_label=0.0 label_smoothing=0.1 regularizer=lasso activation=relu zero_end_reg_param=False save_wanda=False dynamic_pruning=False preempt_handling=True checkpoint_freq=1 masked_reg=scale_only init_seed=31 reg_param_span=150000 add_noise=True noise_eta=0.00005 'spar_levels="(0.8,)"' sparsity_distribution=erk pruning_method=RigL drop_fraction=0.1
wait $!
