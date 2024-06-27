#!/bin/bash

#SBATCH --job-name=spurious_exps
#SBATCH --partition=main                           #  Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=24G   #24G for Resnet18  #32G on cpu nodes for grok                                       # Ask for 10 GB of RAM
#SBATCH --time=24:00:00 #36:00:00 #around 8 for Resnet                                  # The job will run for 2.5 hours
#SBATCH -x 'cn-d[001-004], cn-g[005-012,017-026]'  # Excluding DGX system, will require a jaxlib update

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

# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=true
#export TF_FORCE_GPU_ALLOW_GROWTH=true

# Default configuration:
#    training_steps: int = 250001
#    report_freq: int = 2500
#    record_freq: int = 250
#    pruning_freq: int = 1000
#    # live_freq: int = 25000  # Take a snapshot of the 'effective capacity' every <live_freq> iterations
#    record_gate_grad_stat: bool = False  # Record in logger info about gradient magnitude per layer throughout training
#    mod_via_gate_grad: bool = False  # Use gate gradients to rescale weight gradients if True -> shitty, don't use
#    lr: float = 1e-3
#    gradient_clipping: bool = False
#    lr_schedule: str = "constant"
#    warmup_ratio: float = 0.05  # ratio of total steps used for warming up lr, when applicable
#    final_lr: float = 1e-6
#    lr_decay_steps: Any = 5  # Number of epochs after which lr is decayed
#    lr_decay_scaling_factor: float = 0.1  # scaling factor for lr decay
#    train_batch_size: int = 512
#    eval_batch_size: int = 512
#    death_batch_size: int = 512
#    accumulate_batches: int = 1  # Make effective batch size for training: train_batch_size x accumulate_batches
#    optimizer: str = "adam"
#    alpha_decay: float = 5.0  # Param controlling transition speed from adam to momentum in adam_to_momentum optimizers
#    activation: str = "relu"  # Activation function used throughout the model
#    shifted_relu: float = 0.0  # Shift value (b) applied on output before activation. To promote dead neurons
#    dataset: str = "mnist"
#    normalize_inputs: bool = False  # Substract mean across channels from inputs and divide by variance
#    augment_dataset: bool = False  # Apply a pre-fixed (RandomFlip followed by RandomCrop) on training ds
#    label_smoothing: float = 0.0  # Level of smoothing applied during the loss calculation, 0.0 -> no smoothing
#    kept_classes: Optional[int] = None  # Number of classes in the randomly selected subset
#    noisy_label: float = 0.0  # ratio (between [0,1]) of labels to randomly (uniformly) flip
#    permuted_img_ratio: float = 0  # ratio ([0,1]) of training image in training ds to randomly permute their pixels
#    gaussian_img_ratio: float = 0  # ratio ([0,1]) of img to replace by gaussian noise; same mean and variance as ds
#    architecture: str = "mlp_3"
#    with_bias: bool = True  # Use bias or not in the Linear and Conv layers (option set for whole NN)
#    with_bn: bool = False  # Add batchnorm layers or not in the models
#    bn_config: str = "default"  # Different configs for bn; default have offset and scale trainable params
#    size: Any = 50  # Can also be a tuple for convnets
#    regularizer: Optional[str] = "cdg_l2"
#    reg_params: Any = (0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
#    masked_reg: Optional[str] = None  # If "all" exclude all bias and bn params, if "scale" only exclude scale param/ also offset_only, scale_only and bn_params_only
#    wd_param: Optional[float] = None
#    reg_param_decay_cycles: int = 1  # number of cycles inside a switching_period that reg_param is divided by 10
#    zero_end_reg_param: bool = False  # Put reg_param to 0 at end of training
#    reg_param_schedule: Optional[str] = None  # Schedule for reg_param, priority over reg_param_decay_cycles flag
#    reg_param_span: Optional[int] = None # More general than zero_end_reg_param, allow to decide when reg_param schedule falls to 0
#    epsilon_close: Any = None  # Relaxing criterion for dead neurons, epsilon-close to relu gate (second check)
#    avg_for_eps: bool = False  # Using the mean instead than the sum for the epsilon_close criterion
#    init_seed: int = 41
#    dynamic_pruning: bool = False
#    prune_after: int = 0  # Option: only start pruning after <prune_after> step has been reached
#    prune_at_end: Any = None  # If prune after training, tuple like (reg_param, lr, additional_steps)
#    pruning_reg: Optional[str] = "cdg_l2"
#    pruning_opt: str = "momentum9"  # Optimizer for pruning part after initial training
#    add_noise: bool = False  # Add Gaussian noise to the gradient signal
#    asymmetric_noise: bool = True  # Use an asymmetric noise addition, not applied to all neurons' weights
#    noise_live_only: bool = True  # Only add noise signal to live neurons, not to dead ones. reverse if False
#    noise_offset_only: bool = False  # Special option to only add noise to offset parameters of normalization layers
#    positive_offset: bool = False  # Force the noise on offset to be solely positive (increasing revival rate)
#    going_wild: bool = False  # This is not ok...
#    noise_imp: Any = (1, 1)  # Importance ratio given to (batch gradient, noise)
#    noise_eta: float = 0.01  # Variance of added noise; can only be used with a reg_param_schedule that it will match
#    noise_gamma: float = 0.0
#    noise_seed: int = 1
#    dropout_rate: float = 0
#    perturb_param: float = 0  # Perturbation parameter for rnadam
#    with_rng_seed: int = 428
#    linear_switch: bool = False  # Whether to switch mid-training steps to linear activations
#    measure_linear_perf: bool = False  # Measure performance over the linear network without changing activation
#    record_distribution_data: bool = False  # Whether to record distribution at end of training -- high memory usage
#    preempt_handling: bool = False  # Frequent checkpointing to handle SLURM preemption
#    jobid: Optional[str] = None  # Manually restart previous job from checkpoint
#    checkpoint_freq: int = 1  # in epochs
#    save_wanda: bool = False  # Whether to save weights and activations value or not
#    save_act_only: bool = True  # Only saving distributions with wanda, not the weights
#    info: str = ''  # Option to add additional info regarding the exp; useful for filtering experiments in aim

#### ConvNet
python source/controlling_overfitting.py training_steps=1000001 report_freq=10000 pruning_freq=5000 record_freq=1000 lr=1e-3 optimizer=adamw wd_param=0.0005 regularizer=lasso dataset=color_mnist_0 'size="(32, 64)"' architecture=conv_2_2 activation=relu with_bn=True 'reg_params="(0.000, 0.00001, 0.0001)"' train_batch_size=256 lr_schedule=step_warmup warmup_ratio=1e-5 preempt_handling=True checkpoint_freq=10000 reg_param_span=500000 init_seed=41 #reduced_ds_size=None
wait $!

#### GrokingTransNet

#### ResNet-18
python source/controlling_overfitting.py training_steps=1000001 report_freq=10000 pruning_freq=5000 record_freq=1000 lr=1e-3 optimizer=adamw wd_param=0.0005 regularizer=lasso dataset=color_mnist_0 size=64 architecture=srigl_resnet18 activation=relu with_bn=True 'reg_params="(0.000, 0.00001, 0.0001)"' train_batch_size=256 lr_schedule=step_warmup warmup_ratio=1e-5 preempt_handling=True checkpoint_freq=10000 reg_param_span=500000 init_seed=41 #reduced_ds_size=None
wait $!