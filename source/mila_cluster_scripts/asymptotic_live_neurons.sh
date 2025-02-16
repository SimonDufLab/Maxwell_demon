#!/bin/bash

#SBATCH --job-name=asymptotic_live_neurons
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=24G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00 #36:00:00 #around 8 for Resnet                                  # The job will run for 2.5 hours
#SBATCH -x 'cn-d[001-004], cn-g[005-012,017-026]'  # Excluding DGX system, will require a jaxlib update


# Make sure we are located in the right directory and on right branch
cd ~/repositories/Maxwell_demon || exit
git checkout exp-config

# Load required modules
module load python/3.8
module load cuda/11.2/cudnn/8.1

# Load venv
source venv/bin/activate

# Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=true
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

python source/asymptotic_live_neurons.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=62501 report_freq=2000 record_freq=500 pruning_freq=1000 with_bn=True lr_schedule=one_cycle normalize_inputs=False reg_param_decay_cycles=1 info=Resnet18_cifar10_srigl_setup reg_param=0.0 'sizes="(4, 8, 16, 24, 32, 48, 64)"' optimizer=adamw wd_param=0.0005 lr=0.05 train_batch_size=64 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None activation=relu zero_end_reg_param=True save_wanda=False dynamic_pruning=True kept_classes=10 init_seed=63
wait $!

# Special, for plots of typical death behavior of neurons
#python source/asymptotic_live_neurons.py dataset='cifar10_srigl' architecture='srigl_resnet18' training_steps=62501 report_freq=2000 record_freq=500 pruning_freq=1000 with_bn=True lr_schedule=one_cycle normalize_inputs=False reg_param_decay_cycles=1 info=Resnet18_cifar10_srigl_setup reg_param=0.0 'sizes="(64,)"' optimizer=adamw wd_param=0.0005 lr=0.01 train_batch_size=64 augment_dataset=True gradient_clipping=False noisy_label=0.0 regularizer=None activation=elu zero_end_reg_param=True save_wanda=False dynamic_pruning=False kept_classes=10 init_seed=63
#wait $!

# --------------------------------------------------------------------------------------------------------------------------------------------------

#python source/asymptotic_live_neurons.py dataset='cifar10' architecture='resnet18' training_steps=15626 report_freq=1000 record_freq=100 pruning_freq=1000 live_freq=25000 with_bn=True lr_schedule=one_cycle normalize_inputs=True info=Resnet19_DD_width reg_param=0.00005 optimizer=adam augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None activation=relu wd_param=0.0001 lr=0.01 train_batch_size=256 'sizes="(2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64)"' init_seed=23
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' architecture='resnet18' training_steps=150000 report_freq=10000 record_freq=500 pruning_freq=10000 live_freq=25000 with_bn=True lr_schedule=piecewise_constant normalize_inputs=True info=Resnet19_DD_width reg_param=0.0 optimizer=adam augment_dataset=True gradient_clipping=True noisy_label=0.0 regularizer=None activation=relu wd_param=0.0001 lr=0.01 train_batch_size=256 'sizes="(2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64)"' init_seed=24
#wait $!

#python source/asymptotic_live_neurons.py dataset="cifar10" regularizer=None training_steps=250001 architecture=conv_4_2 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (96, 48), (128, 64), (192, 96), (256, 128), (512, 256))"' report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=rdm_img_perm_conv_4_2 lr_schedule=piecewise_constant train_batch_size=32 optimizer=sgd lr=0.05 init_seed=51
#wait $!

#python source/asymptotic_live_neurons.py dataset="mnist" regularizer=None training_steps=9375 optimizer=adam architecture=mlp_3 report_freq=125 record_freq=25 pruning_freq=50 live_freq=25000 info=zoom_5_epochs_low_noise_mlp_3 lr_schedule=piecewise_constant train_batch_size=512 lr=0.001 init_seed=42 permuted_img_ratio=0.0 death_batch_size=128
#wait $!

#python source/asymptotic_live_neurons.py dataset="cifar10" regularizer=None training_steps=250001 architecture=conv_4_2 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (96, 48), (128, 64), (192, 96), (256, 128), (512, 256))"' report_freq=2500 record_freq=250 pruning_freq=1000 live_freq=25000 info=var_check_test_conv_momentum lr_schedule=piecewise_constant train_batch_size=32 var_check=True optimizer=momentum9 lr=0.001 init_seed=51 with_bn=False
#wait $!

#python source/asymptotic_live_neurons.py dataset="mnist" regularizer=None training_steps=250001 architecture=mlp_3 report_freq=2500 record_freq=250 pruning_freq=1000 live_freq=25000 info=opt_comparison_mlp_3 lr_schedule=piecewise_constant train_batch_size=8 avg_for_eps=True 'epsilon_close="(0.001, 0.005, 0.01)"' optimizer=sgd lr=0.5 init_seed=51
#wait $!

#python source/asymptotic_live_neurons.py dataset="cifar10" regularizer=None training_steps=250001 architecture=conv_4_2 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (96, 48), (128, 64), (192, 96), (256, 128), (512, 256))"' report_freq=2500 record_freq=250 pruning_freq=1000 live_freq=25000 info=opt_comparison_conv_4_2 lr_schedule=piecewise_constant train_batch_size=8 avg_for_eps=True 'epsilon_close="(0.001,0.005,0.01)"' optimizer=momentum9 lr=0.005 init_seed=51 with_bn=True bn_config=no_scale_and_offset
#wait $!

#python source/asymptotic_live_neurons.py dataset="mnist" regularizer=None training_steps=250001 architecture=mlp_3 report_freq=2500 record_freq=250 pruning_freq=1000 live_freq=25000 info=opt_comparison_mlp_3 lr_schedule=piecewise_constant train_batch_size=8 avg_for_eps=True 'epsilon_close="(0.001, 0.005, 0.01)"' optimizer=momentum9 lr=0.005 init_seed=51
#wait $!

#python source/asymptotic_live_neurons.py dataset="fashion mnist" regularizer=None training_steps=250001 optimizer=adam architecture=conv_4_2 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (96, 48), (128, 64), (192, 96), (256, 128), (512, 256))"' report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=bias_comparison_conv_4_2 lr_schedule=piecewise_constant train_batch_size=32 lr=0.001 init_seed=52 with_bias=False
#wait $!

#python source/asymptotic_live_neurons.py dataset="mnist" regularizer=None training_steps=250001 optimizer=adam architecture=mlp_3 report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=bias_comparison_mlp_3 lr_schedule=piecewise_constant train_batch_size=32 lr=0.001 init_seed=78 with_bias=False
#wait $!

#python source/asymptotic_live_neurons.py dataset="mnist" regularizer=None training_steps=250001 optimizer=adam architecture=mlp_3 report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=kept_classes_comparison_mlp_3 lr_schedule=piecewise_constant train_batch_size=32 lr=0.001 init_seed=42 kept_classes=2
#wait $!

#python source/asymptotic_live_neurons.py dataset="fashion mnist" regularizer=None training_steps=250001 architecture=mlp_3 report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=noisy_opt_comparison_mlp_3_fmnist lr_schedule=piecewise_constant train_batch_size=8 noisy_label=0.2 optimizer=adam lr=0.001 init_seed=51
#wait $!

#python source/asymptotic_live_neurons.py dataset="mnist" regularizer=None training_steps=250001 architecture=mlp_3 report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=BN_comparison_mlp_3 lr_schedule=piecewise_constant train_batch_size=32 optimizer=adam lr=0.5 with_bn=True init_seed=52
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=None training_steps=1000001 optimizer='adam' architecture='conv_4_2' 'sizes="((2, 128), (4, 128), (6, 128), (8, 128), (10, 128), (12, 128), (16, 128), (20, 128), (24, 128), (32, 128), (40, 128), (4w8, 128))"' report_freq=5000 record_freq=500 pruning_freq=2500 live_freq=50000 info=DD_and_overfitting_regime init_seed=41 activation=abs noisy_label=0.4
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=1562501 optimizer='adam' lr=0.005 architecture=resnet18 'sizes="(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)"' report_freq=10000 record_freq=500 pruning_freq=10000 live_freq=100000 train_batch_size=64 info=DD_resnet18_long_split_beg_l1_decay lr_schedule=piecewise_constant final_lr=0.00001 activation=relu with_bn=True noisy_label=0.2 augment_dataset=True init_seed=122 reg_param=0.00001 regularizer=lasso reg_param_decay_cycles=4 zero_end_reg_param=True
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=1562501 optimizer='adam' lr=0.005 architecture=resnet18 'sizes="(12, 13, 14, 16, 20, 24, 32, 40, 48, 56, 64)"' report_freq=10000 record_freq=500 pruning_freq=10000 live_freq=100000 train_batch_size=64 info=DD_resnet18_long_split_end_l1_decay lr_schedule=piecewise_constant final_lr=0.00001 activation=relu with_bn=True noisy_label=0.2 augment_dataset=True init_seed=122 reg_param=0.00001 regularizer=lasso reg_param_decay_cycles=4 zero_end_reg_param=True
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=500001 optimizer='adam' lr=0.005 architecture=resnet18 'sizes="(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64)"' report_freq=5000 record_freq=500 pruning_freq=2500 live_freq=50000 info=DD_and_overfitting_regime activation=abs noisy_label=0.1 init_seed=27
#wait $!

#python source/asymptotic_live_neurons.py dataset='mnist' regularizer=None training_steps=50001 optimizer='adam' architecture='mlp_3' 'sizes="(8, 16, 24, 32, 48, 64, 96, 128, 250, 500, 750)"' linear_switch=True info='linear switch exps'
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=250001 architecture='mlp_3' lr=0.001 optimizer='adam' add_noise=False train_batch_size=4 info=batch_size_comparison
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=250001 architecture=mlp_3 lr=1.0 optimizer='adam' add_noise=True 'noise_imp="(0, 1)"' info=noise_only_exp noise_eta=0.001 
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=250001 architecture=mlp_3 lr=1.0 optimizer='adam' add_noise=True 'noise_imp="(0, 1)"' info=noise_only_exp noise_eta=0.01
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=250001 architecture=mlp_3 lr=1.0 optimizer='adam' add_noise=True 'noise_imp="(0, 1)"' info=noise_only_exp noise_eta=0.1
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=250001 architecture=mlp_3 lr=1.0 optimizer='adam' add_noise=True 'noise_imp="(0, 1)"' info=noise_only_exp noise_eta=1.0
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=cdg_l2 training_steps=250001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (48, 32), (64, 32), (96, 48), (128, 64), (192, 96), (256, 128))"' lr=0.001 optimizer='adam' report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=cdg_l2_perf measure_linear_perf=True reg_param=0.0000001 init_seed=55
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=cdg_l2 training_steps=250001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (48, 32), (64, 32), (96, 48), (128, 64), (192, 96), (256, 128))"' lr=0.001 optimizer='adam' report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=cdg_l2_perf measure_linear_perf=True reg_param=0.0000001 init_seed=56
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=cdg_l2 training_steps=250001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (48, 32), (64, 32), (96, 48), (128, 64), (192, 96), (256, 128))"' lr=0.001 optimizer='adam' report_freq=2500 record_freq=250 pruning_freq=500 live_freq=25000 info=cdg_l2_perf measure_linear_perf=True reg_param=0.0000001 init_seed=57
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=None training_steps=250001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (128, 64), (256, 128), (512, 256))"' lr=0.001 optimizer='adam' add_noise=False info=dropout_comparison dropout_rate=0.2
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=None training_steps=250001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (128, 64), (256, 128), (512, 256))"' lr=0.001 optimizer='adam' add_noise=False info=dropout_comparison dropout_rate=0.4
#wait $!

#python source/asymptotic_live_neurons.py dataset='mnist' regularizer=None training_steps=250001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (128, 64), (256, 128), (512, 256))"' lr=0.001 optimizer='adam' add_noise=False info=dropout_comparison dropout_rate=0.2
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=None training_steps=250001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (128, 64), (256, 128), (512, 256))"' lr=0.001 optimizer='adam' add_noise=False info=dropout_comparison dropout_rate=0.8
#wait $!

#python source/asymptotic_live_neurons.py dataset='fashion mnist' regularizer=None training_steps=320001 architecture='conv_4_2' 'sizes="((8, 32), (16, 32), (32, 32), (64, 32), (128, 64), (256, 128), (512, 256), (640, 384), (768, 512))"' optimizer='adam' lr=0.001 add_noise=True noise_eta=0.01
#wait $!

#python source/asymptotic_live_neurons.py dataset='cifar10' regularizer=None training_steps=500001 architecture='resnet18' 'sizes="(4, 8, 12, 16, 24, 32, 64, 96, 128, 256)"' report_freq=5000 record_freq=500 pruning_freq=2500 live_freq=50000 optimizer=adam info=resnet_lr_comp_with_decay lr_schedule=piecewise_constant init_seed=49 lr=0.01
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
