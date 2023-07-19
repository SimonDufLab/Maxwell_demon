#!/bin/bash

#SBATCH --job-name=preparing_imagenet
#SBATCH --partition=unkillable-cpu                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:0                                     # Ask for 1 GPU
#SBATCH --mem=16G   #24G for Resnet18                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00

mkdir $SCRATCH/tf_tar_imagenet
cd /network/datasets/imagenet.var/imagenet_tensorflow
tar czvf $SCRATCH/tf_tar_imagenet/tf_imagenet.tar.gz imagenet2012/
echo "Finished taring the prepared tf imagenet train split"
