"""Optax optimizers configured to work with logger"""
import optax
from utils.utils import load_mnist_torch, load_cifar10_torch, load_fashion_mnist_torch
from utils.utils import load_mnist_tf, load_cifar10_tf, load_fashion_mnist_tf
from models.mlp import mlp_3
from models.convnet import conv_3_2, conv_4_2, conv_6_2

optimizer_choice = {
    "adam": optax.adam,
    "sgd": optax.sgd
}

dataset_choice = {
    "mnist": load_mnist_tf,
    "mnist-torch": load_mnist_torch,
    "fashion mnist": load_fashion_mnist_tf,
    "fashion mnist-torch": load_fashion_mnist_torch,
    "cifar10": load_cifar10_tf,
    "cifar10-torch": load_cifar10_torch,
}

regularizer_choice = (
    "None",
    "cdg_l2",
    "cdg_lasso",
    "l2"
)

architecture_choice = {
    "mlp_3": mlp_3,
    "conv_3_2": conv_3_2,
    "conv_4_2": conv_4_2,
    "conv_6_2": conv_6_2
}
