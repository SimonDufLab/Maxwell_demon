"""Optax optimizers configured to work with logger"""
import optax
from utils.utils import load_mnist, load_cifar10, load_fashion_mnist

optimizer_choice = {
    "adam": optax.adam,
    "sgd": optax.sgd
}

dataset_choice = {
    "mnist": load_mnist,
    "fashion mnist": load_fashion_mnist,
    "cifar10": load_cifar10
}

regularizer_choice = (
    "cdg_l2",
    "cdg_lasso",
    "l2"
)
