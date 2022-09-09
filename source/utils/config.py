"""Optax optimizers configured to work with logger"""
import jax
import optax
from utils.utils import load_mnist_torch, load_cifar10_torch, load_fashion_mnist_torch, load_cifar100_tf
from utils.utils import load_mnist_tf, load_cifar10_tf, load_fashion_mnist_tf
from models.mlp import mlp_3, mlp_3_bn
from models.convnet import conv_3_2, conv_3_2_bn, conv_4_2, conv_4_2_bn, conv_6_2, conv_6_2_bn
from models.resnet import resnet18

optimizer_choice = {
    "adam": optax.adam,
    "sgd": optax.sgd,
    "noisy_sgd": optax.noisy_sgd,
}

dataset_choice = {
    "mnist": load_mnist_tf,
    "mnist-torch": load_mnist_torch,
    "fashion mnist": load_fashion_mnist_tf,
    "fashion mnist-torch": load_fashion_mnist_torch,
    "cifar10": load_cifar10_tf,
    "cifar10-torch": load_cifar10_torch,
    "cifar100": load_cifar100_tf,
}

regularizer_choice = (
    "None",
    "cdg_l2",
    "cdg_lasso",
    "l2"
)

# Return the desired architecture along with a bool indicating if there is a
# is_training flag for this specific model
architecture_choice = {
    "mlp_3": mlp_3,
    "conv_3_2": conv_3_2,
    "conv_4_2": conv_4_2,
    "conv_6_2": conv_6_2,
    "resnet18": resnet18,
}

bn_architecture_choice = {
    "mlp_3": mlp_3_bn,
    "conv_3_2": conv_3_2_bn,
    "conv_4_2": conv_4_2_bn,
    "conv_6_2": conv_6_2_bn,
}

activation_choice = {
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "abs": jax.numpy.abs,  # Absolute value as the activation function
    "elu": jax.nn.elu,
    "swish": jax.nn.swish,
    "tanh": jax.nn.tanh
}

dataset_target_cardinality = {  # Hard-encoding the number of classes in given dataset for easy retrieval
    "mnist": 10,
    "fashion mnist": 10,
    "cifar10": 10,
    "cifar100": 100
}
