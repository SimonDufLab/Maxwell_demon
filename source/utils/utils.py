from typing import Any, Generator, Mapping, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Union, Tuple
from jax.tree_util import Partial

OptState = Any
Batch = Mapping[int, np.ndarray]


##############################
# Reviving utilities
##############################
def death_check_given_model(model, with_activations=False):
    """Return a boolean array per layer; with True values for dead neurons"""
    @jax.jit
    def _death_check(_params: hk.Params, _batch: Batch) -> Union[jnp.ndarray, Tuple[jnp.array, jnp.array]]:
        _, activations = model.apply(_params, _batch, True)
        sum_activations = jax.tree_map(Partial(jnp.sum, axis=0), activations)
        if with_activations:
            return activations, jax.tree_map(lambda arr: arr == 0, sum_activations)
        else:
            return jax.tree_map(lambda arr: arr == 0, sum_activations)

    return _death_check


@jax.jit
def count_dead_neurons(death_check):
    return sum([jnp.sum(layer) for layer in death_check])


@jax.jit
def map_decision(current_leaf, potential_leaf):
    return jnp.where(current_leaf != 0, current_leaf, potential_leaf)


@jax.jit
def reinitialize_dead_neurons(neuron_states, old_params, new_params):
    """ Given the activations value for the whole training set, build a mask that is used for reinitialization
      neurons_state: neuron states (either 0 or 1) post-activation. Will be 1 if y <=  0
      old_params: current parameters value
      new_params: new parameters' dict to pick from weights being reinitialized"""
    neuron_states = [jnp.logical_not(state) for state in neuron_states]
    layers = list(old_params.keys())
    for i in range(len(neuron_states)):
        for weight_type in list(old_params[layers[i]].keys()):  # Usually, 'w' and 'b'
            old_params[layers[i]][weight_type] = old_params[layers[i]][weight_type] * neuron_states[i]
        kernel_param = 'w'
        old_params[layers[i+1]][kernel_param] = old_params[layers[i+1]][kernel_param] * neuron_states[i].reshape(-1, 1)
    reinitialized_params = jax.tree_util.tree_map(map_decision, old_params, new_params)

    return reinitialized_params


##############################
# Training utilities
##############################
def accuracy_given_model(model):
    @jax.jit
    def _accuracy(_params: hk.Params, _batch: Batch) -> jnp.ndarray:
        predictions = model.apply(_params, _batch, False)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == _batch[1])

    return _accuracy


def ce_loss_given_model(model, regularizer=None, reg_param=1e-4, classes=None):
    """ Build the cross-entropy loss given the model"""
    if not classes:
        classes = 10

    @jax.jit
    def _loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        logits = model.apply(params, batch, False)
        labels = jax.nn.one_hot(batch[1], classes)

        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        if regularizer:
            assert regularizer in ["cdg_l2", "cdg_lasso", "l2"]
            if regularizer == "l2":
                reg_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            if regularizer == "cdg_l2":
                reg_loss = 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_leaves(params))
            if regularizer == "cdg_lasso":
                reg_loss = sum(jnp.sum(jnp.clip(p, 0)) for p in jax.tree_leaves(params))
            return softmax_xent + reg_param * reg_loss
        else:
            return softmax_xent
    return _loss


def update_given_loss_and_optimizer(loss, optimizer):
    """Learning rule (stochastic gradient descent)."""

    @jax.jit
    def _update(_params: hk.Params, _opt_state: OptState, _batch: Batch) -> Tuple[hk.Params, OptState]:
        grads = jax.grad(loss)(_params, _batch)
        updates, _opt_state = optimizer.update(grads, _opt_state)
        new_params = optax.apply_updates(_params, updates)
        return new_params, _opt_state

    return _update


##############################
# Dataset loading utilities
##############################
def map_targets(target, indices):
  return jnp.asarray(target == indices).nonzero(size=1)


@jax.jit
def transform_batch_tf(batch, indices):
    transformed_targets = jax.vmap(map_targets, in_axes=(0, None))(batch[1], indices)
    return batch[0], transformed_targets[0].flatten()


class tf_compatibility_iterator:
    def __init__(self, tfds_iterator, indices):
        self.tfds_iterator = tfds_iterator
        self.indices = indices

    def __next__(self):
        return transform_batch_tf(next(self.tfds_iterator), self.indices)


def interval_zero_one(image, label):
    return image/255, label


def load_tf_dataset(dataset: str, split: str, *, is_training: bool, batch_size: int,
                 subset: Optional[int] = None, transform: bool = True,):  # -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches.
    subset: If only want a subset, number of classes to build the subset from
    """
    ds = tfds.load(dataset, split=split, as_supervised=True, data_dir="./data")
    if subset is not None:
        # assert subset < 10, "subset must be smaller than 10"
        # indices = np.random.choice(10, subset, replace=False)

        def filter_fn(image, label):
          return tf.reduce_any(subset == int(label))
        ds = ds.filter(filter_fn)  # Only take the randomly selected subset

    ds = ds.map(interval_zero_one)
    ds = ds.cache().repeat()
    if is_training:
        # if subset is not None:
        #     ds = ds.cache().repeat()
        ds = ds.shuffle(10 * batch_size, seed=0)
        # ds = ds.take(batch_size).cache().repeat()
    ds = ds.batch(batch_size)

    if (subset is not None) and transform:
        return tf_compatibility_iterator(iter(tfds.as_numpy(ds)), subset)
    else:
        return iter(tfds.as_numpy(ds))


def load_mnist_tf(split: str, is_training, batch_size, subset=None, transform=True):
    return load_tf_dataset("mnist:3.*.*", split=split, is_training=is_training, batch_size=batch_size,
                           subset=subset, transform=transform)


def load_cifar10_tf(split: str, is_training, batch_size, subset=None, transform=True):
    return load_tf_dataset("cifar10", split=split, is_training=is_training, batch_size=batch_size,
                           subset=subset, transform=transform)


def load_fashion_mnist_tf(split: str, is_training, batch_size, subset=None, transform=True):
    return load_tf_dataset("fashion_mnist", split=split, is_training=is_training, batch_size=batch_size,
                           subset=subset, transform=transform)


# Pytorch dataloader
# @jax.jit
def transform_batch_pytorch(targets, indices):
    # transformed_targets = jax.vmap(map_targets, in_axes=(0, None))(targets, indices)
    transformed_targets = targets.apply_(lambda t: torch.nonzero(t == torch.tensor(indices))[0])
    return transformed_targets


class compatibility_iterator:
    """ Ensure that pytorch iterator return next batch exactly as tf iterator does"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)

    def __next__(self):
        try:
            next_data, next_target = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            next_data, next_target = next(self.dataloader_iter)

        next_data = jnp.array(next_data.permute(0, 2, 3, 1))
        next_target = jnp.array(next_target)
        return next_data, next_target


def load_dataset(dataset: Any, is_training: bool, batch_size: int, subset: Optional[int] = None,
                 transform: bool = True, num_workers: int = 2):
    if subset is not None:
        # assert subset < 10, "subset must be smaller than 10"
        # indices = np.random.choice(10, subset, replace=False)
        subset_idx = np.isin(dataset.targets, subset)
        dataset.data, dataset.targets = dataset.data[subset_idx], dataset.targets[subset_idx]
        if transform:
            dataset.targets = transform_batch_pytorch(dataset.targets, subset)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=num_workers)
    return compatibility_iterator(data_loader)


def load_mnist_torch(is_training, batch_size, subset=None, transform=True, num_workers=2):
    dataset = datasets.MNIST('./data', train=is_training, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 ]))  # transforms.Normalize((0.1307,), (0.3081,)) -> want positive inputs
    return load_dataset(dataset, is_training=is_training, batch_size=batch_size, subset=subset,
                        transform=transform, num_workers=num_workers)


def load_cifar10_torch(is_training, batch_size, subset=None, transform=True, num_workers=2):
    dataset = datasets.CIFAR10('./data', train=is_training, download=True,
                               transform=transforms.Compose([
                                    transforms.ToTensor()]))
    return load_dataset(dataset, is_training=is_training, batch_size=batch_size, subset=subset,
                        transform=transform, num_workers=num_workers)


def load_fashion_mnist_torch(is_training, batch_size, subset=None, transform=True, num_workers=2):
    dataset = datasets.FashionMNIST('./data', train=is_training, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()]))
    return load_dataset(dataset, is_training=is_training, batch_size=batch_size, subset=subset,
                        transform=transform, num_workers=num_workers)


##############################
# Module utilities
##############################
def build_models(layer_list, name=None):
    """ Take as input a list of haiku modules and return 2 different transform object:
    1) First is the typical model returning the outputs
    2) The other is the same model returning all activations values + output"""

    # # Build the model that only return the outputs
    # def typical_model(x):
    #     x = x[0].astype(jnp.float32) / 255
    #     mlp = hk.Sequential([mdl() for mdl in sum(layer_list, [])], name="mlp")
    #     return mlp(x)

    # And the model that also return the activations
    class ModelAndActivations(hk.Module):
        def __init__(self):
            super().__init__(name=name)
            self.layers = layer_list

        def __call__(self, x, return_activations=False):
            activations = []
            x = x[0].astype(jnp.float32)
            for layer in self.layers[:-1]:  # Don't append final output in activations list
                x = hk.Sequential([mdl() for mdl in layer])(x)
                activations.append(x)
            x = hk.Sequential([mdl() for mdl in self.layers[-1]])(x)
            if return_activations:
                return x, activations
            else:
                return x

    def secondary_model(x, return_activations=False):
        return ModelAndActivations()(x, return_activations)

    # return hk.without_apply_rng(hk.transform(typical_model)), hk.without_apply_rng(hk.transform(secondary_model))
    return hk.without_apply_rng(hk.transform(secondary_model))
