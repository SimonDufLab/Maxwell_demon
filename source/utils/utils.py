import copy
from typing import Any, Generator, Mapping, Optional, Tuple
from dataclasses import fields

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
from jax.flatten_util import ravel_pytree
from collections.abc import Iterable

import psutil
import sys
import gc

OptState = Any
Batch = Mapping[int, np.ndarray]


##############################
# Death detection and revival utilities
##############################
def sum_across_filter(filters):
    if filters.ndim > 1:
        return jnp.sum(filters, axis=tuple(range(filters.ndim - 1)))
    else:
        return filters


def death_check_given_model(model, with_activations=False, epsilon=0, check_tail=False, with_dropout=False):
    """Return a boolean array per layer; with True values for dead neurons"""
    assert epsilon >= 0, "epsilon value must be positive"
    if check_tail:
        assert epsilon <= 1, "for tanh activation fn, epsilon must be smaller than 1"

    def relu_test(arr):  # Test for relu, leaky-relu, elu, swish, etc. activation fn. Check if bigger than epsilon
        return arr <= epsilon

    def tanh_test(arr):  # Test for tanh, sigmoid, etc. activation fn. Check if abs(tanh(x)) >= 1-epsilon
        return jnp.abs(arr) >= 1-epsilon  # TODO: test fn not compatible with convnets

    if check_tail:
        test_fn = tanh_test
    else:
        test_fn = relu_test

    if with_dropout:
        dropout_key = jax.random.PRNGKey(0)  # dropout rate is zero during death eval
        model_apply_fn = Partial(model.apply, rng=dropout_key)
    else:
        model_apply_fn = model.apply

    @jax.jit
    def _death_check(_params: hk.Params, _state: hk.State, _batch: Batch) -> Union[jnp.ndarray, Tuple[jnp.array, jnp.array]]:
        (_, activations), _ = model_apply_fn(_params, _state, x=_batch, return_activations=True, is_training=False)
        activations = jax.tree_map(jax.vmap(sum_across_filter), activations)  # Sum across the filter first if conv layer; do nothing if fully connected
        sum_activations = jax.tree_map(Partial(jnp.sum, axis=0), activations)
        if with_activations:
            return activations, jax.tree_map(test_fn, sum_activations)
        else:
            return jax.tree_map(test_fn, sum_activations)

    return _death_check


def scanned_death_check_fn(death_check_fn, scan_len, with_activations_data=False):
    @jax.jit
    def sum_dead_neurons(leaf1, leaf2):
        return jnp.logical_and(leaf1.astype(bool), leaf2.astype(bool))

    if with_activations_data:
        def scan_death_check(params, state, batch_it):
            # def scan_dead_neurons_over_whole_ds(previous_dead, __):
            #     batched_activations, dead_neurons = death_check_fn(
            #         params, next(batch_it))
            #     return jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons), batched_activations
            #
            # _, carry_init = death_check_fn(params, next(batch_it))
            # dead_neurons, batched_activations = jax.lax.scan(scan_dead_neurons_over_whole_ds, carry_init, None,
            #                                                  scan_len)
            # return batched_activations, dead_neurons

            activations, previous_dead = death_check_fn(params, state, next(batch_it))
            # batched_activations = [activations]
            running_max = jax.tree_map(Partial(jnp.amax, axis=0), activations)
            running_mean = jax.tree_map(Partial(jnp.mean, axis=0), activations)
            running_count = count_activations_occurrence(activations)
            N = 1
            for i in range(scan_len-1):
                activations, dead_neurons = death_check_fn(params, state, next(batch_it))
                # batched_activations.append(activations)
                running_max = update_running_max(activations, running_max)
                running_mean = update_running_mean(activations, running_mean)
                N += 1
                running_count = update_running_count(activations, running_count)

                previous_dead = jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons)

            return (running_max, jax.tree_map(lambda x: x/N, running_mean), running_count), previous_dead

        return scan_death_check
    else:
        def scan_death_check(params, state, batch_it):
            # def scan_dead_neurons_over_whole_ds(previous_dead, __):
            #     dead_neurons = death_check_fn(params, next(batch_it))
            #     return jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons), None
            #
            # carry_init = death_check_fn(params, next(batch_it))
            # dead_neurons, _ = jax.lax.scan(scan_dead_neurons_over_whole_ds, carry_init, None,
            #                                scan_len)
            # return dead_neurons

            previous_dead = death_check_fn(params, state, next(batch_it))
            for i in range(scan_len-1):
                dead_neurons = death_check_fn(params, state, next(batch_it))
                previous_dead = jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons)

            return previous_dead

        return scan_death_check


@jax.jit
def count_dead_neurons(death_check):
    dead_per_layer = [jnp.sum(layer) for layer in death_check]
    return sum(dead_per_layer), tuple(dead_per_layer)


@jax.jit
def logical_and_sum(leaf):
    """Perform a logical_and across the first dimension (over a batch)"""
    def scan_log_or(carry, next_item):
        return jnp.logical_and(carry.astype(bool), next_item.astype(bool)), None
    summed_leaf, _ = jax.lax.scan(scan_log_or, jnp.zeros_like(leaf[0]), leaf)
    return summed_leaf


@jax.jit
def update_running_max(new_batch, previous_max):
    new_max = jax.tree_map(Partial(jnp.amax, axis=0), new_batch)
    new_max = jax.tree_map(jnp.maximum, new_max, previous_max)
    return new_max


@jax.jit
def count_activations_occurrence(activations_list):
    """Count how many times neurons activated (post-relu; > 0) in the given batch"""
    def _count_occurrence(leaf):
        leaf = (leaf > 0).astype(int)
        return jnp.sum(leaf, axis=0)
    return jax.tree_map(_count_occurrence, activations_list)


@jax.jit
def update_running_mean(new_batch, previous_mean):
    mean_sum = jax.tree_map(Partial(jnp.mean, axis=0), new_batch)
    mean_sum = jax.tree_map(jnp.add, mean_sum, previous_mean)
    return mean_sum


@jax.jit
def update_running_count(new_batch, previous_count):
    new_count = count_activations_occurrence(new_batch)
    new_count = jax.tree_map(jnp.add, new_count, previous_count)
    return new_count


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
# Filtering dead utilities (pruning)
##############################
def remove_dead_neurons_weights(params, neurons_state, opt_state=None):
    """Given the current params and the neuron state (True if dead) returns a
     filtered params dict (and its associated optimizer state) with dead weights
      removed and the new size of the layers (that is, # of conv filters or # of
       neurons in fully connected layer, etc.)"""
    neurons_state = jax.tree_map(jnp.logical_not, neurons_state)
    filtered_params = copy.deepcopy(params)

    # print(jax.tree_map(jax.numpy.shape, filtered_params))

    if opt_state:
        # field_names = [field.name for field in fields(opt_state[0])]
        field_names = list(opt_state[0]._fields)
        # print(field_names)
        if 'count' in field_names:
            field_names.remove('count')
        filter_in_opt_state = copy.deepcopy([getattr(opt_state[0], field) for field in field_names])
    layers_name = list(params.keys())
    for i, layer in enumerate(layers_name[:-1]):
        # print(i, layer)
        # print(neurons_state[i].shape)
        for dict_key in filtered_params[layer].keys():
            # print(filtered_params[layer][dict_key].shape)
            filtered_params[layer][dict_key] = filtered_params[layer][dict_key][..., neurons_state[i]]
            if opt_state:
                for j, field in enumerate(filter_in_opt_state):
                    # print(field)
                    filter_in_opt_state[j][layer][dict_key] = field[layer][dict_key][..., neurons_state[i]]

        # for dict_key in filtered_params[layers_name[i+1]].keys():
        #     print(neurons_state[i].shape)
        #     print(filtered_params[layers_name[i+1]][dict_key].shape)
        to_repeat = filtered_params[layers_name[i+1]]['w'].shape[-2] // neurons_state[i].size
        if to_repeat > 1 :
            current_state = jnp.repeat(neurons_state[i].reshape(1, -1), to_repeat, axis=0).flatten()
            # print(neurons_state[i])
            # print(current_state)
        else:
            current_state = neurons_state[i]
        filtered_params[layers_name[i+1]]['w'] = filtered_params[layers_name[i+1]]['w'][..., current_state, :]
        if opt_state:
            for j, field in enumerate(filter_in_opt_state):
                filter_in_opt_state[j][layers_name[i + 1]]['w'] = filter_in_opt_state[j][layers_name[i + 1]]['w'][...,
                                                                  current_state, :]

    if opt_state:
        filtered_opt_state, empty_state = copy.copy(opt_state)
        for j, field in enumerate(field_names):
            # setattr(filtered_opt_state, field, filter_in_opt_state[j])
            filtered_opt_state = filtered_opt_state._replace(**{field:filter_in_opt_state[j]})
        new_opt_state = filtered_opt_state, empty_state

    new_sizes = [int(jnp.sum(layer)) for layer in neurons_state]

    if opt_state:
        return filtered_params, new_opt_state, tuple(new_sizes)
    else:
        return filtered_params, tuple(new_sizes)


# def rebuild_model(new_architecture, sizes):
#     """Rebuild a model with less params to accommodate the filtering of dead neurons"""
#     new_architecture = architecture_choice[exp_architecture](sizes, 10)  # TODO accomate more than 10 classes...
#     new_net = build_models(new_architecture)
#     return new_net


def neuron_state_from_params_magnitude(params, eps):
    """ Return neuron state (to detect 'quais-dead') according to the mean size of
    the parameters of a neuron"""
    neuron_state = []
    for key in list(params.keys())[:-1]:
        neuron_means = jax.tree_leaves(jax.tree_map(abs_mean_except_last_dim, params[key]))
        neuron_means = sum(neuron_means) / len(neuron_means)
        neuron_state.append(neuron_means <= eps)
    return neuron_state


##############################
# Training utilities
##############################
def accuracy_given_model(model, with_dropout=False):

    if with_dropout:
        dropout_key = jax.random.PRNGKey(0)  # dropout rate is zero during death eval
        model_apply_fn = Partial(model.apply, rng=dropout_key)
    else:
        model_apply_fn = model.apply

    @jax.jit
    def _accuracy(_params: hk.Params, _state: hk.State, _batch: Batch) -> jnp.ndarray:
        predictions, _ = model_apply_fn(_params, _state, x=_batch, return_activations=False, is_training=False)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == _batch[1])

    return _accuracy


def create_full_accuracy_fn(accuracy_fn, scan_len):
    def full_accuracy_fn(params, state, batch_it):
        # def scan_accuracy_fn(_, __):
        #     acc = accuracy_fn(params, next(batch_it))
        #     return None, acc
        # _, all_acc = jax.lax.scan(scan_accuracy_fn, None, None, scan_len)
        # return jnp.mean(all_acc)

        acc = [accuracy_fn(params, state, next(batch_it))]
        for i in range(scan_len-1):
            acc.append(accuracy_fn(params, state, next(batch_it)))
        return jnp.mean(jnp.stack(acc))
    return full_accuracy_fn


def ce_loss_given_model(model, regularizer=None, reg_param=1e-4, classes=None, is_training=True, with_dropout=False):
    """ Build the cross-entropy loss given the model"""
    if not classes:
        classes = 10

    if regularizer:
        assert regularizer in ["cdg_l2", "cdg_lasso", "l2"]
        if regularizer == "l2":
            def reg_fn(params):
                return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        if regularizer == "cdg_l2":
            def reg_fn(params):
                return 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_leaves(params))
        if regularizer == "cdg_lasso":
            def reg_fn(params):
                return sum(jnp.sum(jnp.clip(p, 0)) for p in jax.tree_leaves(params))
    else:
        def reg_fn(params):
            return 0

    if is_training and with_dropout:
        @jax.jit
        def _loss(params: hk.Params, state: hk.State, batch: Batch, dropout_key: Any) -> Union[jnp.ndarray, Any]:
            next_dropout_key, rng = jax.random.split(dropout_key)
            logits, state = model.apply(params, state, rng, batch, return_activations=False, is_training=is_training)
            labels = jax.nn.one_hot(batch[1], classes)

            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            loss = softmax_xent + reg_param * reg_fn(params)

            return loss, (state, next_dropout_key)

    else:
        if with_dropout:
            dropout_key = jax.random.PRNGKey(0)  # dropout rate is zero during death eval
            model_apply_fn = Partial(model.apply, rng=dropout_key)
        else:
            model_apply_fn = model.apply

        @jax.jit
        def _loss(params: hk.Params, state: hk.State, batch: Batch) -> Union[jnp.ndarray, Any]:
            logits, state = model_apply_fn(params, state, x=batch, return_activations=False, is_training=is_training)
            labels = jax.nn.one_hot(batch[1], classes)

            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            loss = softmax_xent + reg_param * reg_fn(params)

            if is_training:
                return loss, state
            else:
                return loss

    return _loss


def grad_normalisation_per_layer(param_leaf):
    var = jnp.var(param_leaf)
    return param_leaf/jnp.sqrt(var+1)


def update_given_loss_and_optimizer(loss, optimizer, noise=False, noise_imp=(1, 1), noise_live_only=False, norm_grad=False, with_dropout=False):
    """Learning rule (stochastic gradient descent)."""

    if with_dropout:
        @jax.jit
        def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _drop_key) -> Tuple[
            hk.Params, Any, OptState, jax.random.PRNGKeyArray]:
            grads, (new_state, next_drop_key) = jax.grad(loss, has_aux=True)(_params, _state, _batch, _drop_key)
            if norm_grad:
                grads = jax.tree_map(grad_normalisation_per_layer, grads)
            updates, _opt_state = optimizer.update(grads, _opt_state)
            new_params = optax.apply_updates(_params, updates)
            return new_params, new_state, _opt_state, next_drop_key

    else:
        if not noise:
            @jax.jit
            def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch) -> Tuple[hk.Params, Any, OptState]:
                grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch)
                if norm_grad:
                    grads = jax.tree_map(grad_normalisation_per_layer, grads)
                updates, _opt_state = optimizer.update(grads, _opt_state)
                new_params = optax.apply_updates(_params, updates)
                return new_params, new_state, _opt_state
        else:
            a, b = noise_imp
            if noise_live_only:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                            _key: Any) -> Tuple[hk.Params, Any, OptState, Any]:
                    grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch)
                    key, next_key = jax.random.split(_key)
                    flat_grads, unravel_fn = ravel_pytree(grads)
                    added_noise = _var * jax.random.normal(key, shape=flat_grads.shape)
                    added_noise = added_noise * (jnp.abs(flat_grads) >= 1e-8)  # Only apply noise to weights with gradient!=0
                    # noisy_grad = unravel_fn(a * flat_grads + b * added_noise)
                    updates, _opt_state = optimizer.update(grads, _opt_state)
                    flat_updates, _ = ravel_pytree(updates)
                    noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)
                    return new_params, new_state, _opt_state, next_key
            else:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                            _key: Any) -> Tuple[hk.Params, Any, OptState, Any]:
                    grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch)
                    updates, _opt_state = optimizer.update(grads, _opt_state)
                    key, next_key = jax.random.split(_key)
                    flat_updates, unravel_fn = ravel_pytree(updates)
                    added_noise = _var*jax.random.normal(key, shape=flat_updates.shape)
                    noisy_updates = unravel_fn(a*flat_updates + b*added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)
                    return new_params, new_state, _opt_state, next_key

    return _update


def vmap_axes_mapping(dict_container):
    """Utility function to indicate to vmap that in_axes={key:0} for all keys in dict_container"""
    def _map_over(v):
        return 0  # Need to vmap over all arrays in the container
    return jax.tree_map(_map_over, dict_container)


@jax.jit
def dict_split(container):
    """Split back the containers into their specific components, returning them as a tuple"""
    treedef = jax.tree_structure(container)
    leaves = jax.tree_leaves(container)
    _to = leaves[0].shape[0]

    leaves = jax.tree_map(Partial(jnp.split, indices_or_sections=_to), leaves)
    leaves = jax.tree_map(jnp.squeeze, leaves)
    splitted_dict = tuple([treedef.unflatten(list(z)) for z in zip(*leaves)])
    return splitted_dict


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
    padding = (32 - image.shape[0])//2  # TODO: dirty, ensure compatibility with other shapes than 28, 32, ...
    image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
    return image/255, label


def load_tf_dataset(dataset: str, split: str, *, is_training: bool, batch_size: int,
                    subset: Optional[int] = None, transform: bool = True,
                    cardinality: bool = False):  # -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches.
    subset: If only want a subset, number of classes to build the subset from
    """
    ds = tfds.load(dataset, split=split, as_supervised=True, data_dir="./data")
    ds_size = int(ds.cardinality())
    if subset is not None:
        # assert subset < 10, "subset must be smaller than 10"
        # indices = np.random.choice(10, subset, replace=False)

        def filter_fn(image, label):
          return tf.reduce_any(subset == int(label))
        ds = ds.filter(filter_fn)  # Only take the randomly selected subset

    ds = ds.map(interval_zero_one)
    # ds = ds.cache().repeat()
    if is_training:
        # if subset is not None:
        #     ds = ds.cache().repeat()
        ds = ds.shuffle(10 * batch_size, seed=0)
        # ds = ds.take(batch_size).cache().repeat()
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(-1)

    if (subset is not None) and transform:
        return tf_compatibility_iterator(iter(tfds.as_numpy(ds)), subset)
    else:
        if cardinality:
            return ds_size, iter(tfds.as_numpy(ds))
        else:
            return iter(tfds.as_numpy(ds))


def load_mnist_tf(split: str, is_training, batch_size, subset=None, transform=True, cardinality=False):
    return load_tf_dataset("mnist:3.*.*", split=split, is_training=is_training, batch_size=batch_size,
                           subset=subset, transform=transform, cardinality=cardinality)


def load_cifar10_tf(split: str, is_training, batch_size, subset=None, transform=True, cardinality=False):
    return load_tf_dataset("cifar10", split=split, is_training=is_training, batch_size=batch_size,
                           subset=subset, transform=transform, cardinality=cardinality)


def load_cifar100_tf(split: str, is_training, batch_size, subset=None, transform=True, cardinality=False):
    return load_tf_dataset("cifar100", split=split, is_training=is_training, batch_size=batch_size,
                           subset=subset, transform=transform, cardinality=cardinality)


def load_fashion_mnist_tf(split: str, is_training, batch_size, subset=None, transform=True, cardinality=False):
    return load_tf_dataset("fashion_mnist", split=split, is_training=is_training, batch_size=batch_size,
                           subset=subset, transform=transform, cardinality=cardinality)


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
def build_models(train_layer_list, test_layer_list=None, name=None, with_dropout=False):
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
            self.train_layers = train_layer_list
            self.test_layers = test_layer_list

        def __call__(self, x, return_activations=False, is_training=True):
            activations = []
            x = x[0].astype(jnp.float32)
            if is_training or (self.test_layers is None):
                layers = self.train_layers
            else:
                layers = self.test_layers
            for layer in layers[:-1]:  # Don't append final output in activations list
                x = hk.Sequential([mdl() for mdl in layer])(x)
                if return_activations:
                    if type(x) is tuple:
                        activations += x[1]
                        # activations.append(x[0])
                    else:
                        activations.append(x)
                if type(x) is tuple:
                    x = x[0]
            x = hk.Sequential([mdl() for mdl in layers[-1]])(x)
            if return_activations:
                return x, activations
            else:
                return x

    def primary_model(x, return_activations=False, is_training=True):
        return ModelAndActivations()(x, return_activations, is_training)

    # return hk.without_apply_rng(hk.transform(typical_model)), hk.without_apply_rng(hk.transform(secondary_model))
    if not with_dropout:
        return hk.without_apply_rng(hk.transform_with_state(primary_model))
    else:
        return hk.transform_with_state(primary_model)


##############################
# Varia
##############################
def get_total_neurons(architecture, sizes):
    if architecture == 'mlp_3':
        if type(sizes) == int:  # Size can be specified with 1 arg, an int
            sizes = [sizes, sizes * 3]
    elif architecture == 'conv_3_2':
        if len(sizes) == 2:  # Size can be specified with 2 args
            sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], sizes[1]]
    elif architecture == 'conv_4_2':
        if len(sizes) == 2:  # Size can be specified with 2 args
            sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]
    elif architecture == 'conv_6_2':
        if len(sizes) == 2:  # Size can be specified with 2 args
            sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]
    elif architecture == "resnet18":
        if type(sizes) == int:  # Size can be specified with 1 arg, an int
            sizes = [sizes,
                     sizes, sizes,
                     sizes, sizes,
                     2*sizes, 2*sizes,
                     2*sizes, 2*sizes,
                     4 * sizes, 4 * sizes,
                     4 * sizes, 4 * sizes,
                     8 * sizes, 8 * sizes,
                     8 * sizes, 8 * sizes,
                     ]

    return sum(sizes), tuple(sizes)


def size_to_string(item):
    """Helper function to print correctly layer size within aim logger"""
    if isinstance(item, Iterable):
        x = [(str(i) + '_') for i in list(item)]
        word = ''
        for i in x:  # Python bug? sum(x) doesn't work...
            word += i
        return word[:-1]
    else:
        return str(item)


def abs_mean_except_last_dim(tree_leaf):
    axes = tuple(range(len(tree_leaf.shape)-1))
    return jnp.mean(jnp.abs(tree_leaf), axis=axes)


def clear_caches():
    """Just in case for future needs, clear whole cache associated to jax
    Taken from: https://github.com/google/jax/issues/10828"""
    process = psutil.Process()
    # if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
    for module_name, module in sys.modules.items():
        if module_name.startswith("jax"):
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if hasattr(obj, "cache_clear"):
                    obj.cache_clear()
    gc.collect()
