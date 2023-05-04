import copy
from typing import Any, Generator, Mapping, Optional, Tuple
from dataclasses import fields

import haiku as hk
import jax
import jax.numpy as jnp
import numpy
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Union, Tuple, List, Callable
from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree
from collections.abc import Iterable
from itertools import cycle

from optax._src import base
from optax._src import wrappers
from optax._src import combine
from optax._src import transform
from optax._src.alias import _scale_by_learning_rate

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


def mean_across_filter(filters):
    if filters.ndim > 1:
        return jnp.mean(filters, axis=tuple(range(filters.ndim - 1)))
    else:
        return filters


def death_check_given_model(model, with_activations=False, check_tail=False, with_dropout=False, avg=False, var=False):
    """Return a boolean array per layer; with True values for dead neurons"""
    # assert epsilon >= 0, "epsilon value must be positive"
    # if check_tail:
    #     assert epsilon <= 1, "for tanh activation fn, epsilon must be smaller than 1"

    def relu_test(epsilon, arr):  # Test for relu, leaky-relu, elu, swish, etc. activation fn. Check if bigger than epsilon
        return arr <= jnp.abs(epsilon)

    def tanh_test(epsilon, arr):  # Test for tanh, sigmoid, etc. activation fn. Check if abs(tanh(x)) >= 1-epsilon
        return jnp.abs(arr) >= 1-epsilon  # TODO: test fn not compatible with convnets

    if check_tail:
        _test_fn = tanh_test
    else:
        _test_fn = relu_test

    if with_dropout:
        dropout_key = jax.random.PRNGKey(0)  # dropout rate is zero during death eval
        model_apply_fn = Partial(model.apply, rng=dropout_key)
    else:
        model_apply_fn = model.apply

    if not var:
        @jax.jit
        def _death_check(_params: hk.Params, _state: hk.State, _batch: Batch, epsilon: float = 0) -> Union[
                            jnp.ndarray, Tuple[jnp.array, jnp.array]]:
            test_fn = Partial(_test_fn, epsilon)
            (_, activations), _ = model_apply_fn(_params, _state, x=_batch, return_activations=True, is_training=False)
            if avg:
                activations = jax.tree_map(jax.vmap(mean_across_filter),
                                           activations)  # mean across the filter first only if convnets
                sum_activations = jax.tree_map(Partial(jnp.mean, axis=0), activations)
            else:
                activations = jax.tree_map(jax.vmap(sum_across_filter),
                                           activations)  # Sum across the filter first only if convnets
                sum_activations = jax.tree_map(Partial(jnp.sum, axis=0), activations)
            if with_activations:
                return activations, jax.tree_map(test_fn, sum_activations)
            else:
                return jax.tree_map(test_fn, sum_activations)

        return _death_check

    else:
        @jax.jit
        def _death_check(_params: hk.Params, _state: hk.State, _batch: Batch, epsilon: float = 0) -> Union[
                         List[jnp.ndarray], Tuple[jnp.array, List[jnp.array]]]:
            (_, activations), _ = model_apply_fn(_params, _state, x=_batch, return_activations=True, is_training=False)
            if avg:
                activations = jax.tree_map(jax.vmap(mean_across_filter),
                                           activations)  # mean across the filter first only if convnets
            else:
                activations = jax.tree_map(jax.vmap(sum_across_filter),
                                           activations)  # Sum across the filter first only if convnets
            sum_activations = jax.tree_map(Partial(jnp.var, axis=0), activations)
            layer_eps = [jnp.mean(layer) / 100 for layer in sum_activations]
            if with_activations:
                return activations, [_test_fn(eps, layer) for layer, eps in zip(sum_activations, layer_eps)]
            else:
                return [_test_fn(eps, layer) for layer, eps in zip(sum_activations, layer_eps)]

        return _death_check


def scanned_death_check_fn(death_check_fn, scan_len, with_activations_data=False):
    @jax.jit
    def sum_dead_neurons(leaf1, leaf2):
        return jnp.logical_and(leaf1.astype(bool), leaf2.astype(bool))

    if with_activations_data:
        def scan_death_check(params, state, batch_it, epsilon=0):
            # def scan_dead_neurons_over_whole_ds(previous_dead, __):
            #     batched_activations, dead_neurons = death_check_fn(
            #         params, next(batch_it))
            #     return jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons), batched_activations
            #
            # _, carry_init = death_check_fn(params, next(batch_it))
            # dead_neurons, batched_activations = jax.lax.scan(scan_dead_neurons_over_whole_ds, carry_init, None,
            #                                                  scan_len)
            # return batched_activations, dead_neurons

            activations, previous_dead = death_check_fn(params, state, next(batch_it), epsilon)
            # batched_activations = [activations]
            running_max = jax.tree_map(Partial(jnp.amax, axis=0), activations)
            running_mean = jax.tree_map(Partial(jnp.mean, axis=0), activations)
            running_count = count_activations_occurrence(activations)
            running_var = jax.tree_map(Partial(jnp.var, axis=0), activations)
            N = 1
            for i in range(scan_len-1):
                activations, dead_neurons = death_check_fn(params, state, next(batch_it), epsilon)
                # batched_activations.append(activations)
                running_max = update_running_max(activations, running_max)
                running_mean = update_running_mean(activations, running_mean)
                running_var = update_running_var(activations, running_var)
                N += 1
                running_count = update_running_count(activations, running_count)

                previous_dead = jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons)

            return (running_max, jax.tree_map(lambda x: x/N, running_mean),
                    running_count, jax.tree_map(lambda x: x/N, running_var)), previous_dead

        return scan_death_check
    else:
        def scan_death_check(params, state, batch_it, epsilon=0):
            # def scan_dead_neurons_over_whole_ds(previous_dead, __):
            #     dead_neurons = death_check_fn(params, next(batch_it))
            #     return jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons), None
            #
            # carry_init = death_check_fn(params, next(batch_it))
            # dead_neurons, _ = jax.lax.scan(scan_dead_neurons_over_whole_ds, carry_init, None,
            #                                scan_len)
            # return dead_neurons

            previous_dead = death_check_fn(params, state, next(batch_it), epsilon)
            for i in range(scan_len-1):
                dead_neurons = death_check_fn(params, state, next(batch_it), epsilon)
                previous_dead = jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons)

            return previous_dead

        return scan_death_check


@jax.jit
def count_dead_neurons(death_check):
    dead_per_layer = [jnp.sum(layer) for layer in death_check]
    # return jnp.sum(ravel_pytree(dead_per_layer)[0]), tuple(dead_per_layer)
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
def update_running_var(new_batch, previous_mean):
    var_sum = jax.tree_map(Partial(jnp.var, axis=0), new_batch)
    var_sum = jax.tree_map(jnp.add, var_sum, previous_mean)
    return var_sum


@jax.jit
def update_running_count(new_batch, previous_count):
    new_count = count_activations_occurrence(new_batch)
    new_count = jax.tree_map(jnp.add, new_count, previous_count)
    return new_count


@jax.jit
def map_decision(current_leaf, potential_leaf):
    return jnp.where(current_leaf != 0, current_leaf, potential_leaf)


@jax.jit
def map_decision_with_bool_array(decision, current_leaf, potential_leaf):
    return jnp.where(decision, current_leaf, potential_leaf)


@jax.jit
def reinitialize_dead_neurons(neuron_states, old_params, new_params):
    """ Given the activations value for the whole training set, build a mask that is used for reinitialization
      neurons_state: neuron states (either 0 or 1) post-activation. Will be 1 if y <=  0
      old_params: current parameters value
      new_params: new parameters' dict to pick from weights being reinitialized"""
    neuron_states = [jnp.logical_not(state) for state in neuron_states]
    # neuron_states = jax.tree_map(jnp.logical_not, neuron_states)
    layers = list(old_params.keys())
    for i in range(len(neuron_states)):
        for weight_type in list(old_params[layers[i]].keys()):  # Usually, 'w' and 'b'
            old_params[layers[i]][weight_type] = old_params[layers[i]][weight_type] * neuron_states[i]
        kernel_param = 'w'
        old_params[layers[i + 1]][kernel_param] = old_params[layers[i + 1]][kernel_param] * neuron_states[
            i].reshape(-1, 1)
    reinitialized_params = jax.tree_util.tree_map(map_decision, old_params, new_params)

    return reinitialized_params


@jax.jit
def reinitialize_excluding_head(neuron_states, old_params, new_params):
    """ Given the activations value for the whole training set, build a mask that is used for reinitialization
      neurons_state: neuron states (either 0 or 1) post-activation. Will be 1 if y <=  0
      old_params: current parameters value
      new_params: new parameters' dict to pick from weights being reinitialized"""
    neuron_states = [jnp.logical_not(state) for state in neuron_states]
    bool_params = jax.tree_map(jnp.isfinite, old_params)
    # neuron_states = jax.tree_map(jnp.logical_not, neuron_states)
    layers = list(old_params.keys())
    for i in range(len(neuron_states)):
        for weight_type in list(old_params[layers[i]].keys()):  # Usually, 'w' and 'b'
            bool_params[layers[i]][weight_type] = jnp.logical_and(bool_params[layers[i]][weight_type], neuron_states[i])
        # kernel_param = 'w'
        # if i+2 < len(list(old_params.keys())):
        #     old_params[layers[i + 1]][kernel_param] = old_params[layers[i + 1]][kernel_param] * neuron_states[
        #         i].reshape(-1, 1)
    reinitialized_params = jax.tree_util.tree_map(map_decision_with_bool_array, bool_params, old_params, new_params)

    return reinitialized_params


@ jax.jit
def prune_outgoing_from_dead_neurons(neuron_states, params):
    """ To use when some neurons are frozen. This will remove (mask) the connection between reinitialized
    dead neurons and the frozen ones, preserving the representation"""
    neuron_states = [jnp.logical_not(state) for state in neuron_states]
    layers = list(params.keys())
    for i in range(len(neuron_states)):
        kernel_param = 'w'
        params[layers[i + 1]][kernel_param] = params[layers[i + 1]][kernel_param] * neuron_states[
            i].reshape(-1, 1)

    return jax.tree_map(map_decision, params, jax.tree_map(lambda v: v*0, params))


##############################
# Filtering dead utilities (pruning)
##############################
def extract_layer_lists(params):
    """Extract layer list. This function is solely because we want to do it upon param initialization, where
    we know that they will be in the desired order. This order is not guaranteed to remain the same, that's why we want
    to do it once. Returned layer lists will be kept in memory for later usage"""
    layers_name = list(params.keys())
    # print(layers_name)
    # Remove norm layer form layers_name list; nothing to prune in them
    _layers_name = []
    _shorcut_layers = []
    _short_bn_layers = []
    _bn_layers = []
    for layer in layers_name:
        if ("norm" not in layer) and ("bn" not in layer) and ('short' not in layer):
            _layers_name.append(layer)
        elif ('short' in layer) and ("norm" not in layer):
            _shorcut_layers.append(layer)
        elif ('short' in layer) and ("norm" in layer):
            _short_bn_layers.append(layer)
        elif ("bn" in layer) or ("norm" in layer):
            _bn_layers.append(layer)
    # print(layers_name)
    # print()
    # print(_layers_name)
    # print(len(_layers_name))
    # print(_bn_layers)
    # print(len(_bn_layers))
    # print(_shorcut_layers)
    # print(len(_shorcut_layers))

    return _layers_name, _shorcut_layers, _bn_layers, _short_bn_layers


def remove_dead_neurons_weights(params, neurons_state, frozen_layer_lists, opt_state=None, state=None):
    """Given the current params and the neuron state (True if dead) returns a
     filtered params dict (and its associated optimizer state) with dead weights
      removed and the new size of the layers (that is, # of conv filters or # of
       neurons in fully connected layer, etc.)"""
    neurons_state = jax.tree_map(jnp.logical_not, neurons_state)
    filtered_params = copy.deepcopy(params)

    # print(jax.tree_map(jax.numpy.shape, filtered_params))
    flag_opt = False
    if opt_state:
        if len(opt_state) == 1:
            opt_state = opt_state[0]
            flag_opt = True
        # field_names = [field.name for field in fields(opt_state[0])]
        field_names = list(opt_state[0]._fields)
        if 'count' in field_names:
            field_names.remove('count')
        filter_in_opt_state = copy.deepcopy([getattr(opt_state[0], field) for field in field_names])
    if state:
        filtered_state = copy.deepcopy(state)
        state_names = ["/~/var_ema", "/~/mean_ema"]
        _identity_state_name = [name for name in state.keys() if "identity" in name]
        _identity_state_name.sort()
        # print(_identity_state_name)

    _layers_name, _shortcut_layers, _bn_layers, _short_bn_layers = frozen_layer_lists
    # Flag to check if there is shortcut bn layers (yes if ResnetBlockV1, no if V2)
    shcut_bn_flag = len(_short_bn_layers) > 0
    # print(len(_layers_name))
    # print(len(_shorcut_layers))
    # print(len(neurons_state))
    # print(_layers_name)
    # print("########")
    # print()
    # print(_shorcut_layers)
    shortcut_counter = 0
    identity_skip_counter = -1
    for i, layer in enumerate(_layers_name[:-1]):
        # print(i, layer)
        # print(neurons_state[i].shape)
        if ("conv_1" in layer) and (shortcut_counter < len(_shortcut_layers)):
            shortcut_layer = _shortcut_layers[shortcut_counter]
            location = layer.index("block")
            if layer[location:location + 10] == _shortcut_layers[shortcut_counter][location:location + 10]:
                in_skip_flag = True
                out_skip_flag = False
                in_shortcut_flag = False
                out_shortcut_flag = True
                shortcut_layer = _shortcut_layers[shortcut_counter]
                if shcut_bn_flag:
                    shortcut_bn = _short_bn_layers[shortcut_counter]
                shortcut_counter += 1
                identity_skip_counter += 1

            elif layer[location:location + 10] == "block_v1/~" or layer[location:location + 10] == "block_v2/~":
                in_skip_flag = True
                out_skip_flag = True
                in_shortcut_flag = False
                out_shortcut_flag = False
            else:
                in_skip_flag = False
                out_skip_flag = True
                in_shortcut_flag = True
                out_shortcut_flag = False

        elif "initial_conv" in layer:
            in_skip_flag = True
            out_skip_flag = False
            in_shortcut_flag = False
            out_shortcut_flag = False
            identity_skip_counter += 1
        elif "_7/~/conv_1" in layer:
            in_skip_flag = False
            out_skip_flag = True
            in_shortcut_flag = False
            out_shortcut_flag = False
        else:
            in_skip_flag = False
            out_skip_flag = False
            in_shortcut_flag = False
            out_shortcut_flag = False
        for dict_key in filtered_params[layer].keys():
            # print(filtered_params[layer][dict_key].shape)
            # print(layer)
            # print(dict_key)
            # print(neurons_state[i].shape)
            filtered_params[layer][dict_key] = filtered_params[layer][dict_key][..., neurons_state[i]]
            # print(layer, jax.tree_map(jnp.shape, filtered_params[layer]))
            if out_shortcut_flag:
                filtered_params[shortcut_layer][dict_key] = filtered_params[shortcut_layer][dict_key][..., neurons_state[i]]
                # print(shortcut_layer, jax.tree_map(jnp.shape, filtered_params[shortcut_layer]))
            if opt_state:
                for j, field in enumerate(filter_in_opt_state):
                    # print(field)
                    filter_in_opt_state[j][layer][dict_key] = field[layer][dict_key][..., neurons_state[i]]
                    if out_shortcut_flag:
                        filter_in_opt_state[j][shortcut_layer][dict_key] = field[shortcut_layer][dict_key][..., neurons_state[i]]
        if out_shortcut_flag and shcut_bn_flag:
            for d_key in filtered_params[shortcut_bn].keys():
                filtered_params[shortcut_bn][d_key] = filtered_params[shortcut_bn][d_key][..., neurons_state[i]]
            if opt_state:
                for d_key in filtered_params[shortcut_bn].keys():
                    for j, field in enumerate(filter_in_opt_state):
                        filter_in_opt_state[j][shortcut_bn][d_key] = field[shortcut_bn][d_key][..., neurons_state[i]]
            # print(shortcut_bn, jax.tree_map(jnp.shape, filtered_params[shortcut_bn]))
        if state:
            for nme in state_names:
                filtered_state[_bn_layers[i] + nme]["average"] = filtered_state[_bn_layers[i] + nme]["average"][
                    ..., neurons_state[i]]
                filtered_state[_bn_layers[i] + nme]["hidden"] = filtered_state[_bn_layers[i] + nme]["hidden"][
                    ..., neurons_state[i]]
                if out_shortcut_flag and shcut_bn_flag:
                    filtered_state[shortcut_bn + nme]["average"] = filtered_state[shortcut_bn + nme]["average"][
                        ..., neurons_state[i]]
                    filtered_state[shortcut_bn + nme]["hidden"] = filtered_state[shortcut_bn + nme]["hidden"][
                        ..., neurons_state[i]]
            if out_skip_flag:
                ind = identity_skip_counter
                if layer[location:location + 10] == "block_v1/~" or layer[location:location + 10] == "block_v2/~":
                    identity_skip_counter += 1
                # print("out pruning")
                # print(ind)
                # print(layer)
                # print(_identity_state_name[ind])
                # print()
                filtered_state[_identity_state_name[ind]]["w"] = filtered_state[_identity_state_name[ind]]["w"][
                    ..., neurons_state[i]]
        for dict_key in filtered_params[_bn_layers[i]].keys():
            # print(layer)
            # print(_bn_layers[i])
            filtered_params[_bn_layers[i]][dict_key] = filtered_params[_bn_layers[i]][dict_key][..., neurons_state[i]]
            if opt_state:
                for j, field in enumerate(filter_in_opt_state):
                    # print(field)
                    filter_in_opt_state[j][_bn_layers[i]][dict_key] = field[_bn_layers[i]][dict_key][..., neurons_state[i]]
        # print(_bn_layers[i], jax.tree_map(jnp.shape, filtered_params[_bn_layers[i]]))

        # for dict_key in filtered_params[layers_name[i+1]].keys():
        #     print(neurons_state[i].shape)
        #     print(filtered_params[layers_name[i+1]][dict_key].shape)
        upscaling_factor = neurons_state[i].size
        if upscaling_factor == 0:
            to_repeat = 1
        else:
            to_repeat = filtered_params[_layers_name[i+1]]['w'].shape[-2] // upscaling_factor
        if to_repeat > 1:
            current_state = jnp.repeat(neurons_state[i].reshape(1, -1), to_repeat, axis=0).flatten()
            # print(neurons_state[i])
            # print(current_state)
        else:
            current_state = neurons_state[i]
        filtered_params[_layers_name[i+1]]['w'] = filtered_params[_layers_name[i+1]]['w'][..., current_state, :]
        if in_shortcut_flag:
            filtered_params[shortcut_layer]['w'] = filtered_params[shortcut_layer]['w'][..., current_state, :]
        # else:
        if in_skip_flag:
            ind = identity_skip_counter
            # print("in pruning")
            # print(ind)
            # print(layer)
            # print(_identity_state_name[ind])
            # print()
            filtered_state[_identity_state_name[ind]]["w"] = filtered_state[_identity_state_name[ind]]["w"][...,
                                                             current_state, :]
        if opt_state:
            for j, field in enumerate(filter_in_opt_state):
                # print()
                # print(layer + "-->", _layers_name[i+1])
                # print()
                filter_in_opt_state[j][_layers_name[i + 1]]['w'] = field[_layers_name[i + 1]]['w'][...,
                                                                  current_state, :]
                if in_shortcut_flag:
                    filter_in_opt_state[j][shortcut_layer]['w'] = field[shortcut_layer]['w'][
                                                                       ...,
                                                                       current_state, :]

    if opt_state:
        cp_state = copy.copy(opt_state)
        filtered_opt_state = cp_state[0]
        empty_state = cp_state[1:]
        for j, field in enumerate(field_names):
            # setattr(filtered_opt_state, field, filter_in_opt_state[j])
            filtered_opt_state = filtered_opt_state._replace(**{field: filter_in_opt_state[j]})

        if flag_opt:
            new_opt_state = ((filtered_opt_state,) + empty_state,)
        else:
            new_opt_state = (filtered_opt_state,) + empty_state

    new_sizes = [int(jnp.sum(layer)) for layer in neurons_state]
    # print(list(filtered_params.keys()))

    if opt_state:
        if state:
            return filtered_params, new_opt_state, filtered_state, tuple(new_sizes)
    else:
        if state:
            return filtered_params, filtered_state, tuple(new_sizes)
        else:
            return filtered_params, tuple(new_sizes)


# def rebuild_model(new_architecture, sizes):
#     """Rebuild a model with less params to accommodate the filtering of dead neurons"""
#     new_architecture = architecture_choice[exp_architecture](sizes, 10)  # TODO accomate more than 10 classes...
#     new_net = build_models(new_architecture)
#     return new_net


def neuron_state_from_params_magnitude(params, eps):
    """ Return neuron state (to detect 'quasi-dead') according to the mean size of
    the parameters of a neuron"""
    neuron_state = []
    for key in list(params.keys())[:-1]:
        neuron_means = jax.tree_leaves(jax.tree_map(abs_mean_except_last_dim, params[key]))
        neuron_means = sum(neuron_means) / len(neuron_means)
        neuron_state.append(neuron_means <= eps)
    return neuron_state


def count_params(params):
    """ Return the total number of parameters in the params dict"""
    params_size = jax.tree_map(jnp.size, params)
    params_size, _ = ravel_pytree(params_size)
    return jax.device_get(jnp.sum(params_size))


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


def ce_loss_given_model(model, regularizer=None, reg_param=1e-4, classes=None, is_training=True, with_dropout=False,
                            mask_head=False, reduce_head_gap=False):
    """ Build the cross-entropy loss given the model"""
    if not classes:
        classes = 10

    if regularizer:
        assert regularizer in ["cdg_l2", "cdg_lasso", "l2", "lasso", "cdg_l2_act", "cdg_lasso_act"]
        if regularizer == "l2":
            def reg_fn(params, activations=None):
                return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "lasso":
            def reg_fn(params, activations=None):
                return sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "cdg_l2":
            def reg_fn(params, activations=None):
                return 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "cdg_lasso":
            def reg_fn(params, activations=None):
                return sum(jnp.sum(jnp.clip(p, 0)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "cdg_l2_act":
            def reg_fn(params, activations):
                return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(activations))
        if regularizer == "cdg_lasso_act":
            def reg_fn(params, activations):
                return 0.5 * sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(activations))
    else:
        def reg_fn(params, activations=None):
            return 0

    if is_training and with_dropout:
        @jax.jit
        def _loss(params: hk.Params, state: hk.State, batch: Batch, dropout_key: Any) -> Union[jnp.ndarray, Any]:
            next_dropout_key, rng = jax.random.split(dropout_key)
            (logits, activations), state = model.apply(params, state, rng, batch, return_activations=True, is_training=is_training)
            labels = jax.nn.one_hot(batch[1], classes)

            if mask_head:
                # softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits, where=(logits+labels) > 0, initial=0))
                softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits, where=jnp.sum(labels, axis=0) > 0, initial=0))
            else:
                softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            if reduce_head_gap:
                gap = jnp.max(logits, axis=0, where=jnp.sum(labels, axis=0) > 0, initial=0) - jnp.max(logits, axis=0)
                gap = jnp.sum(jnp.abs(gap)) / labels.shape[0]
            else:
                gap = 0

            loss = softmax_xent + reg_param * reg_fn(params, activations) + gap

            return loss, (state, next_dropout_key)

    else:
        if with_dropout:
            dropout_key = jax.random.PRNGKey(0)  # dropout rate is zero during death eval
            model_apply_fn = Partial(model.apply, rng=dropout_key)
        else:
            model_apply_fn = model.apply

        @jax.jit
        def _loss(params: hk.Params, state: hk.State, batch: Batch) -> Union[jnp.ndarray, Any]:
            (logits, activations), state = model_apply_fn(params, state, x=batch, return_activations=True, is_training=is_training)
            labels = jax.nn.one_hot(batch[1], classes)

            if mask_head:
                # softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits, where=(logits+labels) > 0, initial=0))
                softmax_xent = -jnp.sum(
                    labels * jax.nn.log_softmax(logits, where=jnp.sum(labels, axis=0) > 0, initial=0))
            else:
                softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            if reduce_head_gap:
                gap = jnp.max(logits, axis=0, where=jnp.sum(labels, axis=0) > 0, initial=0) - jnp.max(logits, axis=0)
                gap = jnp.sum(jnp.abs(gap)) / labels.shape[0]
            else:
                gap = 0

            loss = softmax_xent + reg_param * reg_fn(params, activations) + gap

            if is_training:
                return loss, state
            else:
                return loss

    return _loss


def mse_loss_given_model(model, regularizer=None, reg_param=1e-4, is_training=True):
    """ Build mean squared error loss given the model"""

    if regularizer:
        assert regularizer in ["cdg_l2", "cdg_lasso", "l2", "lasso", "cdg_l2_act", "cdg_lasso_act"]
        if regularizer == "l2":
            def reg_fn(params, activations=None):
                return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        if regularizer == "lasso":
            def reg_fn(params, activations=None):
                return sum(jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(params))
        if regularizer == "cdg_l2":
            def reg_fn(params, activations=None):
                return 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_leaves(params))
        if regularizer == "cdg_lasso":
            def reg_fn(params, activations=None):
                return sum(jnp.sum(jnp.clip(p, 0)) for p in jax.tree_leaves(params))
        if regularizer == "cdg_l2_act":
            def reg_fn(params, activations):
                return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(activations))
        if regularizer == "cdg_lasso_act":
            def reg_fn(params, activations):
                return 0.5 * sum(jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(activations))
    else:
        def reg_fn(params, activations=None):
            return 0

    @jax.jit
    def _loss(params: hk.Params, state: hk.State, batch: Batch) -> Union[jnp.ndarray, Any]:
        (outputs, activations), state = model.apply(params, state, x=batch, return_activations=True,
                                                   is_training=is_training)
        targets = batch[1]

        # calculate mse across the batch
        mse = jnp.mean(jnp.square(outputs-targets))

        loss = mse + reg_param * reg_fn(params, activations)

        if is_training:
            return loss, state
        else:
            return loss

    return _loss


def grad_normalisation_per_layer(param_leaf):
    var = jnp.var(param_leaf)
    return param_leaf/jnp.sqrt(var+1)


def update_given_loss_and_optimizer(loss, optimizer, noise=False, noise_imp=(1, 1), noise_live_only=False,
                                    norm_grad=False, with_dropout=False, return_grad=False):
    """Learning rule (stochastic gradient descent)."""

    if with_dropout:
        assert not return_grad, 'return_grad option not coded yet with dropout'

        @jax.jit
        def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _drop_key) -> Tuple[
            hk.Params, Any, OptState, jax.random.PRNGKeyArray]:
            grads, (new_state, next_drop_key) = jax.grad(loss, has_aux=True)(_params, _state, _batch, _drop_key)
            if norm_grad:
                grads = jax.tree_map(grad_normalisation_per_layer, grads)
            updates, _opt_state = optimizer.update(grads, _opt_state, _params)
            new_params = optax.apply_updates(_params, updates)
            return new_params, new_state, _opt_state, next_drop_key

    else:
        if not noise:
            if return_grad:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                            _batch: Batch) -> Tuple[dict, hk.Params, Any, OptState]:
                    grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch)
                    if norm_grad:
                        grads = jax.tree_map(grad_normalisation_per_layer, grads)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    new_params = optax.apply_updates(_params, updates)
                    return grads, new_params, new_state, _opt_state

            else:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                            _batch: Batch) -> Tuple[hk.Params, Any, OptState]:
                    grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch)
                    if norm_grad:
                        grads = jax.tree_map(grad_normalisation_per_layer, grads)
                    # try:
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    # except:
                    #     grad_dict = jax.tree_map(jnp.shape, grads)
                    #     mu_dict = jax.tree_map(jnp.shape, grads)
                    #     print(jax.tree_map(jnp.shape, grads))
                    #     print(jax.tree_map(jnp.shape, _opt_state))
                    #     print(jax.tree_map(jnp.equal, grad_dict, mu_dict))
                    #     raise SystemExit
                    new_params = optax.apply_updates(_params, updates)
                    return new_params, new_state, _opt_state
        else:
            a, b = noise_imp
            assert not return_grad, 'return_grad option not coded yet with noisy grad'
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
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    flat_updates, _ = ravel_pytree(updates)
                    noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)
                    return new_params, new_state, _opt_state, next_key
            else:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                            _key: Any) -> Tuple[hk.Params, Any, OptState, Any]:
                    grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    key, next_key = jax.random.split(_key)
                    flat_updates, unravel_fn = ravel_pytree(updates)
                    added_noise = _var*jax.random.normal(key, shape=flat_updates.shape)
                    noisy_updates = unravel_fn(a*flat_updates + b*added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)
                    return new_params, new_state, _opt_state, next_key

    return _update


def get_mask_update_fn(loss, optimizer):
    """ Return the update function, but taking into account a mask for neurons that we want to freeze the weights"""

    @jax.jit
    def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, grad_mask: hk.Params, zero_grad: hk.Params,
                _batch: Batch) -> Tuple[hk.Params, Any, OptState]:
        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch)
        grads = reinitialize_excluding_head(grad_mask, grads, zero_grad)
        updates, _opt_state = optimizer.update(grads, _opt_state, _params)
        new_params = optax.apply_updates(_params, updates)
        return new_params, new_state, _opt_state

    return _update


def vmap_axes_mapping(dict_container):
    """Utility function to indicate to vmap that in_axes={key:0} for all keys in dict_container"""
    def _map_over(v):
        return 0  # Need to vmap over all arrays in the container
    return jax.tree_map(_map_over, dict_container)


@Partial(jax.jit, static_argnames='_len')
def dict_split(container, _len=2):
    """Split back the containers into their specific components, returning them as a tuple"""
    if not container:  # Empty case
        return (container, ) * _len
    treedef = jax.tree_structure(container)
    leaves = jax.tree_leaves(container)
    _to = leaves[0].shape[0]

    leaves = jax.tree_map(Partial(jnp.split, indices_or_sections=_to), leaves)
    leaves = jax.tree_map(Partial(jnp.squeeze, axis=0), leaves)
    splitted_dict = tuple([treedef.unflatten(list(z)) for z in zip(*leaves)])
    return splitted_dict


##############################
# Dataset loading utilities
##############################
# Prefixed data augmentation when desired
augment_tf_dataset = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.ZeroPadding2D(4, data_format="channels_last"),
    tf.keras.layers.RandomCrop(32, 32)
])


def resize_tf_dataset(images, labels, dataset):
    """Final data augmentation step for cifar10, consisting of a resizing to 64x64"""
    if dataset == 'cifar10':
        # Resize images to 64x64
        images = tf.image.resize(images, [64, 64])
    return images, labels


def map_noisy_labels(sample_from):
    def _map_noisy_labels(image, label):
        # sample_from = np.arange(num_classes-1)
        rdm_choice = np.random.choice(sample_from)
        # rdm_choice = label + tf.cast((label <= rdm_choice), label.dtype)
        rdm_choice = tf.cast(rdm_choice, label.dtype)
        return image, rdm_choice
    return _map_noisy_labels


def map_permuted_img(image, label):
    """ Map each image in the given dataset to a random permutation of its pixel"""
    _shape = image.shape
    image = tf.reshape(image, (-1, _shape[-1]))
    image = tf.random.shuffle(image)
    image = tf.reshape(image, _shape)
    return image, label


def map_gaussian_img(ds):
    """Map each image to a random guassian image sharing mean and variance of the original ds"""
    # get mean and variance
    init_state = np.float32(0), np.float32(0), np.float32(0)

    def running_mean(state, input):
        current_mean, sumsq, counter = state
        image, label = input
        sum_img = tf.cast(tf.math.reduce_mean(image, [0, 1]), tf.float32)
        current_mean = (counter * current_mean + sum_img) / (counter+1)
        sumsq += sum_img**2
        return current_mean, sumsq, counter+1

    avg, sumsq, n = ds.reduce(init_state, running_mean)
    var = sumsq/n - avg**2
    print(avg.dtype)
    print(var.dtype)

    def _map_gaussian_img(image, label):
        image = tf.random.normal(image.shape, avg, var)
        return tf.cast(image, tf.uint8), label

    return _map_gaussian_img


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


def standardize_img(image, label):
    image -= tf.math.reduce_mean(image, axis=[0, 1])
    image /= tf.math.reduce_std(image, axis=[0, 1])
    return image, label


def custom_normalize_img(image, label, dataset):
    assert dataset in ["mnist", 'cifar10', 'cifar100'], "need to implement normalization for others dataset"
    if dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]  # Mean for each channel (R, G, B)
        std = [0.2023, 0.1994, 0.2010]  # Standard deviation for each channel (R, G, B)
    elif dataset == "mnist":
        mean = [0.1307]  # Mean for each channel (R, G, B)
        std = [0.3081]  # Standard deviation for each channel (R, G, B)
    elif dataset == "cifar100":
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    else:
        raise SystemExit
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)

    image = tf.cast(image, dtype=tf.float32)
    image = (image - mean) / std
    return image, label


def load_tf_dataset(dataset: str, split: str, *, is_training: bool, batch_size: int,
                    other_bs: Optional[Iterable] = None,
                    subset: Optional[int] = None, transform: bool = True,
                    cardinality: bool = False, noisy_label: float = 0, permuted_img_ratio: float = 0,
                    gaussian_img_ratio: float = 0, data_augmentation: bool = False, normalize: bool = False):  # -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches.
    subset: If only want a subset, number of classes to build the subset from
    """
    def filter_fn(image, label):
        return tf.reduce_any(subset == int(label))

    if noisy_label or permuted_img_ratio or gaussian_img_ratio:
        assert (noisy_label >= 0) and (noisy_label <= 1), "noisy label ratio must be between 0 and 1"
        assert (permuted_img_ratio >= 0) and (permuted_img_ratio <= 1), "permuted_img ratio must be between 0 and 1"
        assert (gaussian_img_ratio >= 0) and (gaussian_img_ratio <= 1), "gaussian_img ratio must be between 0 and 1"
        noisy_ratio = max(noisy_label, permuted_img_ratio, gaussian_img_ratio)
        split1 = split + '[:' + str(int(noisy_ratio*100)) + '%]'
        split2 = split + '[' + str(int(noisy_ratio*100)) + '%:]'
        ds1, ds_info = tfds.load(dataset, split=split1, as_supervised=True, data_dir="./data", with_info=True)
        ds2 = tfds.load(dataset, split=split2, as_supervised=True, data_dir="./data")
        sample_from = np.arange(ds_info.features["label"].num_classes - 1)
        if subset is not None:
            ds1 = ds1.filter(filter_fn)  # Only take the randomly selected subset
            ds2 = ds2.filter(filter_fn)  # Only take the randomly selected subset
            sample_from = subset

        if noisy_label:
            ds1 = ds1.map(map_noisy_labels(sample_from=sample_from))
        elif permuted_img_ratio:  # TODO: Do not make randomized ds mutually exclusive?
            ds1 = ds1.map(map_permuted_img)
        elif gaussian_img_ratio:
            ds1 = ds1.map(map_gaussian_img(ds1))
        ds = ds1.concatenate(ds2)
    else:
        ds = tfds.load(dataset, split=split, as_supervised=True, data_dir="./data")
        if subset is not None:
            ds = ds.filter(filter_fn)  # Only take the randomly selected subset
    ds_size = int(ds.cardinality())
    # if subset is not None:
    #     # assert subset < 10, "subset must be smaller than 10"
    #     # indices = np.random.choice(10, subset, replace=False)
    #
    #     ds = ds.filter(filter_fn)  # Only take the randomly selected subset

    ds = ds.map(interval_zero_one)
    if normalize:
        ds = ds.map(Partial(custom_normalize_img, dataset=dataset))
    # ds = ds.cache().repeat()
    # if is_training:
    #     # if subset is not None:
    #     #     ds = ds.cache().repeat()
    #     ds = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=False)
    #     if data_augmentation:
    #         ds = ds.map(lambda x, y: (augment_tf_dataset(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    #     # ds = ds.take(batch_size).cache().repeat()
    # ds = ds.cache()
    ds = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
    ds = ds.repeat()
    if other_bs:
        if is_training:  # Only ds1 takes into account 'is_training' flag
            # ds1 = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
            ds1 = ds.batch(batch_size)
            if data_augmentation:
                ds1 = ds1.map(lambda x, y: (augment_tf_dataset(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds1 = ds.batch(batch_size)
        if data_augmentation:  # Resize as well during test if data augmentation
            ds1 = ds1.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
        ds1 = ds1.prefetch(tf.data.AUTOTUNE)
        all_ds = [ds1]
        for bs in other_bs:
            ds2 = ds.batch(bs)
            if data_augmentation:  # Resize as well for proper evaluation if data augmented
                ds2 = ds2.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
            ds2 = ds2.prefetch(tf.data.AUTOTUNE)
            all_ds.append(ds2)

        if (subset is not None) and transform:
            tf_iterators = tuple([tf_compatibility_iterator(iter(tfds.as_numpy(_ds)), subset) for _ds in all_ds])
        else:
            tf_iterators = tuple([iter(tfds.as_numpy(_ds)) for _ds in all_ds])
        if cardinality:
            return (ds_size, ) + tf_iterators
        else:
            return tf_iterators
    else:
        if is_training:
            # ds = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
            ds = ds.batch(batch_size)
            if data_augmentation:
                ds = ds.map(lambda x, y: (augment_tf_dataset(x), y), num_parallel_calls=tf.data.AUTOTUNE)
                ds = ds.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.batch(batch_size)
            if data_augmentation:
                ds = ds.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        if (subset is not None) and transform:
            tf_iterator = tf_compatibility_iterator(iter(tfds.as_numpy(ds)), subset)  # Reorder the labels, ex: 1,5,7 -> 0,1,2
            if cardinality:
                return ds_size, tf_iterator
            else:
                return tf_iterator
        else:
            if cardinality:
                return ds_size, iter(tfds.as_numpy(ds))
            else:
                return iter(tfds.as_numpy(ds))


def load_mnist_tf(split: str, is_training, batch_size, other_bs=None, subset=None, transform=True, cardinality=False,
                  noisy_label=0, permuted_img_ratio=0, gaussian_img_ratio=0, augment_dataset=False, normalize: bool = False):
    return load_tf_dataset("mnist", split=split, is_training=is_training, batch_size=batch_size,
                           other_bs=other_bs, subset=subset, transform=transform, cardinality=cardinality,
                           noisy_label=noisy_label, permuted_img_ratio=permuted_img_ratio,
                           gaussian_img_ratio=gaussian_img_ratio, data_augmentation=augment_dataset, normalize=normalize)


def load_cifar10_tf(split: str, is_training, batch_size, other_bs=None, subset=None, transform=True, cardinality=False,
                    noisy_label=0, permuted_img_ratio=0, gaussian_img_ratio=0, augment_dataset=False, normalize: bool = False):
    return load_tf_dataset("cifar10", split=split, is_training=is_training, batch_size=batch_size, other_bs=other_bs,
                           subset=subset, transform=transform, cardinality=cardinality, noisy_label=noisy_label,
                           permuted_img_ratio=permuted_img_ratio, gaussian_img_ratio=gaussian_img_ratio,
                           data_augmentation=augment_dataset, normalize=normalize)


def load_cifar100_tf(split: str, is_training, batch_size, other_bs=None, subset=None, transform=True, cardinality=False,
                     noisy_label=0, permuted_img_ratio=0, gaussian_img_ratio=0, augment_dataset=False, normalize: bool = False):
    return load_tf_dataset("cifar100", split=split, is_training=is_training, batch_size=batch_size, other_bs=other_bs,
                           subset=subset, transform=transform, cardinality=cardinality, noisy_label=noisy_label,
                           permuted_img_ratio=permuted_img_ratio, gaussian_img_ratio=gaussian_img_ratio,
                           data_augmentation=augment_dataset, normalize=normalize)


def load_fashion_mnist_tf(split: str, is_training, batch_size, other_bs=None, subset=None, transform=True,
                          cardinality=False, noisy_label=0, permuted_img_ratio=0, gaussian_img_ratio=0,
                          augment_dataset=False, normalize: bool = False):
    return load_tf_dataset("fashion_mnist", split=split, is_training=is_training, batch_size=batch_size,
                           other_bs=other_bs, subset=subset, transform=transform, cardinality=cardinality,
                           noisy_label=noisy_label, permuted_img_ratio=permuted_img_ratio,
                           gaussian_img_ratio=gaussian_img_ratio, data_augmentation=augment_dataset, normalize=normalize)


# Pytorch dataloader # TODO: deprecated; should remove!!
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
# lr scheduler utilities
##############################
def constant_schedule(training_steps, base_lr, final_lr, decay_steps):
    return optax.constant_schedule(base_lr)


def piecewise_constant_schedule(training_steps, base_lr, final_lr, decay_steps):
    scaling_factor = (final_lr/base_lr)**(1/(decay_steps-1))
    bound_dict = {int(training_steps/decay_steps*i): scaling_factor for i in range(1, decay_steps)}
    return optax.piecewise_constant_schedule(base_lr, bound_dict)


def cosine_decay(training_steps, base_lr, final_lr, decay_steps):
    alpha_val = final_lr/base_lr
    return optax.cosine_decay_schedule(base_lr, training_steps, alpha_val)


def one_cycle_schedule(training_steps, base_lr, final_lr, decay_steps):
    return optax.cosine_onecycle_schedule(training_steps, base_lr)


##############################
# Modified optax optimizer
##############################
AddDecayedWeightsState = base.EmptyState
ScalarOrSchedule = Union[float, base.Schedule]


def add_cdg_decayed_weights(
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
    """Cdg variant of optax weight_decay
    Add parameter where positive weights are scaled by `weight_decay`

    Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

    Returns:
    A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return AddDecayedWeightsState()

    def cdg_mask(params):
        return jax.tree_util.tree_map(
            lambda x: (x > 0)*x, params)

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, cdg_mask(params))
        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return wrappers.masked(
            base.GradientTransformation(init_fn, update_fn), mask)
    return base.GradientTransformation(init_fn, update_fn)


def adamw_cdg(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """Cdg version of optax Adam with weight decay regularization.

    AdamW uses weight decay to regularize learning towards small weights, as
    this leads to better generalization. In SGD you can also use L2 regularization
    to implement this as an additive loss term, however L2 regularization
    does not behave as intended for adaptive gradient algorithms such as Adam.

    References:
    Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

    Args:
        learning_rate: A fixed global scaling factor.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root
          (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
          in RMSProp), to avoid dividing by zero when rescaling. This is needed for
          instance when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
          `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay: Strength of the weight decay regularization. Note that this
          weight decay is multiplied with the learning rate. This is consistent
          with other frameworks such as PyTorch, but different from
          (Loshchilov et al, 2019) where the weight decay is only multiplied with
          the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
          or a Callable that returns such a pytree given the params/updates.
          The leaves should be booleans, `True` for leaves/subtrees you want to
          apply the weight decay to, and `False` for those you want to skip. Note
          that the Adam gradient transformations are applied to all parameters.

    Returns:
    The corresponding `GradientTransformation`.
    """
    return combine.chain(
      transform.scale_by_adam(
          b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      add_cdg_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
    )


##############################
# Baseline pruning
##############################
# Implementation of some baseline methods for pruning, along with their utilities
# For comparison purpose w/r to our method
def mask_layer_filters(params, layers, prune_ratio):
    """Gradually prune the filters in a layer given a pruning ratio. Compatible with fc and conv layers

    Args:
        params: The params to prune, a dictionnary of type hk.params
        layers: The layers to prune, a list where first one is a conv layer and then associate bn layers if any
        prune_ratio: The ratio of filters to prune in the given layer
        """

    main_layer = layers[0]
    main_params = params[main_layer]

    _axis = tuple(range(len(jnp.shape(main_params["w"]))-1))
    norms = jnp.sum(jnp.abs(main_params["w"]), axis=_axis)
    add_norms = [jnp.abs(main_params[key]) for key in main_params.keys() if key != "w"]  # Adding bias and other to norm if any
    norms += sum(add_norms)
    num_filters_to_prune = int(prune_ratio*norms.size)

    #smallest_filter_indices = jnp.argpartition(norms, num_filters_to_prune)[:num_filters_to_prune]  # not implemented in cluster jacx version

    _, smallest_filter_indices = jax.lax.top_k(-norms, num_filters_to_prune)

    # Generate a mask, (0 if pruned) for the layer
    all_masks = {}
    layer_masks = {}
    for key in main_params.keys():
        mask = jnp.ones_like(main_params[key])
        mask = mask.at[..., smallest_filter_indices].set(0)
        layer_masks[key] = mask

    all_masks[main_layer] = layer_masks

    # Generate the mask for additional layers
    for layer in layers[1:]:
        layer_masks = {}
        layer_param = params[layer]
        for key in layer_param.keys():
            _mask = jnp.ones_like(layer_param[key])
            _mask = _mask.at[..., smallest_filter_indices].set(0)
            layer_masks[key] = _mask
        all_masks[layer] = layer_masks

    return all_masks, smallest_filter_indices


def mask_next_layer_filters(params, next_layers, previous_smallest_filter_indices):
    """Remove the kernel weights associated to filters that were pruned in the previous layer

    Args:
        params: The params to prune, a dictionnary of type hk.params
        next_layers: The layers to prune, a list where first one is a conv layer and then associate bn layers if any
        previous_smallest_filter_indices: The index of the kernel weights to prune
    """

    all_masks = {}
    for layer in next_layers:
        layer_masks = {}
        layer_param = params[layer]
        for key in layer_param.keys():
            if key == "w":
                _mask = jnp.ones_like(layer_param["w"])
                _mask = _mask.at[..., previous_smallest_filter_indices, :].set(0)
                layer_masks["w"] = _mask
            else:
                layer_masks[key] = jnp.ones_like(layer_param[key])
        all_masks[layer] = layer_masks

    return all_masks


def prune_params(params, ordered_layers, layer_index, prune_ratio):
    layers = ordered_layers[layer_index]
    pruned_params = copy.copy(params)

    # prune the currently considered layer
    pruning_masks, smallest_filter_indices = mask_layer_filters(params, layers, prune_ratio)
    for key in pruning_masks.keys():
        pruned_params[key] = jax.tree_map(jnp.multiply, params[key], pruning_masks[key])
    # prune next layer dependent kernel weights, if any
    if layer_index < len(ordered_layers)-1:
        next_layers = ordered_layers[layer_index+1]
        pruning_masks = mask_next_layer_filters(params, next_layers, smallest_filter_indices)
        for key in pruning_masks.keys():
            pruned_params[key] = jax.tree_map(jnp.multiply, params[key], pruning_masks[key])

    return pruned_params


def prune_until_perf_decay(ref_perf, allowed_decay, evaluate_fn, greedy: bool, params, ordered_layers, prune_ratio_step, starting_ratios):
    assert allowed_decay >= 0, "allowed_decay must be positive"
    pruned_params = copy.copy(params)
    for i in range(len(ordered_layers)):
        prune_ratio = starting_ratios[i]
        perf_decay = 0
        while perf_decay < allowed_decay and (prune_ratio+prune_ratio_step) < 1:
            prune_ratio += prune_ratio_step
            _pruned_params = prune_params(params, ordered_layers, i, prune_ratio)
            curr_perf = evaluate_fn(_pruned_params)
            perf_decay = ref_perf - curr_perf
            if perf_decay < allowed_decay:
                for layer in ordered_layers[i]:
                    pruned_params[layer] = _pruned_params[layer]
                if i < len(ordered_layers)-1:
                    for layer in ordered_layers[i + 1]:
                        pruned_params[layer] = _pruned_params[layer]
        if greedy and i < len(ordered_layers)-1:
            for layer in ordered_layers[i+1]:
                params[layer] = pruned_params[layer]
        starting_ratios[i] = prune_ratio - prune_ratio_step

    return pruned_params, starting_ratios


def extract_ordered_layers(params):
    """Extract layer list. Based on `extract_layer_lists`. Here however, we group together
     layers in block for pruning"""
    layers_name = list(params.keys())
    ordered_layers = []
    curr_block = [layers_name[0]]
    for layer in layers_name[1:]:
        if "conv" in layer or 'linear' in layer or "logits" in layer:
            ordered_layers.append(curr_block)
            curr_block = [layer]
        else:
            curr_block.append(layer)

    return ordered_layers


##############################
# Varia
##############################
def get_total_neurons(architecture, sizes):
    if 'mlp_3' in architecture:
        if type(sizes) == int:  # Size can be specified with 1 arg, an int
            sizes = [sizes, sizes * 3]
    elif architecture == 'conv_3_2':
        if len(sizes) == 2:  # Size can be specified with 2 args
            sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], sizes[1]]
    elif (architecture == 'conv_4_2') or (architecture == 'conv_4_2_ln'):
        if len(sizes) == 2:  # Size can be specified with 2 args
            sizes = [sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]
    elif architecture == 'conv_6_2':
        if len(sizes) == 2:  # Size can be specified with 2 args
            sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], sizes[1]]
    elif architecture == "resnet18" or architecture == "resnet18_v2":
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
                     4 * sizes,
                     ]
    else:
        raise NotImplementedError("get_size not implemented for current architecture")

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


def add_comma_in_str(string: str):
    """ Helper fn to format string from hydra before literal eval"""
    string = string.replace("mnist", "'mnist'")
    string = string.replace("fashion mnist", "'fashion mnist'")
    string = string.replace("cifar10", "'cifar10'")
    string = string.replace("cifar100", "'cifar100'")

    return string


def identity_fn(x):
    """ Simple identity fn to use as an activation"""
    return x


def threlu(x):
    """Tanh activation followed by a ReLu. Intended to be used with LayerNorm"""
    return jax.nn.relu(jax.nn.tanh(x))


# @jax.jit  # Call once, no need to jit
def mean_var_over_pytree_list(pytree_list):
    """ Takes a list of pytree sharing the same structure and return their average stored in the same pytree"""

    _, unflatten_fn = ravel_pytree(pytree_list[0])
    tree_ = [ravel_pytree(tree)[0] for tree in pytree_list]
    tree_ = jnp.stack(tree_)
    tree_avg = jnp.mean(tree_, axis=0)
    tree_var = jnp.var(tree_, axis=0)

    return unflatten_fn(tree_avg), unflatten_fn(tree_var)


def concatenate_bias_to_weights(params_pytree):
    layers = list(params_pytree.keys())
    neurons_vec_dict = {}
    for layer in layers:
        # weight_type = tuple(params_pytree[layer].keys())  # Usually, 'w' and 'b'
        # print([params_pytree[layer][w_b].shape for w_b in weight_type])
        neurons_vectors = jnp.vstack([params_pytree[layer][w_b] for w_b in ('w', 'b')])#, axis=0)
        neurons_vec_dict[layer] = [neurons_vectors[:, i] for i in range(neurons_vectors.shape[1])]

    return neurons_vec_dict


def sequential_ds(classes, kept_classes):
    """ Build an iterator that sequentially splits the total number of classes."""
    indices_sequence = np.arange(classes)
    np.random.shuffle(indices_sequence)

    split = np.arange(kept_classes, classes, kept_classes)
    indices_sequence = np.split(indices_sequence, split)

    return cycle(indices_sequence)
