import os
import copy
from typing import Any, Generator, Mapping, Optional, Tuple, TypedDict
from types import FrameType
import dataclasses
from dataclasses import fields

import haiku as hk
import jax
import jax.numpy as jnp
import numpy
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_models as tfm
# from tensorflow_models.vision.ops.augment import RandAugment
import chex
import pickle
import signal
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset, Subset
# from torchvision import datasets, transforms
from typing import Union, Tuple, List, Callable, Sequence
from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree
from collections import OrderedDict
from collections.abc import Iterable
from itertools import cycle
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig

import utils.scores as scr
from utils.augment.augment import RandAugment
from utils.augment.augment import MixupAndCutmix

# from haiku._src.dot import to_graph
# import networkx as nx
# from networkx.drawing.nx_agraph import from_agraph
# import pygraphviz as pgv

from optax._src import base
from optax._src import wrappers
from optax._src import combine
from optax._src import transform
from optax._src import numerics
from optax._src.alias import _scale_by_learning_rate
from jaxpruner import base_updater

import psutil
import sys
import gc

# from ffcv.writer import DatasetWriter
# from ffcv.fields import IntField, RGBImageField
# from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
# from ffcv.loader import Loader, OrderOption
# from ffcv.pipeline.operation import Operation
# from ffcv.transforms import RandomHorizontalFlip, Cutout, \
#     RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
# from ffcv.transforms.common import Squeeze

OptState = Any
Batch = Mapping[int, np.ndarray]
BaseUpdater = base_updater.BaseUpdater


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

        # def scan_death_check(params, state, batch_it, epsilon=0): # TODO: Can't use scan -> cache explode with iter
        #     def update(running_vals, _):
        #         prev_dead, running_max, running_mean, running_var, running_count, N = running_vals
        #         activations, dead_neurons = death_check_fn(params, state, next(batch_it), epsilon)
        #         running_max = update_running_max(activations, running_max)
        #         running_mean = update_running_mean(activations, running_mean)
        #         running_var = update_running_var(activations, running_var)
        #         running_count = update_running_count(activations, running_count)
        #         dead_neurons = jax.tree_map(sum_dead_neurons, prev_dead, dead_neurons)
        #         return (dead_neurons, running_max, running_mean, running_var, running_count, N + 1), None
        #     activations, previous_dead = death_check_fn(params, state, next(batch_it), epsilon)
        #     running_max = jax.tree_map(Partial(jnp.amax, axis=0), activations)
        #     running_mean = jax.tree_map(Partial(jnp.mean, axis=0), activations)
        #     running_var = jax.tree_map(Partial(jnp.var, axis=0), activations)
        #     running_count = count_activations_occurrence(activations)
        #     N = 1
        #
        #     (dead_neurons, running_max, running_mean, running_var, running_count, N), _ = jax.lax.scan(
        #         update, (previous_dead, running_max, running_mean, running_var, running_count, N),
        #         None, scan_len - 1)
        #
        #     return (running_max, jax.tree_map(lambda x: x / N, running_mean),
        #             running_count, jax.tree_map(lambda x: x / N, running_var)), dead_neurons
        #
        # return scan_death_check
    else:
        def scan_death_check(params, state, batch_it, epsilon=0):
            previous_dead = death_check_fn(params, state, next(batch_it), epsilon)
            for i in range(scan_len-1):
                dead_neurons = death_check_fn(params, state, next(batch_it), epsilon)
                previous_dead = jax.tree_map(sum_dead_neurons, previous_dead, dead_neurons)

            return previous_dead

        # def scan_death_check(params, state, batch_it, epsilon=0): # TODO: Can't use scan -> cache explode with iter
        #     previous_dead = death_check_fn(params, state, next(batch_it), epsilon)
        #
        #     def scan_fn(prev_dead, _):
        #         dead_neurons = death_check_fn(params, state, next(batch_it), epsilon)
        #         return jax.tree_map(sum_dead_neurons, prev_dead, dead_neurons), None
        #
        #     dead_neurons, _ = jax.lax.scan(scan_fn, previous_dead, None, scan_len - 1)
        #     return dead_neurons

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


# def reinitialize_dead_neurons(neuron_states, old_params, new_params):
#     # neuron_states = [jnp.logical_not(state) for state in neuron_states]
#     # neuron_states = jax.tree_map(jnp.logical_not, neuron_states)
#     print(print(jax.tree_map(jnp.shape, neuron_states.state())))
#     neuron_states = [jnp.logical_not(state) for layer, state in neuron_states.items() if 'activation' in layer]
#     print(jax.tree_map(jnp.shape, neuron_states))
#     return _reinitialize_dead_neurons(neuron_states, old_params, new_params)


# @jax.jit
def reinitialize_dead_neurons(acti_map, neuron_states, old_params, new_params):
    """ Given the activations value for the whole training set, build a mask that is used for reinitialization
      neurons_state: neuron states (either 0 or 1) post-activation. Will be 1 if y <=  0
      old_params: current parameters value
      new_params: new parameters' dict to pick from weights being reinitialized"""
    for layer in old_params.keys():
        if acti_map[layer]['following'] is not None:
            neuron_state = neuron_states[acti_map[layer]['following']]
            for weight_type in list(old_params[layer].keys()):  # Usually, 'w' and 'b'
                old_params[layer][weight_type] = old_params[layer][weight_type] * jnp.logical_not(neuron_state)
        kernel_param = 'w'
        if acti_map[layer]['preceding'] is not None:
            neuron_state = neuron_states[acti_map[layer]['preceding']]
            old_params[layer][kernel_param] = old_params[layer][kernel_param] * jnp.logical_not(neuron_state).reshape(-1, 1)
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
    filtered_params = jax_deep_copy(params)

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
        filter_in_opt_state = jax_deep_copy([getattr(opt_state[0], field) for field in field_names])
    if state:
        filtered_state = jax_deep_copy(state)
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
                if i in list(range(len(_bn_layers))):
                    filtered_state[_bn_layers[i] + nme]["average"] = filtered_state[_bn_layers[i] + nme]["average"][
                        ..., neurons_state[i]]
                    filtered_state[_bn_layers[i] + nme]["hidden"] = filtered_state[_bn_layers[i] + nme]["hidden"][
                        ..., neurons_state[i]]
                if len(_short_bn_layers)>0:
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
        if i in list(range(len(_bn_layers))):
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
            # print(_layers_name[i])
            # print(_layers_name[i + 1])
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
        cp_state = jax_deep_copy(opt_state)
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
            return filtered_params, new_opt_state, {}, tuple(new_sizes)
    else:
        if state:
            return filtered_params, filtered_state, tuple(new_sizes)
        else:
            return filtered_params, {}, tuple(new_sizes)


class NeuronStates(OrderedDict):
    def __init__(self, keys, activations_list=None):
        # Initialize the neuron state dictionary with all values set to None.
        # Take as input the state keys, which is also ordered.
        keys = [s for s in keys if "activation_module" in s]
        super().__init__({key: None for key in keys})
        if activations_list:
            self.update_from_ordered_list(activations_list)

    def update_from_ordered_list(self, activations_list):
        _neuron_state = {key: value for key, value in zip(self.keys(), activations_list)}

        self.update(_neuron_state)

    def invert_state(self):
        return {key: jnp.logical_not(value) for key, value in self.items()}

    def state(self):
        return {key: value for key, value in self.items()}


def prune_params_state_optstate(params, activation_mapping, neurons_state_dict: OrderedDict, opt_state=None, state=None):
    """Given the current params and the neuron state mapping returns a
     filtered params dict (and its associated optimizer state) with dead weights
      removed and the new size of the layers (that is, # of conv filters or # of
       neurons in fully connected layer, etc.)"""
    filtered_params = jax_deep_copy(params)

    assert not all(value is None for value in list(
        neurons_state_dict.keys())), "neurons state dictionary needs to be updated before attempting to prune"

    if opt_state:
        flag_opt = False
        if len(opt_state) >= 1:  # If combined gradient transformation with optax.chain
            _opt_state = opt_state[-1]  # optimizer must be last in chain
            flag_opt = True
        else:  # Without optax.chain
            _opt_state = opt_state
        _dict_index = 0
        for i, _sub_state in enumerate(_opt_state):
            if len(_sub_state) > 0:  # Finding the dict containing the params state
                _dict_index = i
                break  # TODO: Not a robust approach ...
        field_names = list(_opt_state[_dict_index]._fields)
        if 'count' in field_names:
            field_names.remove('count')
        filter_in_opt_state = jax_deep_copy([getattr(_opt_state[_dict_index], field) for field in field_names])

    if state:
        filtered_state = jax_deep_copy(state)

    for layer_name, mapping_info in activation_mapping.items():
        preceding = mapping_info.get('preceding')
        following = mapping_info.get('following')

        # If there are preceding neurons, prune incoming connections
        if preceding is not None:
            preceding_neurons_state = jnp.logical_not(neurons_state_dict[preceding])

            if layer_name in filtered_params.keys():
                dict_key = "w"  # Not pruning bias based on previous connections
                upscaling_factor = preceding_neurons_state.size  # TODO: Is this still necessary? For what model?
                if upscaling_factor == 0:
                    to_repeat = 1
                else:
                    to_repeat = filtered_params[layer_name][dict_key].shape[-2] // upscaling_factor
                if to_repeat > 1:
                    preceding_neurons_state = jnp.repeat(preceding_neurons_state.reshape(1, -1), to_repeat,
                                                         axis=0).flatten()
                # else:
                #     current_state = preceding_neurons_state
                filtered_params[layer_name][dict_key] = filtered_params[layer_name][dict_key][
                    ..., preceding_neurons_state, :]
                if opt_state:
                    for j, field in enumerate(filter_in_opt_state):
                        filter_in_opt_state[j][layer_name][dict_key] = field[layer_name][dict_key][
                            ..., preceding_neurons_state, :]
            if layer_name in filtered_state.keys():
                filtered_state[layer_name]["w"] = filtered_state[layer_name]["w"][
                    ..., preceding_neurons_state, :]

        # If there are following neurons, prune outgoing connections
        if following is not None:
            following_neurons_state = jnp.logical_not(neurons_state_dict[following])
            if layer_name in filtered_params.keys():
                for dict_key in filtered_params[layer_name].keys():
                    filtered_params[layer_name][dict_key] = filtered_params[layer_name][dict_key][
                        ..., following_neurons_state]
                    if opt_state:
                        for j, field in enumerate(filter_in_opt_state):
                            filter_in_opt_state[j][layer_name][dict_key] = field[layer_name][dict_key][
                                ..., following_neurons_state]
            if layer_name in filtered_state.keys():
                for dict_key in filtered_state[layer_name].keys():
                    if dict_key != "shift_constant":
                        filtered_state[layer_name][dict_key] = filtered_state[layer_name][dict_key][
                            ..., following_neurons_state]

        # If there is state to prune
        if state and following is not None:  # This is for BN layers, no preceding connections for BN
            if f"{layer_name}/~/var_ema" in state:
                filtered_state[f"{layer_name}/~/var_ema"]["average"] = \
                filtered_state[f"{layer_name}/~/var_ema"]["average"][..., following_neurons_state]
                filtered_state[f"{layer_name}/~/var_ema"]["hidden"] = \
                filtered_state[f"{layer_name}/~/var_ema"]["hidden"][..., following_neurons_state]
            if f"{layer_name}/~/mean_ema" in state:
                filtered_state[f"{layer_name}/~/mean_ema"]["average"] = \
                filtered_state[f"{layer_name}/~/mean_ema"]["average"][..., following_neurons_state]
                filtered_state[f"{layer_name}/~/mean_ema"]["hidden"] = \
                filtered_state[f"{layer_name}/~/mean_ema"]["hidden"][..., following_neurons_state]

    if _opt_state:
        cp_state = jax_deep_copy(_opt_state)
        filtered_opt_state = cp_state[_dict_index]
        empty_state_front = cp_state[:_dict_index]
        empty_state_tail = cp_state[_dict_index+1:]

        for j, field in enumerate(field_names):
            filtered_opt_state = filtered_opt_state._replace(**{field: filter_in_opt_state[j]})


        if flag_opt:
            new_opt_state = (empty_state_front + (filtered_opt_state,) + empty_state_tail,)
            new_opt_state = opt_state[:-1] + new_opt_state
        else:
            new_opt_state = (filtered_opt_state,)

    excluded_state = {
        _state: _val if (activation_mapping[_state]["preceding"] is not None or
                         activation_mapping[_state]["following"] is not None)
        else jnp.zeros_like(_val)
        for _state, _val in neurons_state_dict.items()
    }  # Excluded layers from pruning are unaccounted for when returning the new sizes to modify the architecture
    new_sizes = [int(jnp.sum(jnp.logical_not(state))) for state in excluded_state.values()]

    if opt_state:
        if state:
            return filtered_params, new_opt_state, filtered_state, tuple(new_sizes)
        else:
            return filtered_params, new_opt_state, {}, tuple(new_sizes)
    else:
        if state:
            return filtered_params, filtered_state, tuple(new_sizes)
        else:
            return filtered_params, {}, tuple(new_sizes)


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


def count_non_zero_params(params):
    """ Params counter suitable for method that prunes with mask"""
    params_count = jax.tree_map(jnp.count_nonzero, params)
    params_count, _ = ravel_pytree(params_count)
    return jax.device_get(jnp.sum(params_count))


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
        if _batch[1].ndim > 1:
            return jnp.mean(jnp.argmax(predictions, axis=-1) == jnp.argmax(_batch[1], axis=1))
        else:
            return jnp.mean(jnp.argmax(predictions, axis=-1) == _batch[1])

    return _accuracy


def create_full_accuracy_fn(accuracy_fn, scan_len):
    def full_accuracy_fn(params, state, batch_it):

        acc = [accuracy_fn(params, state, next(batch_it))]
        for i in range(scan_len-1):
            acc.append(accuracy_fn(params, state, next(batch_it)))
        return jnp.mean(jnp.stack(acc))

    # def full_accuracy_fn(params: hk.Params, state: hk.State, batch_it): # TODO: can't use scan with iter!
    #     def scan_accuracy_fn(carry, _):
    #         return None, accuracy_fn(params, state, next(batch_it))
    #
    #     _, all_acc = jax.lax.scan(scan_accuracy_fn, None, None, scan_len)
    #     return jnp.mean(all_acc)
    return full_accuracy_fn


def keep_offset_only(dict_params):
    def offset_selection(sub_dict):
        return {k: v for k, v in sub_dict.items() if "offset" in k}

    return {k: offset_selection(v) for k, v in dict_params.items()}


def keep_scale_only(dict_params):
    def scale_selection(sub_dict):
        return {k: v for k, v in sub_dict.items() if "scale" in k}

    return {k: scale_selection(v) for k, v in dict_params.items()}


def keep_bn_params_only(dict_params):
    def bn_selection(sub_dict):
        return {k: v for k, v in sub_dict.items() if (("offset" in k) or ("scale" in k))}

    return {k: bn_selection(v) for k, v in dict_params.items()}


def zero_out_bn_params(dict_params):
    def bn_selection(sub_dict):
        return {k: (v*0 if ("offset" in k or "scale" in k) else v) for k, v in sub_dict.items()}

    return {k: bn_selection(v) for k, v in dict_params.items()}


def zero_out_all_except_bn_offset(dict_params):
    def bn_selection(sub_dict):
        return {k: (v*0 if ("offset" not in k) else jnp.ones_like(v)) for k, v in sub_dict.items()}

    return {k: bn_selection(v) for k, v in dict_params.items()}


def exclude_bn_scale_from_params(dict_params):
    def exclude_scale(sub_dict):
        return {k: v for k, v in sub_dict.items() if "scale" not in k}

    return {k: exclude_scale(v) for k, v in dict_params.items()}


def exclude_bn_offset_from_params(dict_params):
    def exclude_offset(sub_dict):
        return {k: v for k, v in sub_dict.items() if "offset" not in k}

    return {k: exclude_offset(v) for k, v in dict_params.items()}


def exclude_bn_params(dict_params):  # Not a good idea, offset is equivalent to a bias parameter
    return exclude_bn_scale_from_params(exclude_bn_offset_from_params(dict_params))


def exclude_bn_and_bias_params(dict_params):
    def exclude_bias(sub_dict):
        return {k: v for k, v in sub_dict.items() if "b" not in k}

    return exclude_bn_scale_from_params(exclude_bn_offset_from_params({k: exclude_bias(v) for k, v in dict_params.items()}))


def ce_loss_given_model(model, regularizer=None, reg_param=1e-4, classes=None, is_training=True, with_dropout=False,
                            mask_head=False, reduce_head_gap=False, exclude_bias_bn_from_reg=None, label_smoothing=0):
    """ Build the cross-entropy loss given the model"""
    # if not classes:
    #     classes = 10

    assert exclude_bias_bn_from_reg in [None, 'all', 'scale', 'offset_only', 'scale_only', 'bn_params_only'], "Can only exclude some params from reg loss with all or scale rn."
    if exclude_bias_bn_from_reg:
        if exclude_bias_bn_from_reg == "all":
            exclude_fn = lambda x: exclude_bn_and_bias_params(x)
        elif exclude_bias_bn_from_reg == "scale":
            exclude_fn = lambda x: exclude_bn_scale_from_params(x)
        elif exclude_bias_bn_from_reg == "offset_only":
            exclude_fn = lambda x: keep_offset_only(x)
        elif exclude_bias_bn_from_reg == "scale_only":
            exclude_fn = lambda x: keep_scale_only(x)
        elif exclude_bias_bn_from_reg == "bn_params_only":
            exclude_fn = lambda x: keep_bn_params_only(x)
    else:
        exclude_fn = lambda x: x

    if regularizer:
        assert regularizer in ["cdg_l2", "cdg_lasso", "l2", "lasso", "cdg_l2_act", "cdg_lasso_act"]
        if regularizer == "l2":
            def reg_fn(params, activations=None):
                params = exclude_fn(params)
                return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "lasso":
            def reg_fn(params, activations=None):
                params = exclude_fn(params)
                return sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "cdg_l2":
            def reg_fn(params, activations=None):
                params = exclude_fn(params)
                return 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "cdg_lasso":
            def reg_fn(params, activations=None):
                params = exclude_fn(params)
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
        def _loss(params: hk.Params, state: hk.State, batch: Batch, dropout_key: Any, _reg_param: float = reg_param) -> Union[jnp.ndarray, Any]:
            next_dropout_key, rng = jax.random.split(dropout_key)
            (logits, activations), state = model.apply(params, state, rng, batch, return_activations=True, is_training=is_training)
            if batch[1].ndim < 2:
                labels = jax.nn.one_hot(batch[1], classes)
            else:
                labels = batch[1]
            if label_smoothing > 0:
                labels = labels * (1 - label_smoothing) + label_smoothing / classes

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

            loss = softmax_xent + _reg_param * reg_fn(params, activations) + gap

            return loss, (state, next_dropout_key)

    else:
        if with_dropout:
            dropout_key = jax.random.PRNGKey(0)  # dropout rate is zero during death eval
            model_apply_fn = Partial(model.apply, rng=dropout_key)
        else:
            model_apply_fn = model.apply

        @jax.jit
        def _loss(params: hk.Params, state: hk.State, batch: Batch, _reg_param: float = reg_param) -> Union[jnp.ndarray, Any]:
            (logits, activations), state = model_apply_fn(params, state, x=batch, return_activations=True, is_training=is_training)
            if batch[1].ndim < 2:
                labels = jax.nn.one_hot(batch[1], classes)
            else:
                labels = batch[1]
            if label_smoothing > 0:
                labels = labels * (1 - label_smoothing) + label_smoothing / classes

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

            loss = softmax_xent + _reg_param * reg_fn(params, activations) + gap

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
                # params = exclude_bn_scale_from_params(params)
                return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "lasso":
            def reg_fn(params, activations=None):
                # params = exclude_bn_scale_from_params(params)
                return sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "cdg_l2":
            def reg_fn(params, activations=None):
                # params = exclude_bn_scale_from_params(params)
                return 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_util.tree_leaves(params))
        if regularizer == "cdg_lasso":
            def reg_fn(params, activations=None):
                # params = exclude_bn_scale_from_params(params)
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

    @jax.jit
    def _loss(params: hk.Params, state: hk.State, batch: Batch, _reg_param: float = reg_param) -> Union[jnp.ndarray, Any]:
        (outputs, activations), state = model.apply(params, state, x=batch, return_activations=True,
                                                    is_training=is_training)
        targets = batch[1]

        # calculate mse across the batch
        mse = jnp.mean(jnp.square(outputs-targets))

        loss = mse + _reg_param * reg_fn(params, activations)

        if is_training:
            return loss, state
        else:
            return loss

    return _loss


def grad_normalisation_per_layer(param_leaf):
    var = jnp.var(param_leaf)
    return param_leaf/jnp.sqrt(var+1)


def update_given_loss_and_optimizer(loss, optimizer, noise=False, noise_imp=(1, 1), asymmetric_noise=False, going_wild=False,
                                    live_only=True, noise_offset_only=False, positive_offset=False, norm_grad=False, with_dropout=False,
                                    return_grad=False, modulate_via_gate_grad=False, acti_map=None, perturb=0,
                                    init_fn=None):
    """Learning rule (stochastic gradient descent)."""

    if modulate_via_gate_grad:
        assert bool(acti_map), "activation mapping must be given for gradients modulation via gate gradients to work"

    def modulated_grad_from_gate_stat(params, state, batch):
        gate_states, rest = scr.split_state(state)

        def _loss(_params, _gate_states, _rest, _batch):
            _state = scr.recombine_state_dicts(_gate_states, _rest)
            return loss(_params, _state, _batch)

        (grads, gate_grads), new_state = jax.grad(_loss, argnums=(0, 1), has_aux=True)(params, gate_states, rest, batch)
        gate_grads = jax.tree_map(jnp.abs, gate_grads)
        gate_grads = {top_key: list(low_dict.values())[0] for top_key, low_dict in gate_grads.items()}
        max_in_gate_grad = jnp.max(ravel_pytree(jax.tree_map(jnp.mean, gate_grads))[0])
        scaling_factors = jax.tree_map(lambda x: (max_in_gate_grad/(jnp.mean(x)+1e-8)), gate_grads)  # Avoid division by 0
        for key, val in grads.items():
            scale_by = acti_map[key]["following"]
            if scale_by:
                grads[key] = jax.tree_map(lambda x: jnp.clip(x*scaling_factors[scale_by], -10.0, 10.0), val)  # clip grad
        return grads, new_state

    if with_dropout:
        assert not return_grad, 'return_grad option not coded yet with dropout'

        @jax.jit
        def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _drop_key, _reg_param: float = 0.0) -> Tuple[
            hk.Params, Any, OptState, jax.random.PRNGKeyArray]:
            if modulate_via_gate_grad:
                sys.exit("Gradient modulation via gate grad not implemented for dropout yet")
            grads, (new_state, next_drop_key) = jax.grad(loss, has_aux=True)(_params, _state, _batch, _drop_key,
                                                                             _reg_param)
            if norm_grad:
                grads = jax.tree_map(grad_normalisation_per_layer, grads)
            updates, _opt_state = optimizer.update(grads, _opt_state, _params)
            new_params = optax.apply_updates(_params, updates)
            return new_params, new_state, _opt_state, next_drop_key

    else:
        if not noise:
            if perturb:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                            _batch: Batch, _key: Any, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState, Any]:
                    key, next_key = jax.random.split(_key)
                    perturbed_params = jax.tree_map(jnp.add, _params, jax.tree_map(lambda x: x*perturb, zero_out_bn_params(init_fn(key))))  # Don't want to perturb normalisation layers params
                    if modulate_via_gate_grad:
                        grads, new_state = modulated_grad_from_gate_stat(perturbed_params, _state, _batch)
                    else:
                        grads, new_state = jax.grad(loss, has_aux=True)(perturbed_params, _state, _batch, _reg_param)
                    if norm_grad:
                        grads = jax.tree_map(grad_normalisation_per_layer, grads)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    new_params = optax.apply_updates(_params, updates)
                    return new_params, new_state, _opt_state, next_key
            elif return_grad:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                            _batch: Batch, _reg_param: float = 0.0) -> Tuple[dict, hk.Params, Any, OptState]:
                    if modulate_via_gate_grad:
                        grads, new_state = modulated_grad_from_gate_stat(_params, _state, _batch)
                    else:
                        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
                    if norm_grad:
                        grads = jax.tree_map(grad_normalisation_per_layer, grads)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    new_params = optax.apply_updates(_params, updates)
                    return grads, new_params, new_state, _opt_state

            else:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                            _batch: Batch, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState]:
                    if modulate_via_gate_grad:
                        grads, new_state = modulated_grad_from_gate_stat(_params, _state, _batch)
                    else:
                        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
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
            if going_wild:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                            _key: Any, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState, Any]:
                    if modulate_via_gate_grad:
                        grads, new_state = modulated_grad_from_gate_stat(_params, _state, _batch)
                    else:
                        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
                    key, next_key = jax.random.split(_key)
                    flat_grads, unravel_fn = ravel_pytree(grads)
                    added_noise = _var * jax.random.normal(key, shape=flat_grads.shape)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    flat_updates, _ = ravel_pytree(updates)
                    if positive_offset:
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        offset_mask *= (jnp.abs(flat_grads) == 0)
                        added_noise = jnp.where(offset_mask, jnp.abs(added_noise),
                                                added_noise * jnp.clip(jnp.abs(flat_grads), a_min=0.00005/_var))
                    else:
                        added_noise *= jnp.clip(jnp.abs(flat_grads), a_min=0.00005/_var)
                    noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)
                    return new_params, new_state, _opt_state, next_key
            elif asymmetric_noise:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                            _key: Any, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState, Any]:
                    if modulate_via_gate_grad:
                        grads, new_state = modulated_grad_from_gate_stat(_params, _state, _batch)
                    else:
                        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
                    key, next_key = jax.random.split(_key)
                    flat_grads, unravel_fn = ravel_pytree(grads)
                    added_noise = _var * jax.random.normal(key, shape=flat_grads.shape)
                    if live_only:
                        added_noise = added_noise * (
                                jnp.abs(flat_grads) >= 0)  # Only apply noise to weights with gradient!=0 (live neurons)
                    else:
                        added_noise = added_noise * (
                                jnp.abs(flat_grads) == 0)  # Only apply noise to weights with gradient==0 (dead neurons)
                    # noisy_grad = unravel_fn(a * flat_grads + b * added_noise)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    flat_updates, _ = ravel_pytree(updates)
                    if noise_offset_only:  # Watch-out, will work even if no normalization layer with offset params
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        if positive_offset:
                            added_noise = jnp.abs(added_noise)  # More efficient revival if solely increasing offset
                        added_noise *= offset_mask
                    elif positive_offset:
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        added_noise = jnp.where(offset_mask, jnp.abs(added_noise), added_noise)
                    noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)
                    return new_params, new_state, _opt_state, next_key
            else:
                @jax.jit
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                            _key: Any, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState, Any]:
                    if modulate_via_gate_grad:
                        grads, new_state = modulated_grad_from_gate_stat(_params, _state, _batch)
                    else:
                        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    key, next_key = jax.random.split(_key)
                    flat_updates, unravel_fn = ravel_pytree(updates)
                    added_noise = _var*jax.random.normal(key, shape=flat_updates.shape)
                    if noise_offset_only:  # Watch-out, will work even if no normalization layer with offset params
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        if positive_offset:
                            added_noise = jnp.abs(added_noise)  # More efficient revival if solely increasing offset
                        added_noise *= offset_mask
                    elif positive_offset:
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        added_noise = jnp.where(offset_mask, jnp.abs(added_noise), added_noise)
                    noisy_updates = unravel_fn(a*flat_updates + b*added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)
                    return new_params, new_state, _opt_state, next_key

    return _update


def update_with_accumulated_grads(loss, optimizer, noise=False, noise_imp=(1, 1), asymmetric_noise=False, going_wild=False,
                                    live_only=True, noise_offset_only=False, positive_offset=False, norm_grad=False, with_dropout=False,
                                    return_grad=False, modulate_via_gate_grad=False, acti_map=None, perturb=0,
                                    init_fn=None, accumulated_grads=1):
    assert not modulate_via_gate_grad, "modulate_via_gate_grad not supported with accumulated gradients update routine"
    assert not norm_grad, "norm_grad not supported with accumulated gradients update routine"
    assert (accumulated_grads > 0) and isinstance(accumulated_grads, int), "accumulated_grads must be positive integer"

    @jax.jit
    def accumulate_grad(_acc_grads, _params, _state, _batch, _reg_param):
        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
        grads = jax.tree_map(lambda g: g/accumulated_grads, grads)
        _acc_grads = jax.tree_map(jnp.add, _acc_grads, grads)
        return _acc_grads, new_state

    if with_dropout:
        assert not return_grad, 'return_grad option not coded yet with dropout'
        assert False, "dropout not supported rn with acc. grads"
    else:
        if not noise:
            @jax.jit
            def optim_update(grads, _params, _opt_state):
                updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                new_params = optax.apply_updates(_params, updates)
                return new_params, _opt_state
            if perturb:
                assert False, "no perturb with accumulated grads update routine"
            elif return_grad:
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                            batch_iterator: Any, _reg_param: float = 0.0) -> Tuple[dict, hk.Params, Any, OptState]:
                    acc_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), _params)
                    for _ in range(accumulated_grads):
                        acc_grads, new_state = accumulate_grad(acc_grads, _params, _state, next(batch_iterator),
                                                               _reg_param)

                    new_params, _opt_state = optim_update(acc_grads, _params, _opt_state)
                    return acc_grads, new_params, new_state, _opt_state
            else:
                def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                            batch_iterator: Any, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState]:
                    acc_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), _params)
                    for _ in range(accumulated_grads):
                        acc_grads, new_state = accumulate_grad(acc_grads, _params, _state, next(batch_iterator),
                                                               _reg_param)

                    new_params, _opt_state = optim_update(acc_grads, _params, _opt_state)
                    return new_params, new_state, _opt_state
        else:
            a, b = noise_imp
            assert not return_grad, 'return_grad option not coded yet with noisy grad'
            if going_wild:
                assert False, 'no need to support this option with accumulated grads'
            elif asymmetric_noise:
                @jax.jit
                def optim_update(grads, _params, _opt_state, _var, _key):
                    key, next_key = jax.random.split(_key)
                    flat_grads, unravel_fn = ravel_pytree(grads)
                    added_noise = _var * jax.random.normal(key, shape=flat_grads.shape)
                    if live_only:
                        added_noise = added_noise * (
                                jnp.abs(flat_grads) >= 0)  # Only apply noise to weights with gradient!=0 (live neurons)
                    else:
                        added_noise = added_noise * (
                                jnp.abs(flat_grads) == 0)  # Only apply noise to weights with gradient==0 (dead neurons)
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    flat_updates, _ = ravel_pytree(updates)
                    if noise_offset_only:  # Watch-out, will work even if no normalization layer with offset params
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        if positive_offset:
                            added_noise = jnp.abs(added_noise)  # More efficient revival if solely increasing offset
                        added_noise *= offset_mask
                    elif positive_offset:
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        added_noise = jnp.where(offset_mask, jnp.abs(added_noise), added_noise)
                    noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)

                    return new_params, _opt_state, next_key
            else:
                @jax.jit
                def optim_update(grads, _params, _opt_state, _var, _key):
                    updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                    key, next_key = jax.random.split(_key)
                    flat_updates, unravel_fn = ravel_pytree(updates)
                    added_noise = _var * jax.random.normal(key, shape=flat_updates.shape)
                    if noise_offset_only:  # Watch-out, will work even if no normalization layer with offset params
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        if positive_offset:
                            added_noise = jnp.abs(added_noise)  # More efficient revival if solely increasing offset
                        added_noise *= offset_mask
                    elif positive_offset:
                        offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                        added_noise = jnp.where(offset_mask, jnp.abs(added_noise), added_noise)
                    noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                    new_params = optax.apply_updates(_params, noisy_updates)

                    return new_params, _opt_state, next_key

            def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, batch_iterator: Any, _var: float,
                        _key: Any, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState, Any]:
                acc_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), _params)
                for _ in range(accumulated_grads):
                    acc_grads, new_state = accumulate_grad(acc_grads, _params, _state, next(batch_iterator),
                                                           _reg_param)
                new_params, _opt_state, next_key = optim_update(acc_grads, _params, _opt_state, _var, _key)
                return new_params, new_state, _opt_state, next_key

    return _update


def update_from_sgd_noise(loss, optimizer, with_dropout=False):
    """Modified version of update above that trains only on noisy part of signal (true grad - noisy grad)
       Solely used for small experiments meant to support theoretical assumptions"""

    if with_dropout:
        sys.exit("Dropout not supported yet for noisy training")

    @jax.jit
    def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                _noisy_batch: Batch, _full_batch: Batch, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState]:
        noisy_grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _noisy_batch, _reg_param)
        true_grads, _ = jax.grad(loss, has_aux=True)(_params, _state, _full_batch, _reg_param)
        updates, _opt_state = optimizer.update(jax.tree_map(jnp.subtract, true_grads, noisy_grads), _opt_state, _params)
        new_params = optax.apply_updates(_params, updates)
        return new_params, new_state, _opt_state

    return _update


def update_from_gaussian_noise(loss, optimizer, lr, bs, asymmetric_noise=False, with_dropout=False):
    """Modified version of update above that trains only on noisy part of signal (true grad - noisy grad)
       Solely used for small experiments meant to support theoretical assumptions"""

    if with_dropout:
        sys.exit("Dropout not supported yet for noisy training")

    @jax.jit
    def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                _batch: Batch, rdm_key: Any, _reg_param: float = 0.0) -> Tuple[hk.Params, Any, OptState, Any]:
        grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
        flat_grads, unravel_fn = ravel_pytree(grads)
        next_key, key = jax.random.split(rdm_key)
        gauss_noise = jax.random.normal(key, flat_grads.shape) * jnp.sqrt(lr/bs)
        if asymmetric_noise:
            gauss_noise *= (jnp.abs(flat_grads) > 0).astype(jnp.int32)
            # gauss_noise *= flat_grads
        gauss_noise = unravel_fn(gauss_noise)
        updates, _opt_state = optimizer.update(gauss_noise, _opt_state, _params)
        new_params = optax.apply_updates(_params, updates)
        return new_params, new_state, _opt_state, next_key

    return _update


def get_mask_update_fn(loss, optimizer, noise=False, noise_imp=(1, 1), asymmetric_noise=False, going_wild=False,
                       live_only=True, noise_offset_only=False, positive_offset=False, norm_grad=False,
                       with_dropout=False,
                       return_grad=False, modulate_via_gate_grad=False, acti_map=None, perturb=0,
                       init_fn=None):
    """ Return the update function, but taking into account a mask for neurons that we want to freeze the weights"""

    if not noise:
        @jax.jit
        def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState,
                    _batch: Batch, _reg_param: float = 0.0, _mask: Optional[hk.Params] = None) -> Tuple[hk.Params, Any, OptState]:
            grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
            updates, _opt_state = optimizer.update(grads, _opt_state, _params)
            if _mask:
                updates = jax.tree_map(jnp.multiply, updates, _mask)
            new_params = optax.apply_updates(_params, updates)
            return new_params, new_state, _opt_state
    else:
        a, b = noise_imp
        assert not return_grad, 'return_grad option not coded yet with noisy grad'
        if asymmetric_noise:
            @jax.jit
            def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                        _key: Any, _reg_param: float = 0.0, _mask: Optional[hk.Params] = None) -> Tuple[hk.Params, Any, OptState, Any]:
                grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
                key, next_key = jax.random.split(_key)
                flat_grads, unravel_fn = ravel_pytree(grads)
                added_noise = _var * jax.random.normal(key, shape=flat_grads.shape)
                if live_only:
                    added_noise = added_noise * (
                            jnp.abs(flat_grads) >= 0)  # Only apply noise to weights with gradient!=0 (live neurons)
                else:
                    added_noise = added_noise * (
                            jnp.abs(flat_grads) == 0)  # Only apply noise to weights with gradient==0 (dead neurons)
                # noisy_grad = unravel_fn(a * flat_grads + b * added_noise)
                updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                flat_updates, _ = ravel_pytree(updates)
                if noise_offset_only:  # Watch-out, will work even if no normalization layer with offset params
                    offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                    if positive_offset:
                        added_noise = jnp.abs(added_noise)  # More efficient revival if solely increasing offset
                    added_noise *= offset_mask
                elif positive_offset:
                    offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                    added_noise = jnp.where(offset_mask, jnp.abs(added_noise), added_noise)
                noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                if _mask:
                    noisy_updates = jax.tree_map(jnp.multiply, noisy_updates, _mask)
                new_params = optax.apply_updates(_params, noisy_updates)
                return new_params, new_state, _opt_state, next_key
        else:
            @jax.jit
            def _update(_params: hk.Params, _state: hk.State, _opt_state: OptState, _batch: Batch, _var: float,
                        _key: Any, _reg_param: float = 0.0, _mask: Optional[hk.Params] = None) -> Tuple[hk.Params, Any, OptState, Any]:
                grads, new_state = jax.grad(loss, has_aux=True)(_params, _state, _batch, _reg_param)
                updates, _opt_state = optimizer.update(grads, _opt_state, _params)
                key, next_key = jax.random.split(_key)
                flat_updates, unravel_fn = ravel_pytree(updates)
                added_noise = _var * jax.random.normal(key, shape=flat_updates.shape)
                if noise_offset_only:  # Watch-out, will work even if no normalization layer with offset params
                    offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                    if positive_offset:
                        added_noise = jnp.abs(added_noise)  # More efficient revival if solely increasing offset
                    added_noise *= offset_mask
                elif positive_offset:
                    offset_mask, _ = ravel_pytree(zero_out_all_except_bn_offset(grads))
                    added_noise = jnp.where(offset_mask, jnp.abs(added_noise), added_noise)
                noisy_updates = unravel_fn(a * flat_updates + b * added_noise)
                if _mask:
                    noisy_updates = jax.tree_map(jnp.multiply, noisy_updates, _mask)
                new_params = optax.apply_updates(_params, noisy_updates)
                return new_params, new_state, _opt_state, next_key

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
    treedef = jax.tree_util.tree_structure(container)
    leaves = jax.tree_util.tree_leaves(container)
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

# augment_train_imagenet_dataset = tf.keras.Sequential([
    ## tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    # tf.keras.layers.RandomFlip("horizontal"),
    # tf.keras.layers.RandomCrop(224, 224)
# ])


def reflection_padding2d(x, padding=4):
    return tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')


ReflectionPadding2D = tf.keras.layers.Lambda(reflection_padding2d)
srigl_data_augmentation_tf = tf.keras.Sequential([
    ReflectionPadding2D,  # Replaces tf.keras.layers.ZeroPadding2D
    tf.keras.layers.RandomCrop(height=32, width=32),
    tf.keras.layers.RandomFlip('horizontal')
])


@tf.function
def augment_train_imagenet_dataset_res50(image, label):
    # # Map function to apply to each individual image in the batch
    # def augment_image(image):
    #     # Randomly crop the image
    #     image = tf.image.random_crop(image, [224, 224, 3])  # Assuming image has 3 color channels
    #     # Randomly flip the image horizontally
    #     image = tf.image.random_flip_left_right(image)
    #     return image
    #
    # # Apply the `augment_image` function to each image in the batch
    # augmented_images = tf.map_fn(augment_image, images,
    #                              fn_output_signature=tf.TensorSpec(shape=[224, 224, 3], dtype=tf.float32))

    # Randomly crop the image
    image = tf.image.random_crop(image, [224, 224, 3])  # Assuming image has 3 color channels
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)

    return image, tf.one_hot(label, 1000)


@tf.function
def augment_train_imagenet_dataset_vit(image, label):
    # Image is already cropped
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)

    # RandAugment
    image = RandAugment(magnitude=9).distort(image)
    # MixUp and CutMix
    image = tf.cast(image, tf.float32)
    image, label = MixupAndCutmix(num_classes=1000, mixup_alpha=0.2, cutmix_alpha=1.0,
                                                     label_smoothing=0.11)(image, label)
    return image, label


@tf.function
def process_test_imagenet_dataset(image, label):
    image = tf.image.central_crop(image, 224/256)  # assuming input image size is 256x256
    return image, tf.one_hot(label, 1000)


@tf.function
def resize_tf_dataset(images, labels, dataset, is_training=False):
    """Final data augmentation step for cifar10 and imagenet, consisting of resizing."""
    if dataset == 'cifar10':
        # Resize images to 64x64
        images = tf.image.resize(images, [64, 64])
    elif dataset == "imagenet":
        if is_training:
            # Generate a random number to decide the resizing dimensions
            random_num = tf.random.uniform(shape=[], minval=0, maxval=1)

            if random_num < 0.5:
                # Resize images to 480x480 with 50% probability when training
                images = tf.image.resize(images, [480, 480])
            else:
                # Resize images to 256x256 with 50% probability when training
                images = tf.image.resize(images, [256, 256])
        else:
            # Resize images to 256x256 when not training
            images = tf.image.resize(images, [256, 256])
    elif dataset == 'imagenet_vit':
        if is_training:
            # Define parameters for the crop
            size = 224
            scale = (0.08, 1.0)  # Commonly used scale range for RandomResizedCrop
            ratio = (3. / 4., 4. / 3.)  # Commonly used aspect ratio range for RandomResizedCrop

            # Apply random resized crop to each image in the batch
            images = random_resized_crop(images, size, scale, ratio)
        else:
            # Resize images to 256x256 when not training
            images = tf.image.resize(images, [256, 256])
    return images, labels


def random_resized_crop(image, size, scale, ratio, interpolation='bilinear'):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    area = tf.cast(height * width, tf.float32)

    log_ratio = tf.math.log(ratio)

    def _crop_with_attempts():
        center_crop = True
        offset_h, offset_w, h, w = 0, 0, 0, 0
        for _ in tf.range(10):
            target_area = area * tf.random.uniform([], minval=scale[0], maxval=scale[1])
            aspect_ratio = tf.exp(tf.random.uniform([], minval=log_ratio[0], maxval=log_ratio[1]))

            w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
            h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

            w = tf.minimum(w, width)
            h = tf.minimum(h, height)

            if tf.logical_and(tf.greater(w, 0), tf.greater(h, 0)):
                offset_h = tf.random.uniform([], minval=0, maxval=height - h + 1, dtype=tf.int32)
                offset_w = tf.random.uniform([], minval=0, maxval=width - w + 1, dtype=tf.int32)
                center_crop = False
                break
        # Fallback to central crop
        if center_crop:
            in_ratio = width / height
            if in_ratio < min(ratio):
                w = width
                h = tf.cast(tf.round(tf.cast(w, tf.float32) / min(ratio)), tf.int32)
            elif in_ratio > max(ratio):
                h = height
                w = tf.cast(tf.round(tf.cast(h, tf.float32) * max(ratio)), tf.int32)
            else:
                w, h = width, height
            offset_h = (height - h) // 2
            offset_w = (width - w) // 2

        return offset_h, offset_w, h, w

    offset_h, offset_w, h, w = _crop_with_attempts()
    crop = tf.image.crop_to_bounding_box(image, offset_h, offset_w, h, w)

    # Resize the crop to the desired size
    resized_crop = tf.image.resize(crop, (size, size), method=interpolation)
    return resized_crop


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
    if image.shape[0] < 32:
        padding = (32 - image.shape[0])//2  # TODO: dirty, ensure compatibility with other shapes than 28, 32, ...
        image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
    return image/255, label


@tf.function
def imgnet_interval_zero_one(image, label):
    """Same as above, but avoid error resulting from imagenet images having variable input size"""
    return tf.cast(image, tf.float32) / 255., label


def standardize_img(image, label):
    image -= tf.math.reduce_mean(image, axis=[0, 1])
    image /= tf.math.reduce_std(image, axis=[0, 1])
    return image, label


@tf.function
def custom_normalize_img(image, label, dataset):
    assert dataset in ["mnist", 'cifar10', 'cifar10_srigl', 'cifar100', 'imagenet', 'imagenet_vit'], "need to implement normalization for others dataset"
    if dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]  # Mean for each channel (R, G, B)
        std = [0.2023, 0.1994, 0.2010]  # Standard deviation for each channel (R, G, B)
    elif dataset == "mnist":
        mean = [0.1307]  # Mean for each channel (R, G, B)
        std = [0.3081]  # Standard deviation for each channel (R, G, B)
    elif dataset == "cifar100":
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif "imagenet" in dataset:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == "cifar10_srigl":
        # Perform per-image standardization instead
        image = tf.cast(image, dtype=tf.float32)
        mean, variance = tf.nn.moments(image, axes=[0, 1], keepdims=True)
        std = tf.math.sqrt(variance)
        return (image - mean) / std, label
    else:
        raise SystemExit
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)

    image = tf.cast(image, dtype=tf.float32)
    image = (image - mean) / std
    return image, label


def load_imagenet_tf(dataset_dir: str, split: str, *, is_training: bool, batch_size: int,
                    other_bs: Optional[Iterable] = None,
                    subset: Optional[int] = None, transform: bool = True,
                    cardinality: bool = False, noisy_label: float = 0, permuted_img_ratio: float = 0,
                    gaussian_img_ratio: float = 0, augment_dataset: bool = False, normalize: bool = False,
                    reduced_ds_size: Optional[int] = None, dataset="imagenet"):
    """Retrieve the locally downloaded tar-balls for imagenet, prepare and load ds"""
   #  download_config = tfds.download.DownloadConfig(
   #      extract_dir=os.path.join(dataset_dir, 'extracted'),
   #      manual_dir=dataset_dir
   #  )

    data_augmentation = augment_dataset
    if split == "test":
        split = "validation"
    builder = tfds.builder("imagenet2012")
    builder.download_and_prepare(download_dir=dataset_dir)
    if 'vit' in dataset:
        augment_train_imagenet_dataset = augment_train_imagenet_dataset_vit
    else:
        augment_train_imagenet_dataset = augment_train_imagenet_dataset_res50

    # Create AutotuneOptions
    options = tf.data.Options()
    options.autotune.enabled = True
    options.autotune.ram_budget = (250//3) * 1024**3  # TODO: RAM budget should be determine auto. current rule: 1/2 of total RAM
    options.autotune.cpu_budget = 16  # TODO: Also determine auto. current rule: all avail cpus


    def filter_fn(image, label):
        return tf.reduce_any(subset == int(label))
    if reduced_ds_size:
        _split = split + '[:' + str(int(reduced_ds_size)) + ']'
        _split = tfds.split_for_jax_process(_split, drop_remainder=True)
        ds = builder.as_dataset(split=_split, as_supervised=True, shuffle_files=True,
                       read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
        ds = ds.with_options(options)
        if subset is not None:
            ds = ds.filter(filter_fn)  # Only take the randomly selected subset
    elif noisy_label or permuted_img_ratio or gaussian_img_ratio:
        assert (noisy_label >= 0) and (noisy_label <= 1), "noisy label ratio must be between 0 and 1"
        assert (permuted_img_ratio >= 0) and (permuted_img_ratio <= 1), "permuted_img ratio must be between 0 and 1"
        assert (gaussian_img_ratio >= 0) and (gaussian_img_ratio <= 1), "gaussian_img ratio must be between 0 and 1"
        noisy_ratio = max(noisy_label, permuted_img_ratio, gaussian_img_ratio)
        split1 = split + '[:' + str(int(noisy_ratio*100)) + '%]'
        split2 = split + '[' + str(int(noisy_ratio*100)) + '%:]'
        split1 = tfds.split_for_jax_process(split1, drop_remainder=True)
        split2 = tfds.split_for_jax_process(split2, drop_remainder=True)
        ds1, ds_info = builder.as_dataset(split=split1, as_supervised=True, with_info=True, shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
        ds2 = builder.as_dataset(split=split2, as_supervised=True, shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
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
        ds = ds.with_options(options)
    else:
        split = tfds.split_for_jax_process(split, drop_remainder=True)
        ds = builder.as_dataset(split=split, as_supervised=True, shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
        ds = ds.with_options(options)
        if subset is not None:
            ds = ds.filter(filter_fn)  # Only take the randomly selected subset
    ds_size = int(ds.cardinality())
    # ds = ds.map(imgnet_interval_zero_one, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.cache()
    # ds = ds.shuffle(50000, seed=0, reshuffle_each_iteration=True)
    if other_bs:
        ds1 = ds
        if is_training:
            ds1 = ds1.shuffle(250000, seed=0, reshuffle_each_iteration=True)
        ds1 = ds1.map(Partial(resize_tf_dataset, dataset=dataset, is_training=is_training),
                      num_parallel_calls=tf.data.AUTOTUNE)
        if 'vit' in dataset:
            ds1 = ds1.batch(batch_size)
        if is_training and data_augmentation:  # Only ds1 takes into account 'is_training' flag
            ds1 = ds1.map(augment_train_imagenet_dataset, num_parallel_calls=tf.data.AUTOTUNE)
            # ds1 = ds1.shuffle(1024, seed=0, reshuffle_each_iteration=True)
        else:
            ds1 = ds1.map(process_test_imagenet_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        if normalize:
            ds1 = ds1.map(imgnet_interval_zero_one, num_parallel_calls=tf.data.AUTOTUNE)
            ds1 = ds1.map(Partial(custom_normalize_img, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
        if 'vit' not in dataset:
            ds1 = ds1.batch(batch_size)
        ds1 = ds1.prefetch(tf.data.AUTOTUNE)
        ds1 = ds1.repeat()
        all_ds = [ds1]
        for bs in other_bs:
            ds2 = ds
            ds2 = ds2.map(Partial(resize_tf_dataset, dataset=dataset, is_training=False),
                          num_parallel_calls=tf.data.AUTOTUNE)
            if 'vit' in dataset:
                ds2 = ds2.batch(bs)
            ds2 = ds2.map(process_test_imagenet_dataset, num_parallel_calls=tf.data.AUTOTUNE)
            if normalize:
                ds2 = ds2.map(imgnet_interval_zero_one, num_parallel_calls=tf.data.AUTOTUNE)
                ds2 = ds2.map(Partial(custom_normalize_img, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
            # ds2 = ds2.shuffle(50000, seed=0, reshuffle_each_iteration=True)
            if 'vit' not in dataset:
                ds2 = ds2.batch(bs)
            ds2 = ds2.prefetch(tf.data.AUTOTUNE)
            ds2 = ds2.repeat()
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
            ds = ds.shuffle(250000, seed=0, reshuffle_each_iteration=True)
        ds = ds.map(Partial(resize_tf_dataset, dataset=dataset, is_training=is_training),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if 'vit' in dataset:
            ds = ds.batch(batch_size)
        if is_training and data_augmentation:
            ds = ds.map(augment_train_imagenet_dataset, num_parallel_calls=tf.data.AUTOTUNE)
            # ds = ds.shuffle(4096, seed=0, reshuffle_each_iteration=True)
        else:
            ds = ds.map(process_test_imagenet_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        if normalize:
            ds = ds.map(imgnet_interval_zero_one, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(Partial(custom_normalize_img, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
        if 'vit' not in dataset:
            ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.repeat()

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


def load_tf_dataset(dataset: str, split: str, *, is_training: bool, batch_size: int,
                    other_bs: Optional[Iterable] = None,
                    subset: Optional[Sequence[int]] = None, transform: bool = True,
                    cardinality: bool = False, noisy_label: float = 0, permuted_img_ratio: float = 0,
                    gaussian_img_ratio: float = 0, data_augmentation: bool = False, normalize: bool = False,
                    reduced_ds_size: Optional[int] = None):  # -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches.
    subset: If only want a subset, number of classes to build the subset from
    """
    if "srigl" in dataset:
        _dataset = dataset[:-6]  # _dataset only used for loading
        augmentation_routine = srigl_data_augmentation_tf
    else:
        _dataset = dataset
        augmentation_routine = augment_tf_dataset

    def filter_fn(image, label):
        return tf.reduce_any(subset == int(label))
    if reduced_ds_size:
        _split = split + '[:' + str(int(reduced_ds_size)) + ']'
        ds = tfds.load(_dataset, split=_split, as_supervised=True, data_dir="./data",
                       read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=False))
        if subset is not None:
            ds = ds.filter(filter_fn)  # Only take the randomly selected subset
    elif noisy_label or permuted_img_ratio or gaussian_img_ratio:
        assert (noisy_label >= 0) and (noisy_label <= 1), "noisy label ratio must be between 0 and 1"
        assert (permuted_img_ratio >= 0) and (permuted_img_ratio <= 1), "permuted_img ratio must be between 0 and 1"
        assert (gaussian_img_ratio >= 0) and (gaussian_img_ratio <= 1), "gaussian_img ratio must be between 0 and 1"
        noisy_ratio = max(noisy_label, permuted_img_ratio, gaussian_img_ratio)
        split1 = split + '[:' + str(int(noisy_ratio*100)) + '%]'
        split2 = split + '[' + str(int(noisy_ratio*100)) + '%:]'
        split1 = tfds.split_for_jax_process(split1, drop_remainder=True)
        split2 = tfds.split_for_jax_process(split2, drop_remainder=True)
        ds1, ds_info = tfds.load(_dataset, split=split1, as_supervised=True, data_dir="./data", with_info=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=False))
        ds2 = tfds.load(_dataset, split=split2, as_supervised=True, data_dir="./data", read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=False))
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
        # split = tfds.split_for_jax_process(split, drop_remainder=True)
        ds = tfds.load(_dataset, split=split, as_supervised=True, data_dir="./data", read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=False))
        if subset is not None:
            ds = ds.filter(filter_fn)  # Only take the randomly selected subset
    ds_size = int(ds.cardinality())
    if ds_size == -2:
        ds_size = sum(1 for _ in ds)  # This loads the whole dataset into memory... TODO:other workaround
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
    #         ds = ds.map(lambda x, y: (augmentation_routine(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    #     # ds = ds.take(batch_size).cache().repeat()
    ds = ds.cache()
    ds = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
    # ds = ds.repeat()
    if other_bs:
        # ds1 = ds.batch(batch_size)
        if is_training:  # Only ds1 takes into account 'is_training' flag
            # ds1 = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
            ds1 = ds.batch(batch_size)
            if data_augmentation:
                ds1 = ds1.map(lambda x, y: (augmentation_routine(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                ds1 = ds1.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
            # ds1 = ds1.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
        # else:
        #     ds1 = ds1.batch(batch_size)
        ds1 = ds1.repeat()
        # if data_augmentation:  # Resize as well during test if data augmentation
        #     ds1 = ds1.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
        # ds1 = ds1.prefetch(tf.data.AUTOTUNE)
        all_ds = [ds1]
        for bs in other_bs:
            ds2 = ds.batch(bs)
            if data_augmentation:  # Resize as well for proper evaluation if data augmented
                ds2 = ds2.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
            ds2 = ds2.repeat()
            # ds2 = ds2.prefetch(tf.data.AUTOTUNE)
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
        ds = ds.batch(batch_size)
        if is_training:
            # ds = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
            # ds = ds.batch(batch_size)
            if data_augmentation:
                ds = ds.map(lambda x, y: (augmentation_routine(x), y), num_parallel_calls=tf.data.AUTOTUNE)
                ds = ds.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
            # ds = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
        else:
            # ds = ds.batch(batch_size)
            if data_augmentation:
                ds = ds.map(Partial(resize_tf_dataset, dataset=dataset), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()
        # ds = ds.prefetch(tf.data.AUTOTUNE)

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
                  noisy_label=0, permuted_img_ratio=0, gaussian_img_ratio=0, augment_dataset=False,
                  normalize: bool = False, reduced_ds_size: Optional[int] = None):
    return load_tf_dataset("mnist", split=split, is_training=is_training, batch_size=batch_size,
                           other_bs=other_bs, subset=subset, transform=transform, cardinality=cardinality,
                           noisy_label=noisy_label, permuted_img_ratio=permuted_img_ratio,
                           gaussian_img_ratio=gaussian_img_ratio, data_augmentation=augment_dataset,
                           normalize=normalize, reduced_ds_size=reduced_ds_size)


def load_cifar10_tf(split: str, is_training, batch_size, other_bs=None, subset=None, transform=True, cardinality=False,
                    noisy_label=0, permuted_img_ratio=0, gaussian_img_ratio=0, augment_dataset=False, normalize: bool = False, dataset: str = "cifar10"):
    return load_tf_dataset(dataset, split=split, is_training=is_training, batch_size=batch_size, other_bs=other_bs,
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


# FFCV loaders:
# def load_cifar10_ffcv(split: str, is_training, batch_size, other_bs=None, subset=None, noisy_label=0,
#                       permuted_img_ratio=0, gaussian_img_ratio=0,
#                       cardinality=False, augment_dataset=False, normalize: bool = False):
#     assert subset is None, "subset arg not supported"
#     assert noisy_label == 0, "noisy_label arg not supported"
#     assert permuted_img_ratio == 0, "permuted_img_ratio arg not supported"
#     assert gaussian_img_ratio == 0, "gaussian_img_ratio arg not supported"
#     _datasets = {
#         'train': datasets.CIFAR10("./data", train=True, download=True),
#         'test': datasets.CIFAR10("./data", train=False, download=True)
#     }
#
#     directory_path = "./data/ffcv/cifar10"
#     paths = {
#         'train': directory_path + "/train",
#         'test': directory_path + "/test"
#     }
#
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)
#
#     if not bool(os.listdir(directory_path)):  # Run FFCV builder if empty (i.e. if not done previously)
#         for (name, ds) in _datasets.items():
#             path = paths["train"] if name == 'train' else paths['test']
#             writer = DatasetWriter(path, {
#                 'image': RGBImageField(),
#                 'label': IntField()
#             })
#             writer.from_indexed_dataset(ds)
#
#     cifar_mean = [125.307, 122.961, 113.8575]
#     cifar_std = [51.5865, 50.847, 51.255]
#
#     name = split
#     ds_size = 50000 if name == 'train' else 10000
#     label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]
#     image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
#     if name == 'train':
#         if augment_dataset:
#             image_pipeline.extend([
#                 RandomHorizontalFlip(),
#                 RandomTranslate(padding=2, fill=tuple(map(int, cifar_mean))),
#                 Cutout(4, tuple(map(int, cifar_mean))),
#             ])
#     image_pipeline.extend([
#         ToTensor(),
#         ToDevice(torch.device('cuda:0'), non_blocking=True),
#         ToTorchImage(),
#         Convert(torch.float16),
#         transforms.Normalize(cifar_mean, cifar_std),
#     ])
#     if normalize:
#         image_pipeline.append(transforms.Normalize(cifar_mean, cifar_std))
#
#     ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL
#
#     loader = iter(Loader(paths[name], batch_size=batch_size, num_workers=4,
#                            order=ordering, drop_last=(name == 'train'),
#                            pipelines={'image': image_pipeline, 'label': label_pipeline}))
#
#     loaders = [loader]
#     if other_bs:
#         for bs in other_bs:
#             loaders.append(iter(Loader(paths[name], batch_size=bs, num_workers=4,
#                            order=ordering, drop_last=(name == 'train'),
#                            pipelines={'image': image_pipeline, 'label': label_pipeline})))
#
#     if cardinality:
#         return (ds_size, ) + tuple(loaders)
#     else:
#         return tuple(loaders)


# Pytorch dataloader # TODO: deprecated; should remove!!
# @jax.jit
# def transform_batch_pytorch(targets, indices):
#     # transformed_targets = jax.vmap(map_targets, in_axes=(0, None))(targets, indices)
#     transformed_targets = targets.apply_(lambda t: torch.nonzero(t == torch.tensor(indices))[0])
#     return transformed_targets


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


# def load_dataset(dataset: Any, is_training: bool, batch_size: int, subset: Optional[int] = None,
#                  transform: bool = True, num_workers: int = 2):
#     if subset is not None:
#         # assert subset < 10, "subset must be smaller than 10"
#         # indices = np.random.choice(10, subset, replace=False)
#         subset_idx = np.isin(dataset.targets, subset)
#         dataset.data, dataset.targets = dataset.data[subset_idx], dataset.targets[subset_idx]
#         if transform:
#             dataset.targets = transform_batch_pytorch(dataset.targets, subset)
#
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=num_workers)
#     return compatibility_iterator(data_loader)
#
#
# def load_mnist_torch(is_training, batch_size, subset=None, transform=True, num_workers=2):
#     dataset = datasets.MNIST('./data', train=is_training, download=True,
#                              transform=transforms.Compose([
#                                  transforms.ToTensor(),
#                                  ]))  # transforms.Normalize((0.1307,), (0.3081,)) -> want positive inputs
#     return load_dataset(dataset, is_training=is_training, batch_size=batch_size, subset=subset,
#                         transform=transform, num_workers=num_workers)
#
#
# def load_cifar10_torch(is_training, batch_size, subset=None, transform=True, num_workers=2):
#     dataset = datasets.CIFAR10('./data', train=is_training, download=True,
#                                transform=transforms.Compose([
#                                     transforms.ToTensor()]))
#     return load_dataset(dataset, is_training=is_training, batch_size=batch_size, subset=subset,
#                         transform=transform, num_workers=num_workers)
#
#
# def load_fashion_mnist_torch(is_training, batch_size, subset=None, transform=True, num_workers=2):
#     dataset = datasets.FashionMNIST('./data', train=is_training, download=True,
#                                     transform=transforms.Compose([
#                                         transforms.ToTensor()]))
#     return load_dataset(dataset, is_training=is_training, batch_size=batch_size, subset=subset,
#                         transform=transform, num_workers=num_workers)


##############################
# Module utilities
##############################
class FancySequential:
    """Apply hk.Sequential to layer's list to build the module, but retrieve the activation mapping
    along the way"""
    def __init__(
            self,
            layers: Any,
            name: Optional[str] = "",
            parent: Optional[hk.Module] = None
    ):
        # super().__init__(name=name)
        self.layers = tuple(layers)
        self.activation_mapping = {}
        self.parent = parent

    def __call__(self, inputs, *args, **kwargs):
        """Calls all layers sequentially, updating the activation mapping along the way"""
        out = inputs
        parent = self.parent
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer_module = layer(parent=parent)
                out = layer_module(out, *args, **kwargs)
                self.activation_mapping.update(layer_module.get_activation_mapping())
                parent = layer_module
            else:
                layer_module = layer(parent=parent)
                out = layer_module(out)
                self.activation_mapping.update(layer_module.get_activation_mapping())
                parent = layer_module
        self.last_act_name = layer_module.get_last_activation_name()
        self.delayed_layer = layer_module
        return out

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name

    def get_delayed_activations(self):
        return self.delayed_layer.get_delayed_activations()

    def get_delayed_norm(self):
        return self.delayed_layer.get_delayed_norm()


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
            self.activation_mapping = {}

        def __call__(self, x, return_activations=False, is_training=True):
            activations = []
            x = x[0].astype(jnp.float32)
            if is_training or (self.test_layers is None):
                layers = self.train_layers
            else:
                layers = self.test_layers
            parent = None
            for layer in layers[:-1]:  # Don't append final output in activations list
                # x = hk.Sequential([mdl() for mdl in layer])(x)
                layer_modules = FancySequential(layer, parent=parent)
                x = layer_modules(x)
                self.activation_mapping.update(layer_modules.get_activation_mapping())
                parent = layer_modules
                if return_activations:
                    if type(x) is tuple:
                        activations += x[1]
                        # activations.append(x[0])
                    else:
                        activations.append(x)
                if type(x) is tuple:
                    x = x[0]
            # x = hk.Sequential([mdl() for mdl in layers[-1]])(x)
            layer_modules = FancySequential(layers[-1], parent=parent)
            x = layer_modules(x)
            if type(x) is tuple:
                final_output = x[0]
            else:
                final_output = x
            self.activation_mapping.update(layer_modules.get_activation_mapping())
            if return_activations:
                if type(x) is tuple:
                    activations += x[1]
                return final_output, activations
            else:
                return final_output

        def get_activation_mapping(self):
            return self.activation_mapping

    def primary_model(x, return_activations=False, is_training=True):
        return ModelAndActivations()(x, return_activations, is_training)

    # return hk.without_apply_rng(hk.transform(typical_model)), hk.without_apply_rng(hk.transform(secondary_model))
    if not with_dropout:
        return hk.without_apply_rng(hk.transform_with_state(primary_model)), ModelAndActivations
    else:
        return hk.transform_with_state(primary_model), ModelAndActivations


##############################
# lr scheduler utilities
##############################
def constant_schedule(training_steps, base_lr, final_lr, decay_bounds, scaling_factor):
    return optax.constant_schedule(base_lr)


def fix_step_decay(training_steps, base_lr, final_lr, decay_bounds, scaling_factor):
    """ Decay by 1/10 lr at fixed step defined by user"""
    # scaling_factor = 0.1
    # bound_dict = {i: scaling_factor**(j+1) for j, i in enumerate(decay_steps)}
    bound_dict = {i: scaling_factor for i in decay_bounds}
    return optax.piecewise_constant_schedule(base_lr, bound_dict)


def piecewise_constant_schedule(training_steps, base_lr, final_lr, decay_steps, scaling_factor):
    scaling_factor = (final_lr/base_lr)**(1/(decay_steps-1))
    bound_dict = {int(training_steps/decay_steps*i): scaling_factor for i in range(1, decay_steps)}
    return optax.piecewise_constant_schedule(base_lr, bound_dict)


def cosine_decay(training_steps, base_lr, final_lr, decay_bounds, scaling_factor):
    if final_lr > 0:
        alpha_val = final_lr/base_lr
    else:
        alpha_val = 0
    return optax.cosine_decay_schedule(base_lr, training_steps, alpha_val)


def one_cycle_schedule(training_steps, base_lr, final_lr, decay_bounds, scaling_factor):
    return optax.cosine_onecycle_schedule(training_steps, base_lr)


def warmup_cosine_decay(training_steps, base_lr, final_lr, decay_bounds, scaling_factor, warmup_ratio=0.05):
    warmup_steps = training_steps*warmup_ratio  # warmup is done for 5% of training_steps
    return optax.warmup_cosine_decay_schedule(init_value=9.9e-5, peak_value=base_lr,
                                              warmup_steps=warmup_steps, decay_steps=training_steps)


def step_warmup(training_steps, base_lr, final_lr, decay_bounds, scaling_factor, warmup_ratio=1e-5):
    warmup_steps = training_steps * warmup_ratio
    return lambda s: base_lr * jnp.minimum(s/warmup_steps, 1)


def warmup_piecewise_decay_schedule(
    training_steps: int,
    base_lr: float,
    final_lr: float,
    decay_bounds: List,
    scaling_factor: float,
    warmup_ratio: float = 0.05
) -> optax.Schedule:
    """Linear warmup followed by piecewise decay
    """
    warmup_steps = int(warmup_ratio * training_steps)  # warmup is done for 5% of training_steps
    bound_dict = {i-warmup_steps: scaling_factor for i in decay_bounds}
    schedules = [
        optax.linear_schedule(
            init_value=1e-6,
            end_value=base_lr,
            transition_steps=warmup_steps),
        optax.piecewise_constant_schedule(
            init_value=base_lr,
            boundaries_and_scales=bound_dict)]
    return optax.join_schedules(schedules, [warmup_steps])


def linear_warmup(  # Use as a reg_param schedule
        warmup_steps: int,
        end_lr: float,
) -> optax.Schedule:
    """ Linear warmup followed by constant schedule"""
    schedules = [
        optax.linear_schedule(
            init_value=1e-6,
            end_value=end_lr,
            transition_steps=warmup_steps),
        optax.constant_schedule(end_lr)]
    return optax.join_schedules(schedules, [warmup_steps])

##############################
# Modified optax optimizer
##############################
AddDecayedWeightsState = base.EmptyState
ScalarOrSchedule = Union[float, base.Schedule]


def add_scheduled_decayed_weights(
    weight_decay: ScalarOrSchedule = 0.0,
    flip_sign: bool = False,
    cdg: bool = False,
    sched_end: float = jnp.inf,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
    """Allows cdg and scheduled variant of optax weight_decay
    cdg: Add parameter where positive weights are scaled by `weight_decay`
    schedule: scale wd by provided schedule (steps count fn)

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
        return transform.ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def cdg_mask(params, count):
        return jax.tree_util.tree_map(
            lambda x: ((x > 0)+(count >= sched_end))*x, params)

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        m = -1 if flip_sign else 1
        if callable(weight_decay):
            wd = m*weight_decay(state.count)
        else:
            wd = m*weight_decay
        if cdg:
            updates = jax.tree_util.tree_map(
                lambda g, p: g + wd * p, updates, cdg_mask(params, state.count))
        else:
            updates = jax.tree_util.tree_map(
                lambda g, p: g + wd * p, updates, params)
        return updates, transform.ScaleByScheduleState(
            count=numerics.safe_int32_increment(state.count))

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
      add_scheduled_decayed_weights(weight_decay, cdg=True, mask=mask),
      _scale_by_learning_rate(learning_rate),
    )


def sgdw(
    learning_rate: ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """ SGD optimizer from optax, augmented with weight decay.
    The weight decay is multiplied by lr, consistent with pytorch implementation
    """
    return combine.chain(
        (transform.trace(decay=momentum, nesterov=nesterov, accumulator_dtype=accumulator_dtype)
            if momentum is not None else base.identity()),
        transform.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate)
        )


def adam_loschilov_wd(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: ScalarOrSchedule = 1e-4,
    sched_end: float = jnp.inf,
    cdg: bool = False,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """
    Modified version of adamw (cdg optional) where the wd is NOT rescaled by the learning rate, similar to
    https://arxiv.org/pdf/1711.05101.pdf. Thus, DIFFERENT from pytorch implementation.
    Also allow to put a schedule on wd.
    """
    return combine.chain(
        transform.scale_by_adam(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        _scale_by_learning_rate(learning_rate),
        add_scheduled_decayed_weights(weight_decay, flip_sign=True, cdg=cdg, sched_end=sched_end, mask=mask),  # Flip is done in _scale_by_lr, so doing it manually here
    )


def sgd_loschilov_wd(
    learning_rate: ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
    weight_decay: ScalarOrSchedule = 0.0,
    sched_end: float = jnp.inf,
    cdg: bool = False,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """ Modified version of sgdw (cdg optional) where the wd is NOT rescaled by the learning rate, similar to
    https://arxiv.org/pdf/1711.05101.pdf. Thus, DIFFERENT from pytorch implementation.
    Also allow to put a schedule on wd.
    """
    return combine.chain(
        (transform.trace(decay=momentum, nesterov=nesterov, accumulator_dtype=accumulator_dtype)
            if momentum is not None else base.identity()),
        _scale_by_learning_rate(learning_rate),
        add_scheduled_decayed_weights(weight_decay, flip_sign=True, cdg=cdg, sched_end=sched_end, mask=mask)  # Flip is done in _scale_by_lr, so doing it manually here
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
        params: The params to prune, a dictionary of type hk.params
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
    pruned_params = jax_deep_copy(params)

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
    pruned_params = jax_deep_copy(params)
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


# Reimplementing a version of layer weight magnitude pruning, but with jaxpruner
@dataclass
class LayerMagnitudePruning(BaseUpdater):
    """Implements layer magnitude based pruning."""

    def calculate_scores(self, params, sparse_state=None, grads=None):
        del sparse_state, grads

        def sum_and_broadcast(x):
            axes = tuple(range(x.ndim - 1))
            summed = jnp.sum(jnp.abs(x), axis=axes)
            return jnp.broadcast_to(summed, x.shape)

        layer_magnitudes = jax.tree_map(sum_and_broadcast, params)
        return layer_magnitudes


# def net_to_adjacency_matrix(net, x):
#     """ Use the dot representation returned by hk.experimental.to_dot to recover an adjacency matrix
#         that will be used to map neuron sparsity to weight sparsity and for pruning."""
#
#     params, state = net.init(jax.random.PRNGKey(0), x)
#
#     # New function ignoring the state
#     def model_fn_without_state(_x):
#         return net.apply(params, state, _x, return_activations=False, is_training=False)
#
#     # Now this function can be used with hk.experimental.to_dot
#     dot = hk.experimental.to_dot(model_fn_without_state)(x)
#     print(dot)
#     raise SystemExit
#     # dot = hk.experimental.to_dot(lambda rng_k, _x: net.init(rng_k, _x)[0])(jax.random.PRNGKey(0), x)
#     # dot = hk.experimental.to_dot(net.init)(jax.random.PRNGKey(0), x)
#     agraph = pgv.AGraph(string=dot)
#
#     # Convert the AGraph into a NetworkX graph
#     nx_graph = from_agraph(agraph)
#
#     # Now you can generate an adjacency matrix from the graph
#     # We will use a SciPy sparse matrix to store it
#     adjacency_matrix = nx.adjacency_matrix(nx_graph)
#
#     # If you need it as a dense NumPy array, you can do:
#     adjacency_matrix_dense = adjacency_matrix.todense()
#
#     return adjacency_matrix_dense


##############################
# Activation fn as module
# allows to implement trick to easily recover activations gradient
##############################
class ActivationModule(hk.Module):

    def __init__(self, activation_fn: Callable, name: Optional[str] = None):
        """ Module replacement for activation function. Allows to define a state variable (a constant) that we can
        use to calculate gradient values w/r to activation
        """
        super().__init__(name=name)
        self.activation_fn = activation_fn

    def __call__(self,
                 inputs: Any,  # Used to be jax.Array; but cluster jax version < 0.4.1 (not compatible)
                 precision: Optional[jax.lax.Precision] = None,
                 ) -> Any:  # Again; switch to jax.Array when version updated on cluster
        c = hk.get_state("gate_constant", inputs.shape[-1:], inputs.dtype, init=jnp.ones)  # c must have shape
        # matching the amount of neurons
        b = hk.get_state("shift_constant", (1,), inputs.dtype, init=jnp.zeros)

        out = self.activation_fn(c*inputs - b)

        return out

    @property
    def gate_constant(self):
        return hk.get_state("gate_constant")


@jax.jit
def update_gate_constant(state, new_value):
    """ Helper function to update the shift_constant parameter inside a state_dict"""
    for activation_layer in state.keys():
        if "shift_constant" in state[activation_layer].keys():
            state[activation_layer]["shift_constant"] = jnp.full_like(state[activation_layer]["shift_constant"], new_value)

    return state

# @jax.custom_jvp
# @jax.jit
# def shifted_relu(x: jax.typing.ArrayLike, b: float = 0.0) -> jax.typing.Array:  # Using a state constant right now instead
#     """ Direct reimplementation of jax.nn.relu
#     (https://jax.readthedocs.io/en/latest/_modules/jax/_src/nn/functions.html#relu)
#
#     But with the inclusion of a shift parameter (b) that moves the threshold where gradient becomes null
#
#     """
#     return jnp.maximum(x-b, 0)
#
#
# # For behavior at 0, see https://openreview.net/forum?id=urrcVI-_jRm
# shifted_relu.defjvps(lambda g_x, ans, x, b: jax.lax.select(x-b > 0, g_x, jax.lax.full_like(g_x, 0)),
#                      lambda g_b, ans, x, b: jax.lax.select(x-b > 0, g_b, jax.lax.full_like(g_b, 0)))


class ReluActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=jax.nn.relu, name=name)


class LeakyReluActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=Partial(jax.nn.leaky_relu, negative_slope=0.05), name=name)


class AbsActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=jax.numpy.abs, name=name)


class EluActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=jax.nn.elu, name=name)


class SwishActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=jax.nn.swish, name=name)


class GeluActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=jax.nn.gelu, name=name)


class TanhActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=jax.nn.tanh, name=name)


class IdentityActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=identity_fn, name=name)


class ThreluActivationModule(ActivationModule):
    def __init__(self, name: Optional[str] = None):
        super().__init__(activation_fn=threlu, name=name)


##############################
# HK modules with activation mapping
# used to easily build the pruning function
##############################
class MaxPool(hk.MaxPool):
    """Max pool upgraded with activation mapping
    """

    def __init__(self, window_shape: Union[int, Sequence[int]], strides: Union[int, Sequence[int]], padding: str,
               channel_axis: Optional[int] = -1, name: Optional[str] = None,
               parent: Optional[hk.Module] = None, ):
        super().__init__(window_shape=window_shape, strides=strides, padding=padding,
                         channel_axis=channel_axis, name=name)
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.activation_mapping = {}

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.preceding_activation_name


class AvgPool(hk.AvgPool):
    """Avg pool upgraded with activation mapping
    """

    def __init__(self, window_shape: Union[int, Sequence[int]], strides: Union[int, Sequence[int]], padding: str,
               channel_axis: Optional[int] = -1, name: Optional[str] = None,
               parent: Optional[hk.Module] = None, ):
        super().__init__(window_shape=window_shape, strides=strides, padding=padding,
                         channel_axis=channel_axis, name=name)
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.activation_mapping = {}

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.preceding_activation_name


def get_activation_mapping(net, inputs):

    def model_fn(_net, x):
        model = _net()
        output = model(x)
        activation_fn_mapping = model.get_activation_mapping()
        parent_name = model.name
        return output, activation_fn_mapping, parent_name

    model_transformed = hk.transform_with_state(Partial(model_fn, net))

    params, state = model_transformed.init(jax.random.PRNGKey(42), inputs)
    (output, activation_mapping, parent_name), state = model_transformed.apply(params, state, jax.random.PRNGKey(42), inputs)

    # def prepend_parent_name_to_string(s):
    #     if isinstance(s, str):
    #         return parent_name + '/' + s
    #     return s
    #
    # new_activation_mapping = {}
    #
    # for key, value in activation_mapping.items():
    #     new_key = parent_name + '/' + key
    #     new_value = jax.tree_map(prepend_parent_name_to_string, value)
    #     new_activation_mapping[new_key] = new_value

    return activation_mapping


##############################
# Minimnist utils
# Utils for aggregating info about noise dynamics during the minimnist experiments
##############################
class GroupedHistory(dict):
    """Dict object to collect statistics about individual neurons/weights throughout training. After training completes,
       the stats will be mapped toward their respective group, live or dead neurons, and recorded accordingly."""

    def __init__(self, neuron_noise_ratio: bool):
        super().__init__()  # Initialize as an empty dict
        if neuron_noise_ratio:
            self["neuron_noise_ratio"] = {}
            self.neuron_noise_ratio = self["neuron_noise_ratio"]

    def update_neuron_noise_ratio(self, step, params, state, test_loss, dataloader_noisy, dataloader_full_batch):
        gate_states, rest = scr.split_state(state)

        def loss_wr_gate(_gate_states, _batch):
            _state = scr.recombine_state_dicts(_gate_states, rest)
            return test_loss(params, _state, _batch)

        def grad_fn(_batch):
            gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
            return gate_grad

        def ratio_fn(noise, grad):
            return jnp.clip(jnp.abs(noise/(grad+1e-8)), a_max=1)

        true_neuron_gradient = grad_fn(next(dataloader_full_batch))
        neuron_noise = jax.tree_map(jnp.subtract, grad_fn(next(dataloader_noisy)), true_neuron_gradient)
        neuron_noise_to_grad_ratio = jax.tree_map(ratio_fn, neuron_noise, true_neuron_gradient)

        self.neuron_noise_ratio[step] = neuron_noise_to_grad_ratio


##############################
# Modified optimizers
##############################
def scale_by_adam_to_momentum(
    b1: float = 0.9,
    b2: float = 0.999,
    dampening: float = 0.0,
    eps_start: float = 1e-8,
    transition_steps: int = 1e4,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Gradually transform Adam rescaling into momentum rescaling
    """

    mu_dtype = optax._src.utils.canonicalize_dtype(mu_dtype)

    eps_scaling = optax.linear_schedule(eps_start, 1, transition_steps)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return optax._src.transform.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        # mu = optax._src.transform.update_moment(updates, state.mu, b1, 1)
        mu = jax.tree_util.tree_map(lambda g, t: (1 - dampening) * g + b1 * t, updates, state.mu)  # More similart to base momentum optimizer
        nu = optax._src.transform.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax._src.numerics.safe_int32_increment(state.count)
        # mu_hat = optax._src.transform.bias_correction(mu, b1, count_inc)
        mu_bias_correction = (1-dampening)*(1 - b1**count_inc)/(1-b1)  # bias correction must now account for variable dampening
        mu_hat = jax.tree_util.tree_map(lambda t: t / mu_bias_correction.astype(t.dtype), mu)
        nu_hat = optax._src.transform.bias_correction(nu, b2, count_inc)
        eps = eps_scaling(count_inc)
        updates = jax.tree_util.tree_map(
            # lambda m, v: m * (eps / (jnp.sqrt(v + eps_root) + eps)), mu_hat, nu_hat)
            lambda m, v: m * ((jnp.sqrt(jnp.mean(v) + eps_root) + eps) / (jnp.sqrt(v + eps_root) + eps)), mu_hat, nu_hat)
        mu = optax._src.utils.cast_tree(mu, mu_dtype)
        return updates, optax._src.transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def adam_to_momentum(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    dampening: float = 0.0,
    eps_start: float = 1e-8,
    transition_steps: int = 1e4,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    """An optimizer that start as adam but gradually becomes momentum via gradual increase of eps parameter
    """
    return combine.chain(
      scale_by_adam_to_momentum(
          b1=b1, b2=b2, dampening=dampening, eps_start=eps_start, transition_steps=transition_steps, eps_root=eps_root, mu_dtype=mu_dtype),
      _scale_by_learning_rate(learning_rate),
    )


def scale_by_adam_to_momentum_v2(
    t_f,  # Total steps
    b1: float = 0.9,
    b2: float = 0.999,
    dampening: float = 0.0,
    alpha: float = 5.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
    """Gradually transform Adam rescaling into momentum rescaling
    """

    mu_dtype = optax._src.utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return optax._src.transform.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def scale_v(v, t):
        return v * jnp.exp(-alpha * t / t_f) + (1 - jnp.exp(-alpha * t / t_f))

    def update_fn(updates, state, params=None):
        del params
        # mu = optax._src.transform.update_moment(updates, state.mu, b1, 1)
        mu = jax.tree_util.tree_map(lambda g, t: (1 - dampening) * g + b1 * t, updates, state.mu)  # More similart to base momentum optimizer
        nu = optax._src.transform.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax._src.numerics.safe_int32_increment(state.count)
        mu_hat = optax._src.transform.bias_correction(mu, b1, count_inc)
        nu_hat = optax._src.transform.bias_correction(nu, b2, count_inc)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / scale_v(jnp.sqrt(v + eps_root) + eps, count_inc), mu_hat, nu_hat)
        mu = optax._src.utils.cast_tree(mu, mu_dtype)
        return updates, optax._src.transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def adam_to_momentum_v2(
    learning_rate: ScalarOrSchedule,
    t_f,  # Total steps
    b1: float = 0.9,
    b2: float = 0.999,
    dampening: float = 0.0,
    alpha: float = 5.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    """An optimizer that start as adam but gradually becomes momentum via gradual increase of eps parameter
    """
    return combine.chain(
      scale_by_adam_to_momentum_v2(
          t_f=t_f, b1=b1, b2=b2, dampening=dampening, alpha=alpha, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      _scale_by_learning_rate(learning_rate),
    )


def adamw_to_momentumw_v2(
    learning_rate: ScalarOrSchedule,
    t_f,  # Total steps
    b1: float = 0.9,
    b2: float = 0.999,
    dampening: float = 0.0,
    alpha: float = 5.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """An optimizer that start as adam but gradually becomes momentum via gradual increase of eps parameter
    """
    return combine.chain(
      scale_by_adam_to_momentum_v2(
          t_f=t_f, b1=b1, b2=b2, dampening=dampening, alpha=alpha, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
      transform.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
    )


##############################
# Checkpointing
##############################
def save_pytree_state(ckpt_dir: str, state) -> None:
    # Save the numpy arrays (parameters) to disk
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            jnp.save(f, x, allow_pickle=True)

    # Save the structure of the state tree
    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore_pytree_state(ckpt_dir, verbose=False):
    # Load the structure of the state tree
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    if verbose:
        print(jax.tree_map(jnp.shape, tree_struct))

    # Load the flat state (parameters) from disk
    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [jnp.load(f, allow_pickle=True) for _ in leaves]

    # Reconstruct the state tree from its structure and parameters
    return jax.tree_util.tree_unflatten(treedef, flat_state)


def save_all_pytree_states(parent_dir: str, params, state, opt_state):
    # Create directories for params, state, and opt_state
    params_dir = os.path.join(parent_dir, "params")
    state_dir = os.path.join(parent_dir, "state")
    opt_state_dir = os.path.join(parent_dir, "opt_state")

    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(opt_state_dir, exist_ok=True)

    # Use the existing save function
    save_pytree_state(params_dir, params)
    save_pytree_state(state_dir, state)
    save_pytree_state(opt_state_dir, opt_state)


def restore_all_pytree_states(parent_dir: str):
    # Directories for params, state, and opt_state
    params_dir = os.path.join(parent_dir, "params")
    state_dir = os.path.join(parent_dir, "state")
    opt_state_dir = os.path.join(parent_dir, "opt_state")

    # Use the existing restore function
    restored_params = restore_pytree_state(params_dir)
    restored_state = restore_pytree_state(state_dir)
    restored_opt_state = restore_pytree_state(opt_state_dir)

    return restored_params, restored_state, restored_opt_state


class RunState(TypedDict):  # Taken from https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
    """Typed dictionary containing the state of the training run which is saved at each epoch.

    Using type hints helps prevent bugs and makes your code easier to read for both humans and
    machines (e.g. Copilot). This leads to less time spent debugging and better code suggestions.
    """

    epoch: int
    training_step: int
    model_dir: str  # Parent dir contains params, model state and opt_state pytrees in separate children dir
    curr_arch_sizes: List  # To rebuild the pruned model
    aim_hash: Optional[str]  # Unique hash identifying experiment in aim (logger)
    slurm_jobid: str  # Unique experiment identifier attributed by SLURM
    exp_name: str
    curr_starting_size: Optional[List[int]]  # To use with exps that loop over size
    curr_reg_param: Optional[float]  # To use with exps that loop over reg_param
    dropout_key: Optional[jax.random.PRNGKey]
    decaying_reg_param: Optional[float]
    best_accuracy: float  # Best accuracy so far
    best_params_count: Optional[int]  # Number of remaining params for the best run so far
    best_total_neurons: Optional[int]  # Number of remaining neurons for the best run so far
    training_time: Optional[Any]  # Total training time for the run
    pruned_flag: Optional[bool]  # For structure_baseline experiments; recording if pruning happened or not
    cumulative_dead_neurons: Optional[Any]  # For dead neurons overlap


class JaxPrunerRunState(TypedDict):  # Taken from https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
    """Typed dictionary containing the state of the training run which is saved at each epoch.

    Using type hints helps prevent bugs and makes your code easier to read for both humans and
    machines (e.g. Copilot). This leads to less time spent debugging and better code suggestions.
    """

    epoch: int
    training_step: int
    model_dir: str  # Parent dir contains params, model state and opt_state pytrees in separate children dir
    aim_hash: Optional[str]  # Unique hash identifying experiment in aim (logger)
    slurm_jobid: str  # Unique experiment identifier attributed by SLURM
    exp_name: str
    curr_pruning_density: Optional[float]  # For the loop over multiple pruning_density
    dropout_key: Optional[jax.random.PRNGKey]
    decaying_reg_param: Optional[float]


def load_run_state(checkpoint_dir: Path) -> Optional[RunState]: # Taken from https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
    """Loads the latest checkpoint if possible, otherwise returns `None`."""
    checkpoint_file = checkpoint_dir / "checkpoint_run_state.pkl"
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
    if restart_count:
        print(f"NOTE: This job has been restarted {restart_count} times by SLURM.")

    if not checkpoint_file.exists():
        print(f"No checkpoint found in checkpoints dir ({checkpoint_dir}).")
        if restart_count:
            raise RuntimeWarning(
                f"This job has been restarted {restart_count} times by SLURM, but no "
                "checkpoint was found! This either means that your checkpointing code is "
                "broken, or that the job did not reach the checkpointing portion of your "
                "training loop."
            )
        return None

    with open(checkpoint_file, "rb") as f:
        checkpoint_state = pickle.load(f)

    print(f"Resuming from the checkpoint file at {checkpoint_file}:")
    print(checkpoint_state)
    print()
    state: RunState = checkpoint_state  # type: ignore
    return state


def checkpoint_exp(run_state: RunState, params, state, opt_state, curr_epoch: int, curr_step: int,
                   curr_arch_sizes, curr_starting_size, curr_reg_param, dropout_key, decaying_reg_param,
                   best_acc, best_params_count, best_total_neurons, training_time, dead_neurons_union=None):
    run_state["epoch"] = curr_epoch
    run_state["training_step"] = curr_step
    run_state["curr_arch_sizes"] = curr_arch_sizes
    run_state["curr_starting_size"] = curr_starting_size
    run_state["curr_reg_param"] = curr_reg_param
    run_state["dropout_key"] = dropout_key
    run_state["decaying_reg_param"] = decaying_reg_param
    run_state["best_accuracy"] = best_acc
    run_state["best_params_count"] = best_params_count
    run_state["best_total_neurons"] = best_total_neurons
    run_state["training_time"] = training_time
    run_state["cumulative_dead_neurons"] = dead_neurons_union

    with open(os.path.join(run_state["model_dir"], "checkpoint_run_state.pkl"), "wb") as f:
        pickle.dump(run_state, f)

    # Update weights
    save_all_pytree_states(run_state["model_dir"], params, state, opt_state)


def jaxpruner_checkpoint_exp(run_state: JaxPrunerRunState, params, state, opt_state, curr_epoch: int, curr_step: int,
                   curr_pruning_density, dropout_key, decaying_reg_param):
    run_state["epoch"] = curr_epoch
    run_state["training_step"] = curr_step
    run_state["curr_pruning_density"] = curr_pruning_density
    run_state["dropout_key"] = dropout_key
    run_state["decaying_reg_param"] = decaying_reg_param

    with open(os.path.join(run_state["model_dir"], "checkpoint_run_state.pkl"), "wb") as f:
        pickle.dump(run_state, f)

    # Update weights
    save_all_pytree_states(run_state["model_dir"], params, state, opt_state)


def signal_handler(signum: int, frame: Optional[FrameType]):  # Taken from: https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
    """Called before the job gets pre-empted or reaches the time-limit.

    This should run quickly. Performing a full checkpoint here mid-epoch is not recommended.
    """
    signal_enum = signal.Signals(signum)
    print(f"Job received a {signal_enum.name} signal!")


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
    elif architecture == 'vgg16':
        if len(sizes) == 2:  # Size can be specified with 2 args
            sizes = [sizes[0], sizes[0], 2 * sizes[0], 2 * sizes[0], 4 * sizes[0], 4 * sizes[0], 4 * sizes[0],
                     8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 8 * sizes[0], 16 * sizes[0], sizes[1],
                     sizes[1], sizes[1]]
    elif "resnet18_proj_instead" in architecture:
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
                     8 * sizes,
                     ]
    elif "resnet18" in architecture:
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
                     # 2*sizes,
                     ]
    elif "resnet19" in architecture:
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
                     # 2*sizes,
                     ]
    elif "resnet50" in architecture:
        if type(sizes) == int:  # Size can be specified with 1 arg, an int
            sizes = [sizes] + [sizes, sizes, sizes*4]*3 + [2*sizes, 2*sizes, 2*sizes*4]*4 + [4*sizes, 4*sizes, 4*sizes*4]*6 + [8*sizes, 8*sizes, 8*sizes*4]*3  # + [16*sizes]
    elif "vit_b" in architecture:  # Cover vit_b_4 and vit_b_16, both having 12 layers
        if type(sizes) == int:  # Size can be specified with 1 arg, an int
            sizes = [sizes,]*12
    elif "grok_model_depth2" in architecture:  # Cover vit_b_4 and vit_b_16, both having 12 layers
        if type(sizes) == int:  # Size can be specified with 1 arg, an int
            sizes = [sizes,]*2
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
    for module_name, module in sys.modules.items():
        if module_name.startswith("jax"):
            if module_name not in ["jax.interpreters.partial_eval"]:
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        try:
                            obj.cache_clear()
                        except:
                            pass
    gc.collect()


def add_comma_in_str(string: str):
    """ Helper fn to format string from hydra before literal eval"""
    string = string.replace("mnist", "'mnist'")
    string = string.replace("fashion mnist", "'fashion mnist'")
    if "srigl" in string:
        string = string.replace("cifar10_srigl", "'cifar10_srigl'")
    else:
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


def get_checkpoint_step(architecture, step):
    """ Return the iteration at which to rewind for pruned reinit. Numbers taken from LTH rewinding litterature."""
    if "resnet18" in architecture:
        return [500, 2000, 10000]  # fix
    elif "vgg16" in architecture:
        return [100]  # fix
    else:
        raise ValueError('No rewinding steps encoded for other architectures rn')


def avg_neuron_magnitude_in_layer(layer_params):
    if 'b' in layer_params.keys():
        def conca_and_norm(arr1, arr2):
            return jnp.linalg.norm(jnp.concatenate([arr1, arr2], axis=None))
        return jnp.mean(jax.vmap(conca_and_norm, in_axes=(1, 0))(layer_params["w"], layer_params['b']))
    else:
        return jnp.mean(jax.vmap(jnp.linalg.norm, in_axes=1)(layer_params["w"]))


def jax_deep_copy(pytree):
    """ALERT: copy.deepcopy relies on pickle which creates the copy on host before transfer to device.
    This function tries to avoid potential memory issue implied by this procedure."""
    return jax.tree_util.tree_map(lambda x: jax.device_put(x), pytree)


def get_init_fn(net, dummy_data):
    """ Return an init_fn function for the given net and data that only depends on a random key"""

    def init_fn(rdm_key):
        return net.init(rdm_key, dummy_data)[0]  # .init() return the tuple (params, state)

    return init_fn


def reformat_dict_config(config: DictConfig):
    """
    Modifies the given Hydra DictConfig object in place,
    setting any values that are the string 'None' to Python's NoneType.

    This function is intended for cleaning up configuration data where
    'None' strings should be interpreted as actual None values, which is
    common in Hydra configurations for experiments.

    Args:
        config (DictConfig): An OmegaConf DictConfig instance representing a Hydra configuration.

    Returns:
        None: The modification is made in place, so there is no return value.
    """
    for key, value in config.items():
        if value == 'None':  # Check if the value is 'None' (string)
            config[key] = None  # Set the value to None (NoneType)
