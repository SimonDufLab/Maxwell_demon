"""Utl specifics to the grokking experiments"""

import jax
import copy
import jax.numpy as jnp


##############################
# Weight masking utilities
##############################
def mask_ff_init_layer(params, target="wb"):
    assert target in ["wb", "bw", "b", "w"]  # mask out w+b, w or b
    _params = copy.deepcopy(params)
    for _key, _val in params.items():
        if 'init_layer' in _key:
            if '/feed_forward/~/linear' in _key:
                for subkey, subval in params[_key].items():
                    if subkey in target:
                        _params[_key][subkey] = jnp.zeros_like(subval)
    return _params


def mask_ff_last_layer(params, target="wb"):
    assert target in ["wb", "bw", "b", "w"]  # mask out w+b, w or b
    _params = copy.deepcopy(params)
    for _key, _val in params.items():
        if 'init_layer' in _key:
            if 'model_and_activations/transf_layer/' in _key:
                for subkey, subval in params[_key].items():
                    if subkey in target:
                        _params[_key][subkey] = jnp.zeros_like(subval)
    return _params


def mask_all_except_norm(params):
    _params = copy.deepcopy(params)
    for _key, _val in params.items():
        if 'norm' not in _key:
            for subkey, subval in params[_key].items():
                _params[_key][subkey] = jnp.zeros_like(subval)
    return _params


def mask_all_except_norm_and_output(params):
    _params = copy.deepcopy(params)
    for _key, _val in params.items():
        if ('norm' not in _key) and ('logits' not in _key):
            for subkey, subval in params[_key].items():
                _params[_key][subkey] = jnp.zeros_like(subval)
    return _params
