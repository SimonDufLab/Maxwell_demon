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
        if ('norm' not in _key) and ('bn' not in _key):
            for subkey, subval in params[_key].items():
                _params[_key][subkey] = jnp.zeros_like(subval)
        else:
            for subkey, subval in params[_key].items():
                _params[_key][subkey] = jnp.ones_like(subval)
    return _params


def mask_untargeted_weights(params, targets):
    _params = copy.deepcopy(params)
    layer_keywords = []
    exclude_block = ['scale', 'offset']
    if 'all' in targets:
        layer_keywords.extend(['norm', 'logits', 'bn'])
        exclude_block = []
    else:
        if 'norm' in targets:
            layer_keywords.extend(['norm', 'bn'])
            exclude_block = []
        if 'head' in targets:
            layer_keywords.append('logits')
        if any(_layer_block in targets for _layer_block in ['scale', 'offset']):
            layer_keywords.extend(['norm', 'bn'])
            exclude_block = [_excluded for _excluded in exclude_block if _excluded not in targets]

    for _key, _val in params.items():
        if not any(keyword in _key for keyword in layer_keywords):
            for subkey, subval in params[_key].items():
                _params[_key][subkey] = jnp.zeros_like(subval)
        else:
            for subkey, subval in params[_key].items():
                if subkey in exclude_block:
                    _params[_key][subkey] = jnp.zeros_like(subval)
                else:
                    _params[_key][subkey] = jnp.ones_like(subval)
    return _params


def clip_norm_params(params, scale_min_val=-2.0, scale_max_val=2.0, offset_min_val=-1.0, offset_max_val=1.0):

    def clip_fn(param):
        for key in param:
            if key == 'scale':
                param[key] = jnp.clip(param[key], scale_min_val, scale_max_val)
            elif key == 'offset':
                param[key] = jnp.clip(param[key], offset_min_val, offset_max_val)
            else:
                raise SystemExit
        return param

    # Apply clipping to all normalization layers
    # def recursive_clip(params):
    #     if isinstance(params, dict):
    return {k: clip_fn(v) if (('norm' in k) or ('bn' in k)) else v for k, v in params.items()}
        # else:
        #     return clip_fn(params)

    # return recursive_clip(params)


def modify_scale_params(params):
    """ Multiply by 4 the scale param from normalisation layer, intended for applying a sigmoid function on top"""
    def multiply_scale(param):
        if isinstance(param, dict):
            # If the parameter is a dictionary, check for the 'scale' key and modify it
            if 'scale' in param:
                param['scale'] = param['scale'] * 4
            # Recursively apply the function to nested dictionaries
            return {k: multiply_scale(v) for k, v in param.items()}
        return param

    params_copy = copy.deepcopy(params)
    return jax.tree_util.tree_map(multiply_scale, params_copy, is_leaf=lambda x: isinstance(x, dict))


def split_norm_layers(params):
    """Split normalisation layers from the params dictionary"""
    norm_layers = {}
    rest = {}

    for _key, _val in params.items():
        if 'norm' in _key or 'bn' in _key:
            norm_layers[_key] = _val
        else:
            rest[_key] = _val

    return rest, norm_layers
