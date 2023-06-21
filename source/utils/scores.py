""" Scores functions used to determine which weights/neurons to prune"""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial


##############################
# Utilities
##############################
def split_state(d):
    """Split the state dictionary to isolate the gate constant parameter in activation modules"""
    gates_state = {k: v for k, v in d.items() if 'activation_module' in k}
    rest = {k: v for k, v in d.items() if 'activation_module' not in k}
    return gates_state, rest


def recombine_state_dicts(d1, d2):
    """Recombine together the split state dicts"""
    return {**d1, **d2}


def score_to_neuron_mask(desired_sparsity, score_dict):
    """ Transform the score dictionary over the activations module state parameters to a mask over neurons"""
    flat_scores, _ = ravel_pytree(score_dict)
    split_value = jnp.percentile(flat_scores, desired_sparsity*100)

    gate_bool_mask = jax.tree_map(lambda x: x < split_value, score_dict)  # True for dead neurons

    return gate_bool_mask


##############################
# Structured pruning
##############################


##############################
# Structured scores
##############################
def early_crop_score(params, state, test_loss, dataloader, scan_len, with_dropout=False):
    """ Calculate the score used by early crop method: https://arxiv.org/pdf/2206.10451.pdf

    Try to preserve the gradient flow by scoring each node via a gating constant parameter, stored in state"""

    # if with_dropout:
    #     dropout_key = jax.random.PRNGKey(0)  # dropout rate is zero during death eval
    #     model_apply_fn = Partial(model.apply, rng=dropout_key)
    # else:
    #     model_apply_fn = model.apply

    # split state
    gate_states, rest = split_state(state)

    def loss_wr_gate(_gate_states, _batch):
        _state = recombine_state_dicts(_gate_states, rest)
        return test_loss(params, _state, _batch)

    # def pre_score_fn(_gate_states, _batch_grad, _batch):
    def pre_score_fn(_batch_grad, _gate_states, _batch):
        gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
        _batch_grad = jax.flatten_util.ravel_pytree(_batch_grad)[0]
        gate_grad = jax.flatten_util.ravel_pytree(gate_grad)[0]

        return jnp.sum(gate_grad*_batch_grad)

    def batch_grad(_batch):
        gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
        return gate_grad

    def gate_grad_fn(_batch):
        gate_batch_grad = batch_grad(_batch)
        # batch_score = jax.grad(pre_score_fn)(gate_states, gate_batch_grad, _batch)
        batch_score = jax.grad(pre_score_fn)(gate_batch_grad, gate_states, _batch)

        return batch_score

    score = gate_grad_fn(next(dataloader))
    for i in range(scan_len-1):
        curr_score = gate_grad_fn(next(dataloader))
        score = jax.tree_map(jnp.add, score, curr_score)
    abs_total_score = jax.tree_map(jnp.abs, score)  # Return abs of score

    return {top_key: list(low_dict.values())[0] for top_key, low_dict in abs_total_score.items()}  # Remove inner dict


##############################
# When to prune
##############################
# @jax.jit
def test_earlycrop_pruning_step(target_density, curr_weights, init_weights, prev_dist):
    """ Implment the earlycrop pruning time score: : https://arxiv.org/pdf/2206.10451.pdf

        Tries to detect when training phase enter lazy kernel regime by measuring the relative weight change
        between two epochs.

        target_density: in [0, 1]; defines the threshold (th)
        curr_weights: parameters at current epoch
        init_weights: initial params
        prev_dist: Euclidean distance between params at initialization and from previous epoch"""

    th = 1 - target_density

    curr_dist = jnp.linalg.norm(jax.flatten_util.ravel_pytree(jax.tree_map(jnp.subtract, curr_weights, init_weights))[0])
    norm_factor = jnp.linalg.norm(jax.flatten_util.ravel_pytree(init_weights)[0])

    return 0 < (jnp.abs(curr_dist-prev_dist)/(norm_factor + 1e-6)) < th, curr_dist
    # return (jnp.abs(curr_dist - prev_dist) / (norm_factor + 1e-6)) < th, curr_dist  # TODO: erase after debugging
