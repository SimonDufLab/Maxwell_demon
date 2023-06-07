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


def score_to_neuron_mask(desired_sparsity, score_dict, score_to_neuron_mapping):
    """ Transform the score dictionnary over the activations module state parameters to a mask over neurons"""
    flat_scores, _ = ravel_pytree(score_dict)
    split_value = jnp.percentile(flat_scores, desired_sparsity*100)

    gate_bool_mask = jax.tree_map(lambda x: x < split_value, score_dict)  # True for dead neurons

    pass  # TODO: map the mask over the gate to a mask over all layer neurons


##############################
# Structured pruning
##############################
def early_crop_score(params, state, model, test_loss, dataloader, scan_len, with_dropout=False):
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

    def pre_score_fn(_gate_states, _batch_grad, _batch):
        gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
        return jnp.sum(gate_grad*_batch_grad)

    def batch_grad(_batch):
        gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
        return gate_grad

    def gate_grad_fn(_batch):
        gate_batch_grad = batch_grad(_batch)
        batch_score = jax.grad(pre_score_fn)(gate_states, gate_batch_grad, _batch)
        return batch_score

    score = gate_grad_fn(next(dataloader))
    for i in range(scan_len-1):
        curr_score = gate_grad_fn(next(dataloader))
        score = jax.tree_map(jnp.add, score, curr_score)
    return jax.tree_map(jnp.abs, score)  # Return abs of score
