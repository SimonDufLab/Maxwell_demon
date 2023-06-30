""" Scores functions used to determine which weights/neurons to prune"""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

import utils.utils as utl


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
def iterative_single_shot_pruning(target_density, params, state, opt_state, acti_map, neuron_states,
                                  pruning_score_fn, get_architecture, test_loss_fn, get_test_loss_fn,
                                  dataset, num_batches=10, start=0.5, pruning_steps=5):
    """Introduced by https://arxiv.org/pdf/2006.00896.pdf
    Iteratively applies the pruning criterion until the target density is obtained. All pruning is done at the same
    time step, so this it still single-shot pruning
    """
    if target_density <= start or pruning_steps == 1:
        steps = [target_density]
    else:
        steps = [target_density - (target_density - start) * (0.5 ** i) for i in range(pruning_steps + 1)] + [
            target_density]
    pruned = [0] + steps
    iterative_pruning_densities = [(steps[i] - pruned[i])/(1-pruned[i]+1e-8) for i in range(len(steps))]

    for density in iterative_pruning_densities:
        neuron_scores = pruning_score_fn(params, state, test_loss_fn, dataset, num_batches)
        neuron_states.update(score_to_neuron_mask(density, neuron_scores))
        params, opt_state, state, new_sizes = utl.prune_params_state_optstate(params,
                                                                              acti_map,
                                                                              neuron_states,
                                                                              opt_state,
                                                                              state)

        architecture = get_architecture(new_sizes)
        net, raw_net = utl.build_models(*architecture)
        test_loss_fn = get_test_loss_fn(net)

    return neuron_states, params, opt_state, state, new_sizes


##############################
# Structured scores
##############################
def snap_score(params, state, test_loss, dataloader, scan_len, with_dropout=False):
    """Equivalent to gate saliency score; i.e. the derivative w.r. to gate constant"""
    gate_states, rest = split_state(state)

    def loss_wr_gate(_gate_states, _batch):
        _state = recombine_state_dicts(_gate_states, rest)
        return test_loss(params, _state, _batch)

    def grad_fn(_batch):
        gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
        return gate_grad

    score = grad_fn(next(dataloader))
    for i in range(scan_len - 1):
        curr_score = grad_fn(next(dataloader))
        score = jax.tree_map(jnp.add, score, curr_score)
    abs_total_score = jax.tree_map(jnp.abs, score)  # Return abs of score

    return {top_key: list(low_dict.values())[0] for top_key, low_dict in abs_total_score.items()}  # Remove inner dict


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

    def pre_score_fn(_gate_states, _batch_grad, _batch):
    # def pre_score_fn(_batch_grad, _gate_states, _batch):
        gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
        _batch_grad = jax.flatten_util.ravel_pytree(_batch_grad)[0]
        gate_grad = jax.flatten_util.ravel_pytree(gate_grad)[0]

        return jnp.sum(gate_grad*_batch_grad)

    def batch_grad(_batch):
        gate_grad = jax.grad(loss_wr_gate)(gate_states, _batch)
        return gate_grad

    # forward-over-reverse hvp
    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f), primals, tangents)[1]

    def gate_grad_fn(_batch):
        gate_batch_grad = batch_grad(_batch)
        # batch_score = jax.grad(pre_score_fn)(gate_states, gate_batch_grad, _batch)
        # batch_score = jax.grad(pre_score_fn)(gate_batch_grad, gate_states, _batch)
        partial_fn = Partial(loss_wr_gate, _batch=_batch)
        batch_score = hvp(partial_fn, (gate_states,), (gate_batch_grad,))

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
def prune_before_training(target_density, curr_weights, init_weights, prev_dist):
    """Always return true, so pruning is done at step 0, before beginning training"""
    return True, 0.0


def test_earlycrop_pruning_step(target_density, curr_weights, init_weights, prev_dist):
    """ Implement the EarlyCrop pruning time score: : https://arxiv.org/pdf/2206.10451.pdf

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


def modulate_target_density(target_density, weights_epoch_1, init_weights):
    """ While EarlyCrop paper say they use 1-\rho for threshold calculation about when to prune, the implementation
    is modulated by the weights displacement between epoch 1 and initialisation divided by initialisation norm.
    See: https://github.com/johnrachwan123/Early-Cropression-via-Gradient-Flow-Preservation/blob/ab078b84f80e2905807926967945f6bdbe3294c1/Image%20Classification/models/trainers/DefaultTrainer.py#L256"""

    displacement = jnp.linalg.norm(jax.flatten_util.ravel_pytree(jax.tree_map(jnp.subtract, weights_epoch_1,
                                                                              init_weights))[0])
    init_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(init_weights)[0])
    return 1 - ((displacement/init_norm) * jnp.minimum(1-target_density-0.1, 0.99))

