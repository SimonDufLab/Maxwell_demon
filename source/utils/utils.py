import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import Partial

from typing import Any, Mapping, Tuple

OptState = Any
Batch = Mapping[int, np.ndarray]


def death_check(model, params: hk.Params, batch: Batch):
    @jax.jit
    def _death_check(_params: hk.Params, _batch: Batch) -> jnp.ndarray:
        _, activations = model.apply(_params, _batch)
        return jax.tree_map(Partial(jnp.sum, axis=0), activations)

    return _death_check(params, batch)


def accuracy(model, params: hk.Params, batch: Batch, shift=0):
    @jax.jit
    def _accuracy(_params: hk.Params, _batch: Batch, _shift=0) -> jnp.ndarray:
        predictions = model.apply(_params, _batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == _batch[1]-_shift)

    return _accuracy(params, batch, shift)


def loss_given_model(model):
    @jax.jit
    def loss(params: hk.Params, batch: Batch, shift=0) -> jnp.ndarray:
        logits = model.apply(params, batch)
        labels = jax.nn.one_hot(batch[1] - shift, 10)

        # l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        cdg_loss = 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_leaves(params))
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1e-4 * cdg_loss
    return loss


def update(model, params: hk.Params, optimizer: Any, opt_state: OptState, batch: Batch,
           shift=0) -> Tuple[hk.Params, OptState]:
    """Learning rule (stochastic gradient descent)."""
    loss = loss_given_model(model)

    @jax.jit
    def _update(_params: hk.Params, _opt_state: OptState, _batch: Batch,
                _shift=0) -> Tuple[hk.Params, OptState]:
        grads = jax.grad(loss)(_params, _batch, _shift)
        updates, _opt_state = optimizer.update(grads, _opt_state)
        new_params = optax.apply_updates(_params, updates)
        return new_params, _opt_state

    return _update(params, opt_state, batch, shift)
