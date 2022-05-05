import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from jax.tree_util import Partial

from typing import Any, Iterator, Mapping, Tuple, Union

OptState = Any
Batch = Mapping[int, np.ndarray]


def death_check_given_model(model):
    @jax.jit
    def _death_check(_params: hk.Params, _batch: Batch) -> jnp.ndarray:
        _, activations = model.apply(_params, _batch)
        return jax.tree_map(Partial(jnp.sum, axis=0), activations)

    return _death_check


def accuracy_given_model(model):
    @jax.jit
    def _accuracy(_params: hk.Params, _batch: Batch) -> jnp.ndarray:
        predictions = model.apply(_params, _batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == _batch[1])

    return _accuracy


def loss_given_model(model):
    @jax.jit
    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        logits = model.apply(params, batch)
        labels = jax.nn.one_hot(batch[1], 10)

        # l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        cdg_loss = 0.5 * sum(jnp.sum(jnp.power(jnp.clip(p, 0), 2)) for p in jax.tree_leaves(params))
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1e-4 * cdg_loss
    return loss


def update_given_model_and_optimizer(model, optimizer):
    """Learning rule (stochastic gradient descent)."""
    loss = loss_given_model(model)

    @jax.jit
    def _update(_params: hk.Params, _opt_state: OptState, _batch: Batch) -> Tuple[hk.Params, OptState]:
        grads = jax.grad(loss)(_params, _batch)
        updates, _opt_state = optimizer.update(grads, _opt_state)
        new_params = optax.apply_updates(_params, updates)
        return new_params, _opt_state

    return _update


def load_mnist(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
    subset: Union[None, int] = None,
) -> Iterator[Batch, None, None]:
  """Loads the dataset as a generator of batches.
    subset: If only want a subset, number of classes to build the subset from
    """
  ds = tfds.load("mnist:3.*.*", split=split, as_supervised=True).cache().repeat()
  if subset:
      assert subset < 10, "subset must be smaller than 10"
      indices = np.random.choice(10, subset, replace=False)

      def filter_fn(image, label):
        return tf.reduce_any(indices == int(label))
      ds = ds.filter(filter_fn)  # Only tae the randomly selected subset

  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))
