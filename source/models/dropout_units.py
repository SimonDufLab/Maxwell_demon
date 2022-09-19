""" Base unit used to build the dropout layer. Workaround to use it in our """
from jax.tree_util import Partial
import haiku as hk
from typing import Optional


class Base_Dropout(hk.Module):
    """Simply a BN layer, with is_training option pre-specified"""
    def __init__(
            self,
            dropout_rate: bool,
            name: Optional[str] = None):
        super().__init__(name=name)
        self.drop = Partial(hk.dropout, rate=dropout_rate)

    def __call__(self, x):
        x = self.drop(rng=hk.next_rng_key(), x=x)
        return x
