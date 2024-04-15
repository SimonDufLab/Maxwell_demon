"""Utl specifics to the grokking experiments"""

import jax
import jax.numpy as jnp


##############################
# Weight masking utilities
##############################
def mask_ff_init_layer(params, target="wb"):
    assert target in ["wb", "bw", "b", "w"]  # mask out w+b, w or b
    pass
