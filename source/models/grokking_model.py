""" Adapted from https://github.com/Sea-Snell/grokking/blob/main/grokk_replica/transformer.py"""

from functools import partial

import haiku as hk
from haiku import PRNGSequence

import jax
from jax import random, nn
from jax.tree_util import Partial

import jax.numpy as jnp

# from einops import rearrange, repeat

from utils.utils import GeluActivationModule
from models.vit import IdentityLayer, PreNorm, FeedForward
from typing import Optional

# Helper functions

LayerNorm = partial(hk.LayerNorm, create_scale=True, create_offset=False, axis=-1)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MaskedAttention(hk.Module):
    def __init__(self, hidden_dim, heads=4, attn_dim=64):
        dim_head = attn_dim
        dim = hidden_dim
        super(MaskedAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = hk.Linear(output_size=inner_dim * 3, with_bias=False)
        self.to_out = hk.Linear(dim) if project_out else IdentityLayer()

    def __call__(self, x, mask):
        qkv = self.to_qkv(x)
        qkv = jnp.split(qkv, 3, axis=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q, k, v = [jnp.reshape(tensor, (tensor.shape[0], tensor.shape[1], self.heads, -1)) for tensor in qkv]
        q, k, v = [jnp.transpose(tensor, (0, 2, 1, 3)) for tensor in (q, k, v)]

        dots = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        masked_dots = jnp.where(mask, -jnp.inf, dots)
        attn = nn.softmax(masked_dots, axis=-1)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)

        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = jnp.reshape(out, (out.shape[0], out.shape[1], -1))

        out = self.to_out(out)

        # out = hk.dropout(hk.next_rng_key(), rate=0.0, x=out)
        return out


class TransfLayer(hk.Module):
    def __init__(self, dim, heads, attn_dim, mlp_dim, activation_fn: hk.Module = GeluActivationModule,
                 parent: Optional[hk.Module] = None):
        super(TransfLayer, self).__init__()
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None

        self.attn = MaskedAttention(dim, heads=heads, attn_dim=attn_dim)
        self.ff = FeedForward(dim, mlp_dim, activation_module=activation_fn, parent=self.preceding_activation_name)
        self.layer_norm1 = LayerNorm()
        self.layer_norm2 = LayerNorm()

    def __call__(self, inputs):
        x, mask = inputs
        x_norm = self.layer_norm1(x)
        x = self.attn(x_norm, mask) + x
        x_norm = self.layer_norm2(x)
        mlp_out, activations = self.ff(x_norm)

        return (mlp_out + x, mask), activations

    def get_activation_mapping(self):
        return self.ff.get_activation_mapping()

    def get_last_activation_name(self):
        self.ff.get_last_activation_name()


class InitLayer(hk.Module):
    def __init__(self, vocab_size, max_length, heads, hidden_dim, attn_dim, mlp_dim, dropout=0.1,
                 activation_fn: hk.Module = GeluActivationModule, parent: Optional[hk.Module] = None,):
        super(InitLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.embeddings = hk.Embed(vocab_size, hidden_dim)
        self.positions = hk.Embed(max_length, hidden_dim)
        self.init_transformer = TransfLayer(hidden_dim, heads, attn_dim, mlp_dim, activation_fn=activation_fn)

    def __call__(self, inputs):
        x, attn_mask = inputs
        initial_pos = 0
        assert initial_pos+x.shape[1] <= self.max_length, 'sequence too long'
        x = self.embeddings(x) * jnp.sqrt(self.hidden_dim) + self.positions.weight[initial_pos:initial_pos+x.shape[1], :]
        # Processing a single transformer block
        outputs = self.init_transformer((x, attn_mask))
        return outputs


class LastLayer(hk.Module):
    def __init__(self, *, num_classes, pool='cls', parent: Optional[hk.Module] = None,):
        super(LastLayer, self).__init__()
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None

        self.pool = pool

        self.mlp_head = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_classes)
        ])

    def __call__(self, inputs):
        x, mask = inputs
        if self.pool == 'mean':
            x = jnp.mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x, []