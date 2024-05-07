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
from typing import Optional, Union, Sequence

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

        masked_dots = jnp.where(jnp.expand_dims(mask, axis=1), -jnp.inf, dots)
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


def causal_attn_mask(seq_len):
    return jnp.triu(jnp.ones((seq_len, seq_len)), k=1) == 1


class InitLayer(hk.Module):
    def __init__(self, vocab_size, max_length, heads, hidden_dim, attn_dim, mlp_dim, dropout=0.1,
                 activation_fn: hk.Module = GeluActivationModule, parent: Optional[hk.Module] = None,):
        super(InitLayer, self).__init__()
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.embeddings = hk.Embed(vocab_size, hidden_dim)
        self.positions = hk.Embed(max_length, hidden_dim)
        self.init_transformer = TransfLayer(hidden_dim, heads, attn_dim, mlp_dim, activation_fn=activation_fn)

    def __call__(self, x):
        x = x.astype('int32')
        # x, attn_mask = inputs
        # attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1)
        # Step 1: Get the mask from causal_attn_mask and expand its dimensions
        attn_mask = causal_attn_mask(x.shape[1])
        attn_mask = jnp.expand_dims(attn_mask, axis=0)  # Similar to unsqueeze(0) in PyTorch

        # Step 2: Repeat the mask along the 0th dimension to match x's 0th dimension (batch size)
        # Note: jnp.tile repeats the whole array, so we specify how many times to repeat along each axis
        repeat_factor = (x.shape[0], 1, 1)  # Repeat 'batch_size' times along the 0th axis, and don't repeat along the 1st and 2nd axes
        attn_mask = jnp.tile(attn_mask, repeat_factor)

        initial_pos = 0
        assert initial_pos+x.shape[1] <= self.max_length, 'sequence too long'
        x = self.embeddings(x) * jnp.sqrt(self.hidden_dim) + self.positions.embeddings[initial_pos:initial_pos+x.shape[1], :]
        # Processing a single transformer block
        outputs = self.init_transformer((x, attn_mask))
        return outputs

    def get_activation_mapping(self):
        return self.init_transformer.get_activation_mapping()

    def get_last_activation_name(self):
        self.init_transformer.get_last_activation_name()


class LastLayer(hk.Module):
    def __init__(self, *, num_classes, parent: Optional[hk.Module] = None,):
        super(LastLayer, self).__init__()
        self.activation_mapping = {}
        if parent:
            self.preceding_activation_name = parent.get_last_activation_name()
        else:
            self.preceding_activation_name = None

        self.mlp_head = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_classes, name='logits')
        ])

    def __call__(self, inputs):
        x, mask = inputs
        x = x.reshape((x.shape[0], -1))
        x = self.mlp_head(x)

        return x, []

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return None


def grokking_base_models(*, vocab_size, max_length, num_classes, heads, hidden_dim, attn_dim, mlp_dim, dropout=0.1,
                         depth=2, activation_fn=GeluActivationModule):

    layers = [[Partial(InitLayer, vocab_size=vocab_size, max_length=max_length, heads=heads, hidden_dim=hidden_dim,
                       attn_dim=attn_dim, mlp_dim=mlp_dim[0], activation_fn=activation_fn)]]
    for i in range(1, depth):  # Already have 1 transformer layer above
        layers.append([Partial(TransfLayer, dim=hidden_dim, heads=heads, attn_dim=attn_dim, mlp_dim=mlp_dim[i],
                               activation_fn=activation_fn)])
    layers.append([Partial(LastLayer, num_classes=num_classes)])

    return layers, None


# def grok_model_depth2(size: Union[int, Sequence[int]],
#                 num_classes: int,
#                 vocab_size: int,
#                 activation_fn: hk.Module = GeluActivationModule,
#                 bn_config: dict = {},
#                 ):
#     if type(size) == int:
#         sizes = [size,]*2
#     else:
#         sizes = size
#
#     return grokking_base_models(
#         vocab_size=vocab_size,
#         max_length=5,
#         num_classes=num_classes,
#         heads=4,
#         hidden_dim=128,
#         attn_dim=32,
#         mlp_dim=sizes,
#         depth=2,
#         activation_fn=activation_fn,
#     )


def grok_models(size: Union[int, Sequence[int]],
                num_classes: int,
                vocab_size: int,
                activation_fn: hk.Module = GeluActivationModule,
                bn_config: dict = {},
                depth: int = 2,
                ):
    if type(size) == int:
        sizes = [size,]*depth
    else:
        sizes = size

    return grokking_base_models(
        vocab_size=vocab_size,
        max_length=5,
        num_classes=num_classes,
        heads=4,
        hidden_dim=128,
        attn_dim=32,
        mlp_dim=sizes,
        depth=depth,
        activation_fn=activation_fn,
    )