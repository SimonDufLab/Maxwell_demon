""" Adapted from https://github.com/conceptofmind/ViT-haiku/blob/main/vit.py"""

from functools import partial

import haiku as hk
from haiku import PRNGSequence

import jax
from jax import random, nn

import jax.numpy as jnp

# from einops import rearrange, repeat

from utils.utils import GeluActivationModule
from typing import Optional

# Helper functions

LayerNorm = partial(hk.LayerNorm, create_scale=True, create_offset=False, axis=-1)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Module classes

class IdentityLayer(hk.Module):
    def __call__(self, x):
        x = hk.Sequential([])
        return x


class PreNorm(hk.Module):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNorm()
        self.fn = fn

    def __call__(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(hk.Module):
    def __init__(self, dim, hidden_dim, preceding_activation_name: Optional[str] = None):
        super(FeedForward, self).__init__()
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name

        self.linear1 = hk.Linear(hidden_dim)
        self.linear2 = hk.Linear(dim)
        self.gelu = GeluActivationModule()

    def __call__(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        activation = x
        # x = hk.dropout(hk.next_rng_key(), rate=0.0, x=x)
        x = self.linear2(x)
        # x = hk.dropout(hk.next_rng_key(), rate=0.0, x=x)
        return x, [activation,]

    def get_activation_mapping(self):
        return self.activation_mapping

    def get_last_activation_name(self):
        return self.last_act_name


class Attention(hk.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = hk.Linear(output_size=inner_dim * 3, with_bias=False)
        self.to_out = hk.Linear(dim) if project_out else IdentityLayer()

    def __call__(self, x):
        qkv = self.to_qkv(x)
        qkv = jnp.split(qkv, 3, axis=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q, k, v = [jnp.reshape(tensor, (tensor.shape[0], tensor.shape[1], self.heads, -1)) for tensor in qkv]
        q, k, v = [jnp.transpose(tensor, (0, 2, 1, 3)) for tensor in (q, k, v)]

        dots = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = nn.softmax(dots, axis=-1)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)

        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = jnp.reshape(out, (out.shape[0], out.shape[1], -1))

        out = self.to_out(out)

        # out = hk.dropout(hk.next_rng_key(), rate=0.0, x=out)
        return out


class TransfLayer(hk.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, preceding_activation_name: Optional[str] = None):
        super(TransfLayer, self).__init__()
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name

        self.attn = PreNorm(Attention(dim, heads=heads, dim_head=dim_head))
        self.ff = FeedForward(dim, mlp_dim, preceding_activation_name=preceding_activation_name)
        self.layer_norm = LayerNorm()

    def __call__(self, x):
        x = self.attn(x)
        x = self.layer_norm(x)

        return self.ff(x)

    def get_activation_mapping(self):
        return self.ff.get_activation_mapping()

    def get_last_activation_name(self):
        self.ff.get_last_activation_name()


class Transformer(hk.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(FeedForward(dim, mlp_dim))
            ])

    def __call__(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class VitFirstLayer(hk.Module):
    def __init__(self, *, image_size, patch_size, dim, heads, mlp_dim, pool='cls',
                 dim_head=64):
        super(VitFirstLayer, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = hk.Linear(dim)

        self.pos_embedding = hk.get_parameter('pos_embedding', shape=[1, num_patches + 1, dim], init=jnp.zeros)
        self.cls_token = hk.get_parameter('cls_token', shape=[1, 1, dim], init=jnp.zeros)

        self.init_transformer = TransfLayer(dim, heads, dim_head, mlp_dim)

    def __call__(self, img):

        # img = rearrange(img, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
        b, h, w, c = img.shape
        p1, p2 = self.patch_height, self.patch_width

        img = jnp.reshape(img, (b, h // p1, p1, w // p2, p2, c))
        img = jnp.transpose(img, (0, 1, 3, 2, 4, 5))
        img = jnp.reshape(img, (b, (h // p1) * (w // p2), p1 * p2 * c))

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        cls_tokens = jnp.tile(self.cls_token[None, :, :], (b, 1, 1))

        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x = hk.dropout(hk.next_rng_key(), rate=0.0, x=x)

        x = self.init_transformer(x)

        return x

    def get_activation_mapping(self):
        return self.init_transformer.get_activation_mapping()

    def get_last_activation_name(self):
        self.init_transformer.get_last_activation_name()


class VitLastLayer(hk.Module):
    def __init__(self, *, num_classes, pool='cls', preceding_activation_name: Optional[str] = None,):
        super(VitLastLayer, self).__init__()
        self.activation_mapping = {}
        self.preceding_activation_name = preceding_activation_name

        self.pool = pool

        self.mlp_head = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_classes)
        ])

    def __call__(self, x):
        if self.pool == 'mean':
            x = jnp.mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x


class VitBase(hk.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64):
        super(VitBase, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = hk.Linear(dim)

        self.pos_embedding = hk.get_parameter('pos_embedding', shape=[1, num_patches + 1, dim], init=jnp.zeros)
        self.cls_token = hk.get_parameter('cls_token', shape=[1, 1, dim], init=jnp.zeros)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = pool

        self.mlp_head = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_classes)
        ])

    def __call__(self, img):

        # img = rearrange(img, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width)
        b, h, w, c = img.shape
        p1, p2 = self.patch_height, self.patch_width

        img = jnp.reshape(img, (b, h // p1, p1, w // p2, p2, c))
        img = jnp.transpose(img, (0, 1, 3, 2, 4, 5))
        img = jnp.reshape(img, (b, (h // p1) * (w // p2), p1 * p2 * c))

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        cls_tokens = jnp.tile(self.cls_token[None, :, :], (b, 1, 1))

        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x = hk.dropout(hk.next_rng_key(), rate=0.0, x=x)

        x = self.transformer(x)

        if self.pool == 'mean':
            x = jnp.mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x


def ViT(**kwargs):
    @hk.transform
    def inner(img):
        return VitBase(**kwargs)(img)

    return inner


if __name__ == '__main__':
    v = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
    )

    key = PRNGSequence(42)

    img = random.normal(next(key), (1, 224, 224, 3))

    params = v.init(next(key), img)
    logits = v.apply(params, next(key), img)
    print(logits.shape)