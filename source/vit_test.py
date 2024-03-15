import haiku as hk
from jax import random
import jax.numpy as jnp

from utils.utils import build_models
from models.vit import vit_base_models

if __name__ == '__main__':
    v, _ = build_models(vit_base_models(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
    ))

    key = hk.PRNGSequence(42)

    img = (random.normal(next(key), (16, 224, 224, 3)), 0)

    params, state = v.init(next(key), img)
    logits, _ = v.apply(params, state, img)
    print(logits.shape)
    print(jnp.sum(logits, axis=1))  # Should return !=1 if returning logits as expected
