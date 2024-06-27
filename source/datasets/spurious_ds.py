import tensorflow as tf
from typing import Optional
from jax.tree_util import Partial

from utils.utils import load_tf_dataset


def col_mnist_transform(images, labels, sp_noise, core_noise):
    # Convert to TensorFlow tensors if not already
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)

    # 2x subsample for computational convenience
    images = tf.cast(tf.squeeze(images[::2, ::2, :]), tf.int32)

    # Assign a binary label based on the digit; flip label with probability 'core_noise'
    labels = tf.cast(labels < 5, tf.int32)
    flip_label = tf.random.uniform(tf.shape(labels)) < core_noise
    labels = tf.math.abs(labels - tf.cast(flip_label, tf.int32))

    # Assign a color based on the label; flip the color with probability 'sp_noise'
    flip_color = tf.random.uniform(tf.shape(labels)) < sp_noise
    colors = tf.math.abs(labels - tf.cast(flip_color, tf.int32))

    # Convert grayscale images to RGB by stacking
    images = tf.stack([images, images, images], axis=-1)

    # Zero out one channel based on color to simulate color flip
    mask = tf.one_hot((1 - colors), depth=3, on_value=0, off_value=1)
    mask = tf.reshape(mask, (1, 1, 3))
    images *= mask

    # Calculate groups for further analysis or training
    # groups = 4 * tf.cast(flip_label, tf.float32) + 2 * tf.cast(flip_color, tf.float32) + labels

    return images, tf.cast(labels, tf.int32)#, tf.cast(flip_color, tf.float32), groups


def load_colmnist_tf(split: str, is_training, batch_size, sp_noise_train, sp_noise_test, core_noise, other_bs=None,
                     subset=None, transform=True, cardinality=False,
                     noisy_label=0, permuted_img_ratio=0, gaussian_img_ratio=0, augment_dataset=False,
                     normalize: bool = False, reduced_ds_size: Optional[int] = None):
    if split == 'test':
        sp_noise = sp_noise_test
    else:
        sp_noise = sp_noise_train

    return load_tf_dataset("mnist", split=split, is_training=is_training, batch_size=batch_size,
                           other_bs=other_bs, subset=subset, transform=transform, cardinality=cardinality,
                           noisy_label=noisy_label, permuted_img_ratio=permuted_img_ratio,
                           gaussian_img_ratio=gaussian_img_ratio, data_augmentation=augment_dataset,
                           normalize=normalize, reduced_ds_size=reduced_ds_size,
                           transform_ds=Partial(col_mnist_transform, sp_noise=sp_noise, core_noise=core_noise))
