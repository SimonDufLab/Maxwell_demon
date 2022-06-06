""" Models definition for convolutional nets architecture. Defined fitting requirements of repo"""
import haiku as hk
import jax
from jax.tree_util import Partial


def conv_3_2(sizes, number_classes, dim=2):
    """ Build a MLP with 2 hidden layers similar to popular LeNet, but with varying number of hidden units
    sizes: Tuple containing output_channel of first conv layer and number of features in first fully connected
    """
    def act():
        return jax.nn.relu
    if dim==1:
        conv_fn = hk.Conv1D
    if dim==2:
        conv_fn = hk.Conv2D
    else:
        raise Exception("Convnet dimension restricted to 1 or 2")
    max_pool = Partial(hk.MaxPool, window_shape=(2, 2), strides=2, padding="VALID")
    first_max_pool = Partial(hk.MaxPool, window_shape=(4, 4), strides=4, padding="VALID")

    layer_1 = [Partial(conv_fn, sizes[0], 5), act]
    layer_2 = [first_max_pool, Partial(conv_fn, 2*sizes[0], 3), act]
    layer_3 = [max_pool, Partial(conv_fn, 4*sizes[0], 3), act]
    layer_4 = [max_pool, hk.Flatten, Partial(hk.Linear, sizes[1]), act]
    layer_5 = [Partial(hk.Linear, number_classes)]

    return [layer_1, layer_2, layer_3, layer_4, layer_5]
