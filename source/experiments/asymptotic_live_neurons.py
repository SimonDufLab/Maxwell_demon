""" Script to run the experiment trying to measure the effective capacity in overfitting regime with
the final number of live neurons. Knowing that if left to train for a long period neurons keep dying until a plateau is
reached and that accuracy behave in the same manner (convergence/overfitting regime) the set up is as follow:
For a given dataset and a given architecture, vary the width (to increase capacity) and measure the number of live
neurons after reaching the overfitting regime and the plateau. Empirical observation: Even with increased capacity, the
number of live neurons at the end eventually also reaches a plateau."""

from source.utils.module_utils import build_models

from source.models.mlp import lenet_var_size
from source.utils.utils import load_mnist

# Load the different dataset

train = load_mnist("train", is_training=True, batch_size=50)
train_eval = load_mnist("train", is_training=False, batch_size=500)
test_eval = load_mnist("test", is_training=False, batch_size=500)
final_test_eval = load_mnist("test", is_training=False, batch_size=10000)

test_death = load_mnist("train", is_training=False, batch_size=1000)
final_test_death = load_mnist("train", is_training=False, batch_size=6e4)

for size in [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000]: # Vary the NN width
    # Make the network and optimiser
    architecture = lenet_var_size(size, 10)
    net, net_activations = build_models(architecture)
