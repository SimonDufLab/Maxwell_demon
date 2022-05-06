""" Script to run the experiment trying to measure the effective capacity in overfitting regime with
the final number of live neurons. Knowing that if left to train for a long period neurons keep dying until a plateau is
reached and that accuracy behave in the same manner (convergence/overfitting regime) the set up is as follow:
For a given dataset and a given architecture, vary the width (to increase capacity) and measure the number of live
neurons after reaching the overfitting regime and the plateau. Empirical observation: Even with increased capacity, the
number of live neurons at the end eventually also reaches a plateau."""

import haiku as hk
import jax
import optax

from models.mlp import lenet_var_size
from utils.utils import load_mnist, build_models

if __name__ == "__main__":
    # Load the different dataset

    train = load_mnist("train", is_training=True, batch_size=50)
    train_eval = load_mnist("train", is_training=False, batch_size=500)
    test_eval = load_mnist("test", is_training=False, batch_size=500)
    final_test_eval = load_mnist("test", is_training=False, batch_size=10000)

    test_death = load_mnist("train", is_training=False, batch_size=1000)
    final_test_death = load_mnist("train", is_training=False, batch_size=60000)

    for size in [50]:  # , 100, 250, 500, 750, 1000, 1250, 1500, 2000]:  # Vary the NN width
        # Make the network and optimiser
        architecture = lenet_var_size(size, 10)
        net, net_activations = build_models(architecture)

        opt = optax.adam(1e-3)
        dead_neurons = []
        accuracies = []

        params = net.init(jax.random.PRNGKey(42 - 1), next(train))
        opt_state = opt.init(params)

        print("test")
        print(net.apply(params, next(train_eval)))
        out, activ = net_activations.apply(params, next(train_eval))
        # for ac in activ:
        #     print(ac.shape)
