""" Script to run the experiment trying to measure the effective capacity in overfitting regime with
the final number of live neurons. Knowing that if left to train for a long period neurons keep dying until a plateau is
reached and that accuracy behave in the same manner (convergence/overfitting regime) the set up is as follow:
For a given dataset and a given architecture, vary the width (to increase capacity) and measure the number of live
neurons after reaching the overfitting regime and the plateau. Empirical observation: Even with increased capacity, the
number of live neurons at the end eventually also reaches a plateau."""

import haiku as hk
import jax
import optax
import matplotlib.pyplot as plt

from models.mlp import lenet_var_size
import utils.utils as utl
from utils.utils import load_mnist, build_models

if __name__ == "__main__":

    training_steps = 20001
    report_freq = 500

    # Load the different dataset
    train = load_mnist(is_training=True, batch_size=50)
    # tf_train = load_mnist_tf("train", is_training=True, batch_size=50)

    train_eval = load_mnist(is_training=True, batch_size=500)
    test_eval = load_mnist(is_training=False, batch_size=500)
    final_test_eval = load_mnist(is_training=False, batch_size=10000)

    test_death = load_mnist(is_training=True, batch_size=1000)
    final_test_death = load_mnist(is_training=True, batch_size=60000)

    # Recording over all widths
    live_neurons = []
    size_arr = []
    f_acc = []

    for size in [50]:  # , 100, 250, 500, 750, 1000, 1250, 1500, 2000]:  # Vary the NN width
        # Make the network and optimiser
        architecture = lenet_var_size(size, 10)
        net = build_models(architecture)

        opt = optax.adam(1e-3)
        dead_neurons_log = []
        accuracies_log = []

        # Set training/monitoring functions
        loss = utl.ce_loss_given_model(net, regularizer="cdg_l2", reg_param=1e-4)
        accuracy_fn = utl.accuracy_given_model(net)
        update_fn = utl.update_given_loss_and_optimizer(loss, opt)

        params = net.init(jax.random.PRNGKey(42 - 1), next(train))
        opt_state = opt.init(params)

        for step in range(training_steps):
            if step % report_freq == 0:
                # Periodically evaluate classification accuracy on train & test sets.
                train_loss = loss(params, next(train_eval))
                train_accuracy = accuracy_fn(params, next(train_eval))
                test_accuracy = accuracy_fn(params, next(test_eval))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                      f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")

                dead_neurons = utl.death_check_given_model(net)(params, next(test_death))

                # Record some metrics
                dead_neurons_log.append(utl.count_dead_neurons(dead_neurons))
                accuracies_log.append(test_accuracy)

            # Train step over single batch
            params, opt_state = update_fn(params, opt_state, next(train))

        total_neurons = size + 3*size  # TODO: Adapt to other NN than 2-hidden MLP
        final_accuracy = jax.device_get(accuracy_fn(params, next(final_test_eval)))
        size_arr.append(total_neurons)
        live_neurons.append(total_neurons - dead_neurons_log[-1])
        f_acc.append(final_accuracy)
        plt.plot(dead_neurons_log)
        plt.show()
