"""Experiment to showcase that increasing via reducing minibatch size per gradient update lead to more neuron
deaths"""

import jax
import jax.numpy as jnp
from aim import Run
import time
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional, Any
from ast import literal_eval
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import utils.utils as utl
from utils.utils import build_models
from utils.config import optimizer_choice, dataset_choice, regularizer_choice, architecture_choice

# Experience name -> for aim logger
exp_name = "batch_size_exp"


# Configuration
@dataclass
class ExpConfig:
    training_steps: int = 50001
    report_freq: int = 3000
    record_freq: int = 100
    full_ds_sweep_freq: int = 5000
    lr: float = 1e-3
    batch_size_seq: Any = (1, 32, 128, 512, 2048, 8192, 32768, "full")
    eval_batch_size: int = 512
    death_batch_size: int = 512
    optimizer: str = "adam"
    dataset: str = "mnist"
    classes: int = 10  # Number of classes in the training dataset
    architecture: str = "mlp_3"
    size: Any = 100
    regularizer: Optional[str] = 'None'
    reg_param: float = 1e-4
    init_seed: int = 41


cs = ConfigStore.instance()
# Registering the Config class with the name '_config'.
cs.store(name=exp_name+"_config", node=ExpConfig)


@hydra.main(version_base=None, config_name=exp_name+"_config")
def run_exp(exp_config: ExpConfig) -> None:

    run_start_time = time.time()

    assert exp_config.optimizer in optimizer_choice.keys(), "Currently supported optimizers: " + str(optimizer_choice.keys())
    assert exp_config.dataset in dataset_choice.keys(), "Currently supported datasets: " + str(dataset_choice.keys())
    assert exp_config.regularizer in regularizer_choice, "Currently supported regularizers: " + str(regularizer_choice)
    assert exp_config.architecture in architecture_choice.keys(), "Current architectures available: " + str(architecture_choice.keys())

    if exp_config.regularizer == 'None':
        exp_config.regularizer = None
    if type(exp_config.size) == str:
        exp_config.size = literal_eval(exp_config.size)
    if type(exp_config.batch_size_seq) == str:
        if 'full' in exp_config.batch_size_seq:
            exp_config.batch_size_seq = exp_config.batch_size_seq.replace('full', '')
            exp_config.batch_size_seq = literal_eval(exp_config.batch_size_seq)
            exp_config.batch_size_seq = tuple(list(exp_config.batch_size_seq) + ['full'])
        else:
            exp_config.batch_size_seq = literal_eval(exp_config.batch_size_seq)

    # Logger config
    exp_run = Run(repo="./logs", experiment=exp_name)
    exp_run["configuration"] = OmegaConf.to_container(exp_config)

    # Load the different dataset
    load_data = dataset_choice[exp_config.dataset]

    eval_size = exp_config.eval_batch_size
    train_size, train_eval = load_data(split="train", is_training=False, batch_size=eval_size, cardinality=True)
    test_size, test_eval = load_data(split="test", is_training=False, batch_size=eval_size, cardinality=True)

    death_minibatch_size = exp_config.death_batch_size
    dataset_size, test_death = load_data(split="train", is_training=False,
                                         batch_size=death_minibatch_size, cardinality=True)

    # If full in batch_size seq
    if 'full' in exp_config.batch_size_seq:
        batch_size_seq = tuple([train_size if x == "full" else x for x in exp_config.batch_size_seq])
    else:
        batch_size_seq = exp_config.batch_size_seq

    # Make the network and optimiser
    architecture = architecture_choice[exp_config.architecture](exp_config.size, exp_config.classes)
    net = build_models(*architecture)

    opt = optimizer_choice[exp_config.optimizer](exp_config.lr)

    # Set training/monitoring functions
    loss = utl.ce_loss_given_model(net, regularizer=exp_config.regularizer, reg_param=exp_config.reg_param)
    accuracy_fn = utl.accuracy_given_model(net)
    update_fn = utl.update_given_loss_and_optimizer(loss, opt)
    death_check_fn = utl.death_check_given_model(net)
    scan_len = dataset_size // death_minibatch_size
    scan_death_check_fn = utl.scanned_death_check_fn(death_check_fn, scan_len)
    final_accuracy_fn = utl.create_full_accuracy_fn(accuracy_fn, test_size // eval_size)
    full_train_acc_fn = utl.create_full_accuracy_fn(accuracy_fn, train_size // eval_size)

    for batch_size in batch_size_seq:  # Vary the NN width
        # Time the subrun for the different sizes
        subrun_start_time = time.time()

        # Load training dataset
        train = load_data(split="train", is_training=True, batch_size=batch_size)

        # Reinitialize the params
        params = net.init(jax.random.PRNGKey(exp_config.init_seed), next(train))
        opt_state = opt.init(params)

        total_neurons, total_per_layer = utl.get_total_neurons(exp_config.architecture, exp_config.size)

        for step in range(exp_config.training_steps):
            if step % exp_config.record_freq == 0:
                train_loss = loss(params, next(train_eval))
                train_accuracy = accuracy_fn(params, next(train_eval))
                test_accuracy = accuracy_fn(params, next(test_eval))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                # Periodically print classification accuracy on train & test sets.
                if step % exp_config.report_freq == 0:
                    print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / "
                          f"{test_accuracy:.3f}. Loss: {train_loss:.3f}.")
                dead_neurons = death_check_fn(params, next(test_death))
                # Record some metrics
                dead_neurons_count, _ = utl.count_dead_neurons(dead_neurons)
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons", step=step,
                              context={"batch size": batch_size})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count), name="Live neurons", step=step,
                              context={"batch size": batch_size})
                exp_run.track(test_accuracy, name="Test accuracy", step=step,
                              context={"batch size": batch_size})
                exp_run.track(train_accuracy, name="Train accuracy", step=step,
                              context={"batch size": batch_size})
                exp_run.track(jax.device_get(train_loss), name="Train loss", step=step,
                              context={"batch size": batch_size})

            if step % exp_config.full_ds_sweep_freq == 0:
                dead_neurons = scan_death_check_fn(params, test_death)
                dead_neurons_count, dead_per_layers = utl.count_dead_neurons(dead_neurons)

                del dead_neurons  # Freeing memory
                exp_run.track(jax.device_get(dead_neurons_count), name="Dead neurons; whole training dataset",
                              step=step,
                              context={"batch size": batch_size})
                exp_run.track(jax.device_get(total_neurons - dead_neurons_count),
                              name="Live neurons; whole training dataset",
                              step=step,
                              context={"batch size": batch_size})
                for i, layer_dead in enumerate(dead_per_layers):
                    total_neuron_in_layer = total_per_layer[i]
                    exp_run.track(jax.device_get(layer_dead),
                                  name=f"Dead neurons in layer {i}; whole training dataset", step=step,
                                  context={"batch size": batch_size})
                    exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                                  name=f"Live neurons in layer {i}; whole training dataset", step=step,
                                  context={"batch size": batch_size})
                del dead_per_layers
                train_acc_whole_ds = jax.device_get(full_train_acc_fn(params, train_eval))
                exp_run.track(train_acc_whole_ds, name="Train accuracy; whole training dataset",
                              step=step,
                              context={"batch size": batch_size})

            # Train step over single batch
            params, opt_state = update_fn(params, opt_state, next(train))

        final_accuracy = jax.device_get(final_accuracy_fn(params, test_eval))
        final_dead_neurons = scan_death_check_fn(params, test_death)

        final_dead_neurons_count, final_dead_per_layer = utl.count_dead_neurons(final_dead_neurons)
        del final_dead_neurons  # Freeing memory

        batch_dead_neurons = death_check_fn(params, next(test_death))
        batches_final_live_neurons = [total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0]]
        for i in range(scan_len - 1):
            batch_dead_neurons = death_check_fn(params, next(test_death))
            batches_final_live_neurons.append(total_neurons - utl.count_dead_neurons(batch_dead_neurons)[0])
        batches_final_live_neurons = jnp.stack(batches_final_live_neurons)

        avg_final_live_neurons = jnp.mean(batches_final_live_neurons, axis=0)
        std_final_live_neurons = jnp.std(batches_final_live_neurons, axis=0)

        exp_run.track(jax.device_get(avg_final_live_neurons),
                      name="On average, live neurons after convergence w/r batch size", step=batch_size)
        exp_run.track(jax.device_get(total_neurons - final_dead_neurons_count),
                      name="Live neurons after convergence w/r batch size", step=batch_size)
        for i, layer_dead in enumerate(final_dead_per_layer):
            total_neuron_in_layer = total_per_layer[i]
            exp_run.track(jax.device_get(total_neuron_in_layer - layer_dead),
                          name=f"Live neurons in layer {i} after convergence w/r batch size",
                          step=batch_size)  # Either total_neuron_in_layer/total_neurons
        exp_run.track(final_accuracy,
                      name="Accuracy after convergence w/r batch size", step=batch_size)

        # Making sure compiled fn cache was cleared
        loss._clear_cache()
        accuracy_fn._clear_cache()
        update_fn._clear_cache()
        death_check_fn._clear_cache()

        # Print running time
        print()
        print(f"Running time for batch size {batch_size}: " + str(timedelta(seconds=time.time() - subrun_start_time)))
        print("----------------------------------------------")
        print()

    # Print total runtime
    print()
    print("==================================")
    print("Whole experiment completed in: " + str(timedelta(seconds=time.time() - run_start_time)))


if __name__ == "__main__":
    run_exp()
