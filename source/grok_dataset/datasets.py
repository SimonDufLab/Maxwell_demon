"""Inspired from https://github.com/Sea-Snell/grokking/blob/main/grokk_replica/datasets.py, with addition for
compatibilty"""

import abc
import random
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from collections.abc import Iterable
from itertools import permutations
from typing import Set, Optional, Sequence


def get_group_elements_and_output_fn(dataset, p, k):
    if dataset == "mod_sum_dataset":
        group_elements1, group_elements2 = set(range(p)), set(range(p))

        def fetch_output(a, b):
            return (a + b) % p
    elif dataset == "mod_subtract_dataset":
        group_elements1, group_elements2 = set(range(p)), set(range(p))

        def fetch_output(a, b):
            return (a - b) % p
    elif dataset == "mod_division_dataset":
        group_elements1, group_elements2 = set(range(p)), set(range(1, p))

        def fetch_output(a, b):
            return (a * jnp.power(b, p - 2) % p) % p
    elif dataset == "permutation_group_dataset":
        perms = set(map(tuple, permutations(list(range(k)))))
        group_elements1, group_elements2 = perms, perms

        def fetch_output(a, b):
            return tuple([a[b[i]] for i in range(len(b))])
    else:
        raise NotImplementedError(dataset+" dataset not implemented yet.")

    return fetch_output, group_elements1, group_elements2


def get_batch_generator(dataset, frac_train, bs, split='train', p=97, k=5, split_seed=0):
    assert split in {'train', 'test'}
    fetch_output, group_elements1, group_elements2 = get_group_elements_and_output_fn(dataset, p, k)

    ordered_group_elements1 = jnp.array(list(group_elements1))
    ordered_group_elements2 = jnp.array(list(group_elements2))
    all_elements = list(group_elements1.union(group_elements2))
    idx2vocab = ['o', '='] + all_elements
    vocab2idx = {vocab: idx for idx, vocab in enumerate(idx2vocab)}
    # max_value = max(all_elements)
    # vocab2idx_array = jnp.zeros(max_value + 1, dtype=int)  # +1 because indexing starts at 0
    # # Now populate the array with actual indices
    # for i, element in enumerate(all_elements):
    #     vocab2idx_array = vocab2idx_array.at[element].set(i+2)
    # n_out = len(group_elements1.union(group_elements2))
    idxs = jax.random.permutation(jax.random.PRNGKey(split_seed), len(group_elements1) * len(group_elements2))
    if split == 'train':
        pairs = idxs[:int(len(idxs) * frac_train)]
    elif split == 'test':
        pairs = idxs[int(len(idxs) * frac_train):]
    train_cardinality = len(idxs[:int(len(idxs) * frac_train)])
    test_cardinality = len(idxs[int(len(idxs) * frac_train):])

    # Construct dataset and store in memory:
    pairs_a = jax.vmap(lambda idx: ordered_group_elements1[idx // len(group_elements2)])(pairs)
    pairs_b = jax.vmap(lambda idx: ordered_group_elements2[idx % len(group_elements2)])(pairs)
    pairs_c = jax.vmap(fetch_output)(pairs_a, pairs_b)
    pairs = jnp.stack([pairs_a, pairs_b, pairs_c], axis=-1)

    def apply_mapping(x):
        a_i, b_i, c_i = x
        a_i, b_i, c_i = int(a_i), int(b_i), int(c_i)
        mapped_indices = jnp.array([vocab2idx.get(a_i), vocab2idx['o'], vocab2idx.get(b_i), vocab2idx['='], vocab2idx.get(c_i)-2])
        return mapped_indices

    array_dataset = apply_mapping(pairs[0])
    for el in pairs[1:]:
        array_dataset = jnp.vstack((array_dataset, apply_mapping(el)))
    # _, array_dataset = jax.lax.scan(apply_mapping, None, pairs)

    # @jax.jit
    # def fetch_batch(rng_key):
    #     batch_idxs = jax.random.choice(rng_key, len(pairs), shape=(bs,), replace=False)
    #     batch_a = jax.vmap(lambda idx: ordered_group_elements1[idx // len(group_elements2)])(batch_idxs)
    #     batch_b = jax.vmap(lambda idx: ordered_group_elements2[idx % len(group_elements2)])(batch_idxs)
    #     batch_c = jax.vmap(fetch_output)(batch_a, batch_b)
    #     return (jax.vmap(lambda a, b: jnp.array([vocab2idx_array[a], vocab2idx['o'], vocab2idx_array[b], vocab2idx['=']]))(batch_a, batch_b),
    #             jax.vmap(lambda c: vocab2idx_array[c] - 2)(batch_c))

    @jax.jit
    def fetch_batch(rng_key):
        batch_ind = jax.random.randint(rng_key, shape=(bs,), minval=0, maxval=len(array_dataset))
        return array_dataset[batch_ind, :-1], array_dataset[batch_ind, -1]

    # return fetch_batch, len(idx2vocab), n_out, train_cardinality, test_cardinality
    return fetch_batch, train_cardinality, test_cardinality


@dataclass
class AbstractDataset:
    dataset: str
    frac_train: float
    p: int
    k: int
    # n_vocab: int = field(init=False)
    # n_out: int = field(init=False)
    train_cardinality: int = field(init=False)
    test_cardinality: int = field(init=False)

    def __post_init__(self):
        # _, self.n_vocab, self.n_out, self.train_cardinality, self.test_cardinality = get_batch_generator(self.dataset,
        #                                                                                      self.frac_train, 1,
        #                                                                                      split='train', p=self.p,
        #                                                                                      k=self.k)
        _, self.train_cardinality, self.test_cardinality = get_batch_generator(self.dataset,
                                                                               self.frac_train,
                                                                               1,
                                                                               split='train',
                                                                               p=self.p,
                                                                               k=self.k)


# class AbstractDataset(abc.ABC):
#     def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float):
#         self.frac_train = frac_train
#         self.group_elements1 = group_elements1
#         self.group_elements2 = group_elements2
#         self.ordered_group_elements1 = list(self.group_elements1)
#         self.ordered_group_elements2 = list(self.group_elements2)
#         self.idx2vocab = ['o', '='] + list(group_elements1.union(group_elements2))
#         self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
#         self.n_vocab = len(self.idx2vocab)
#         self.n_out = len(group_elements1.union(group_elements2))
#         idxs = list(range(len(self.group_elements1) * len(self.group_elements2)))
#         random.shuffle(idxs)
#         self.train_pairs, self.val_pairs = idxs[:int(len(idxs) * frac_train)], idxs[int(len(idxs) * frac_train):]
#         self.trainset_cardinality = len(self.train_pairs)
#
#     @abc.abstractmethod
#     def fetch_output(self, a, b):
#         pass
#
#     def encode(self, sequence):
#         return [self.vocab2idx[item] for item in sequence]
#
#     def decode(self, sequence):
#         return [self.idx2vocab[item] for item in sequence]
#
#     def form_equation(self, a, b, c):
#         return [a, 'o', b, '=', c]
#
#     def fetch_example(self, idx):
#         a = self.ordered_group_elements1[idx // len(self.group_elements2)]
#         b = self.ordered_group_elements2[idx % len(self.group_elements2)]
#         c = self.fetch_output(a, b)
#         equation = self.form_equation(a, b, c)
#         return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation
#
#     def fetch_train_example(self):
#         idx = random.choice(self.train_pairs)
#         return self.fetch_example(idx)
#
#     def fetch_val_example(self):
#         idx = random.choice(self.val_pairs)
#         return self.fetch_example(idx)


# class ModSumDataset(AbstractDataset):
#     def __init__(self, frac_train):
#         p = 97
#         super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
#         self.p = p
#
#     def fetch_output(self, a, b):
#         return (a + b) % self.p

class ModSumDataset(AbstractDataset):
    def __init__(self, frac_train, p, k):
        super(ModSumDataset, self).__init__("mod_sum_dataset", frac_train, p, k)


# class ModSubtractDataset(AbstractDataset):
#     def __init__(self, frac_train):
#         p = 97
#         super(ModSubtractDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
#         self.p = p
#
#     def fetch_output(self, a, b):
#         return (a - b) % self.p


class ModSubtractDataset(AbstractDataset):
    def __init__(self, frac_train, p, k):
        super(ModSubtractDataset, self).__init__("mod_subtract_dataset", frac_train, p, k)


# class ModDivisonDataset(AbstractDataset):
#     def __init__(self, frac_train):
#         p = 97
#         super(ModDivisonDataset, self).__init__(set(range(p)), set(range(1, p)), frac_train)
#         self.p = p
#
#     def fetch_output(self, a, b):
#         return (a * pow(b, self.p - 2, self.p)) % self.p

class ModDivisonDataset(AbstractDataset):
    def __init__(self, frac_train, p, k):
        super(ModDivisonDataset, self).__init__("mod_division_dataset", frac_train, p, k)


# class PermutationGroup(AbstractDataset):
#     def __init__(self, frac_train):
#         k = 5
#         perms = set(map(tuple, permutations(list(range(k)))))
#         super(PermutationGroup, self).__init__(perms, perms, frac_train)
#         self.k = k
#
#     def fetch_output(self, a, b):
#         return tuple([a[b[i]] for i in range(len(b))])


class PermutationGroup(AbstractDataset):
    def __init__(self, frac_train, p, k):
        super(PermutationGroup, self).__init__("permutation_group_dataset", frac_train, p, k)


# class GroupDataset(Iterable):
#     def __init__(self, dataset: AbstractDataset, split: str):
#         super(GroupDataset, self).__init__()
#         assert split in {'train', 'test'}
#         self.dataset = dataset
#         self.split = split
#         self.fetch_f = None
#         if self.split == 'train':
#             self.fetch_f = self.dataset.fetch_train_example
#         elif self.split == 'test':
#             self.fetch_f = self.dataset.fetch_val_example
#         else:
#             raise NotImplementedError
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         x, y, _ = self.fetch_f()
#         return jnp.array(x, dtype=int), jnp.array(y, dtype=int)


class BatchingIterator:
    def __init__(self, fetching_fn):
        self.fetching_fn = fetching_fn
        self.key = jax.random.PRNGKey(42)

    def __iter__(self):
        return self

    def __next__(self):
        self.key, rng_key = jax.random.split(self.key)

        return self.fetching_fn(rng_key)


def get_n_out(_, group_elements1, group_elements2):
    return len(group_elements1.union(group_elements2))


def get_vocab_size(_, group_elements1, group_elements2):
    return len(list(group_elements1.union(group_elements2))) + 2


all_ds = ["mod_division_dataset", "mod_subtract_dataset", "mod_sum_dataset", "permutation_group_dataset"]
n_out_mapping = {ds: get_n_out(*get_group_elements_and_output_fn(ds, p=97, k=5)) for ds in all_ds}
vocab_size_mapping = {ds: get_vocab_size(*get_group_elements_and_output_fn(ds, p=97, k=5)) for ds in all_ds}
# vocab_size_mapping = {
#     "mod_division_dataset": ModDivisonDataset(1.0, p=97, k=5).n_vocab,
#     "mod_subtract_dataset": ModSubtractDataset(1.0, p=97, k=5).n_vocab,
#     "mod_sum_dataset": ModSumDataset(1.0, p=97, k=5).n_vocab,
#     "permutation_group_dataset": PermutationGroup(1.0, p=97, k=5).n_vocab,
# }


def load_grok_ds(dataset: AbstractDataset, split: str, *, is_training: bool, batch_size: int,
                    other_bs: Optional[Iterable] = None,
                    subset: Optional[Sequence[int]] = None,
                    cardinality: bool = False, noisy_label: float = 0, permuted_img_ratio: float = 0,
                    gaussian_img_ratio: float = 0, augment_dataset: bool = False, normalize: bool = False,
                    reduced_ds_size: Optional[int] = None):
    assert subset is None, "subset must be None for grokking datasets"
    assert noisy_label == 0, "noisy_label must be 0 for grokking datasets"
    assert permuted_img_ratio == 0, "permuted_img_ratio must be 0 for grokking datasets"
    assert gaussian_img_ratio == 0, "gaussian_img_ratio must be 0 for grokking datasets"
    assert not augment_dataset, "grokking datasets do not support data augmentation"
    assert not normalize, "grokking datasets do not support data normalization"
    assert reduced_ds_size is None, "reduced_ds_size must be None for grokking datasets"

    if split == "train":
        ds_size = dataset.train_cardinality
    elif split == "test":
        ds_size = dataset.test_cardinality
    # dataset is a AbstractDataset object
    if other_bs:
        all_ds = [BatchingIterator(
            get_batch_generator(dataset.dataset, frac_train=dataset.frac_train, bs=batch_size, split=split, p=dataset.p,
                                k=dataset.k)[0])]

        for bs in other_bs:
            all_ds.append(BatchingIterator(
                get_batch_generator(dataset.dataset, frac_train=dataset.frac_train, bs=bs, split=split, p=dataset.p,
                                    k=dataset.k)[0]))

        if cardinality:
            return (ds_size, ) + tuple(all_ds)
        else:
            return tuple(all_ds)
    else:
        grok_iterator = BatchingIterator(
            get_batch_generator(dataset.dataset, frac_train=dataset.frac_train, bs=batch_size, split=split, p=dataset.p,
                                k=dataset.k)[0])
        if cardinality:
            return ds_size, grok_iterator
        else:
            return grok_iterator
