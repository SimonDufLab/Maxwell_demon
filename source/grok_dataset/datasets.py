"""Copied from https://github.com/Sea-Snell/grokking/blob/main/grokk_replica/datasets.py, with addition for
compatibilty"""

import abc
import random
import jax.numpy as jnp
from collections.abc import Iterable
from itertools import permutations
from typing import Set, Optional, Sequence


class AbstractDataset(abc.ABC):
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float):
        self.frac_train = frac_train
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ['o', '='] + list(group_elements1.union(group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(group_elements1.union(group_elements2))
        idxs = list(range(len(self.group_elements1) * len(self.group_elements2)))
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = idxs[:int(len(idxs) * frac_train)], idxs[int(len(idxs) * frac_train):]
        self.trainset_cardinality = len(self.train_pairs)

    @abc.abstractmethod
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]

    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]

    def form_equation(self, a, b, c):
        return [a, 'o', b, '=', c]

    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)


class ModSumDataset(AbstractDataset):
    def __init__(self, frac_train):
        p=97
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a, b):
        return (a + b) % self.p


class ModSubtractDataset(AbstractDataset):
    def __init__(self, frac_train):
        p=97
        super(ModSubtractDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a, b):
        return (a - b) % self.p


class ModDivisonDataset(AbstractDataset):
    def __init__(self, frac_train):
        p=97
        super(ModDivisonDataset, self).__init__(set(range(p)), set(range(1, p)), frac_train)
        self.p = p

    def fetch_output(self, a, b):
        return (a * pow(b, self.p - 2, self.p)) % self.p


class PermutationGroup(AbstractDataset):
    def __init__(self, frac_train):
        k=5
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train)
        self.k = k

    def fetch_output(self, a, b):
        return tuple([a[b[i]] for i in range(len(b))])


class GroupDataset(Iterable):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'test'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'test':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return jnp.array(x, dtype=int), jnp.array(y, dtype=int)


class BatchingIterator:
    def __init__(self, dataset_iterator, batch_size):
        self.dataset_iterator = dataset_iterator
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        batch_x = []
        batch_y = []

        for _ in range(self.batch_size):
            x, y = next(self.dataset_iterator)
            batch_x.append(x)
            batch_y.append(y)

        return jnp.stack(batch_x), jnp.stack(batch_y)


vocab_size_mapping = {
    "mod_division_dataset": ModDivisonDataset(1.0).n_vocab,
    "mod_subtract_dataset": ModSubtractDataset(1.0).n_vocab,
    "mod_sum_dataset": ModSumDataset(1.0).n_vocab,
    "permutation_group_dataset": PermutationGroup(1.0).n_vocab,
}


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

    # dataset is a AbstractDataset object
    if other_bs:
        ds_size = dataset.trainset_cardinality
        all_ds = [BatchingIterator(GroupDataset(dataset, split=split), batch_size=batch_size)]

        for bs in other_bs:
            all_ds.append(BatchingIterator(GroupDataset(dataset, split=split), batch_size=bs))

        if cardinality:
            return (ds_size, ) + tuple(all_ds)
        else:
            return tuple(all_ds)
    else:
        ds_size = dataset.trainset_cardinality
        grok_iterator = BatchingIterator(GroupDataset(dataset, split=split), batch_size=batch_size)
        if cardinality:
            return ds_size, grok_iterator
        else:
            return grok_iterator
