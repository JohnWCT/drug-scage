import pickle
from pathlib import Path

from torch.utils.data import Subset

__all__ = [
    'ScaffoldSplitter',
    'RandomScaffoldSplitter',
    'CyclicScaffoldSplitter',
]

from utils.userconfig_util import get_split_dir


def create_splitter(split_type, seed, split_pkl_path=None, allow_empty_test=False):
    """Return a splitter according to the ``split_type``"""
    if split_type == 'scaffold':
        splitter = ScaffoldSplitter(split_pkl_path, allow_empty_test)
    elif split_type == 'random_scaffold':
        splitter = RandomScaffoldSplitter(seed, split_pkl_path, allow_empty_test)
    elif split_type == 'cyclic_scaffold':
        splitter = CyclicScaffoldSplitter(seed, split_pkl_path, allow_empty_test)
    else:
        raise ValueError('%s not supported' % split_type)
    return splitter


class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test
    subsets.
    """

    def __init__(self):
        super(Splitter, self).__init__()


def _load_split_idx(split_path):
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f'Split file not found: {split_path}')
    with open(split_path, 'rb') as f:
        return pickle.load(f)


def _subset_triplet(dataset, split_idx, allow_empty_test=False):
    train_dataset = Subset(dataset, split_idx['train_idx'])
    valid_dataset = Subset(dataset, split_idx['valid_idx'])
    if 'test_idx' not in split_idx and not allow_empty_test:
        raise KeyError('Missing key "test_idx" in split file.')
    test_idx = split_idx.get('test_idx', [])
    if not test_idx and allow_empty_test:
        test_dataset = None
    else:
        test_dataset = Subset(dataset, test_idx)
    return train_dataset, valid_dataset, test_dataset


class ScaffoldSplitter(Splitter):

    def __init__(self, split_pkl_path=None, allow_empty_test=False):
        super(ScaffoldSplitter, self).__init__()
        self.split_pkl_path = split_pkl_path
        self.allow_empty_test = allow_empty_test

    def split(self, dataset, task_name):
        split_path = self.split_pkl_path or (get_split_dir() / f'scaffold/{task_name}.pkl')
        split_idx = _load_split_idx(split_path)
        return _subset_triplet(dataset, split_idx, self.allow_empty_test)


class RandomScaffoldSplitter(Splitter):

    def __init__(self, seed, split_pkl_path=None, allow_empty_test=False):
        super(RandomScaffoldSplitter, self).__init__()
        self.seed = seed
        self.split_pkl_path = split_pkl_path
        self.allow_empty_test = allow_empty_test

    def split(self, dataset, task_name):
        seed_ = self.seed
        split_path = self.split_pkl_path or (get_split_dir() / f'random_scaffold/{task_name}_{seed_}.pkl')
        split_idx = _load_split_idx(split_path)
        return _subset_triplet(dataset, split_idx, self.allow_empty_test)


class CyclicScaffoldSplitter(Splitter):

    def __init__(self, seed, split_pkl_path=None, allow_empty_test=False):
        super(CyclicScaffoldSplitter, self).__init__()
        self.seed = seed
        self.split_pkl_path = split_pkl_path
        self.allow_empty_test = allow_empty_test

    def split(self, dataset, task_name):
        seed_ = self.seed
        split_path = self.split_pkl_path or (
            get_split_dir() / f'cyclic_scaffold/{task_name}_cyclic_scaffold_seed{seed_}_v0.10_t0.10.pkl'
        )
        split_idx = _load_split_idx(split_path)
        return _subset_triplet(dataset, split_idx, self.allow_empty_test)
