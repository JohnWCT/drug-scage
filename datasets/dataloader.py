import os
import pickle
import random

import lmdb

from torch.utils.data import Dataset, Subset

class PretrainDataset(Dataset):
    def __init__(self, root):
        self.root = root

        lmdb_file = self.root
        self.env = lmdb.open(
            lmdb_file,
            subdir=False,
            readonly=True, lock=False,
            readahead=False, meminit=False
        )
        self.txn = self.env.begin()
        self.is_raw = False
        self.length = int(self.txn.get(b'__len__').decode())


        self.safe_mol_indices = []
        self.safe_mol_num = 100
        self.safe_mol_indices_init_flag = False

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.txn.get(str(idx).encode())
        data = pickle.loads(data)
        if check_if_atom_num_bigger_than(data['atomic_num'], 120):
            if not self.safe_mol_indices_init_flag:
                while len(self.safe_mol_indices) < self.safe_mol_num:
                    idx += 1
                    data = self.txn.get(str(idx).encode())
                    data = pickle.loads(data)
                    if not check_if_atom_num_bigger_than(data['atomic_num'], 120):
                        self.safe_mol_indices.append(idx)
                self.safe_mol_indices_init_flag = True
            sample_idx = random.choice(self.safe_mol_indices)
            return self.__getitem__(sample_idx)
        return data


def check_if_atom_num_bigger_than(atomic_num_numpy, num):
    # print(atomic_num_numpy.shape)
    return atomic_num_numpy.shape[0] > num


class FinetuneDataset(Dataset):
    def __init__(self, root, task_name):
        # torch.multiprocessing.set_start_method('spawn')
        self.root = root
        self.task_name = task_name
        self.data = pickle.load(open(os.path.join(self.root, f'{self.task_name}.pkl'), "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CliffDataset(Dataset):
    def __init__(self, root, task_name, mode):
        # torch.multiprocessing.set_start_method('spawn')
        self.root = root
        self.task_name = task_name
        self.data = pickle.load(open(os.path.join(self.root, f'{self.task_name}_{mode}.pkl'), "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

