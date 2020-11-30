import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import hashlib
import os, warnings, requests
import random as rd
from PIL import Image
import json
from itertools import repeat
from multiprocessing import Pool

import pandas as pd
import os.path as osp
import random

import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw

import networkx as nx

from rdkit.Chem import QED, MolFromSmiles
from rdkit.Chem.AllChem import GetSymmSSSR
from torch_scatter import scatter_add

import torch_geometric
from torch_geometric.data import Dataset, Data, InMemoryDataset

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


class CustomSplitOGB(PygGraphPropPredDataset):
    def __init__(self, *args, split=None, **kwargs):
        self.split = split.lower() if split is not None else None
        super().__init__(*args, **kwargs)

    def get_idx_split(self):
        if self.split is None or self.split == 'scaffold':  # ogb default
            return super().get_idx_split()
        elif self.split == "random":
            self.make_idx_split()
            idxs_path = osp.join(self.root, "split", "random", 'idxs.txt')
            with open(idxs_path, 'r') as f:
                idxs = json.load(f)
            return {k: torch.tensor(v).long() for k, v in idxs.items()}
        else:
            raise NotImplemented('invalid split {}'.format(self.split))

    def make_idx_split(self):
        idxs_path = osp.join(self.root, "split", "random", "idxs.txt")
        os.makedirs(osp.dirname(idxs_path), exist_ok=True)
        if not osp.exists(idxs_path):
            path = osp.join(self.root, "split", "scaffold")
            train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header=None).values.T[0]
            valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
            test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header=None).values.T[0]
            concat_dataset = [i for i in range(len(train_idx) + len(valid_idx) + len(test_idx))]
            random.shuffle(concat_dataset)
            train_idx = concat_dataset[:len(train_idx)]
            valid_idx = concat_dataset[len(train_idx):len(train_idx) + len(valid_idx)]
            test_idx = concat_dataset[len(train_idx) + len(valid_idx):]
            result = {"train": train_idx,
                      "valid": valid_idx,
                      "test": test_idx}
            with open(idxs_path, 'w') as save_file:
                json.dump(result, save_file, indent=4)

    def download(self):
        return super().download()

    def process(self):
        return super().process()


def parse_combined_name(name):
    names = []
    splits = []
    for ds in name.split('+'):
        sp = ds.split('_')
        names.append(sp[0])
        splits.append(sp[1].lower() if len(sp) == 2 else None)
    return names, splits


class CombinedOGBEvaluator:
    def __init__(self, name=""):
        self.names, self.splits = parse_combined_name(name)
        self.evaluators = [Evaluator(name='ogbg-' + n)
                           for n in self.names]
        self.base_task_idx = [0] + [ev.num_tasks for ev in self.evaluators]
        self.base_task_idx = np.cumsum(self.base_task_idx)

    def eval(self, input_dict):
        result = {}
        for i, (name, split) in enumerate(zip(self.names, self.splits)):
            ds_idx = (input_dict['dataset_idx'] == i)
            task_slice = slice(self.base_task_idx[i], self.base_task_idx[i + 1])
            ds_input = {
                'y_true': input_dict['y_true'][ds_idx][:, task_slice],
                'y_pred': input_dict['y_pred'][ds_idx][:, task_slice],
            }
            dataset_result = self.evaluators[i].eval(ds_input)
            for k, v in dataset_result.items():
                result[name + '_' + (split or 'default') + '-split' + '_' + k] = v
        return result

    @property
    def eval_metric(self):
        return self.names[0] + '_' + (self.splits[0] or 'default') + '-split' + '_' + self.evaluators[0].eval_metric


class CombinedOGBDataset(Dataset):
    def __init__(self, name="", root=".", transform=None, pre_transform=None):
        # super is not called ON PURPOSE.
        # -> I only want to inherit __getitem__ inner machinery
        self.names, self.splits = parse_combined_name(name)
        self.root = root
        self.datasets = [CustomSplitOGB(name='ogbg-' + n, split=s, root=root,
                                        transform=transform, pre_transform=pre_transform)
                         for n, s in zip(self.names, self.splits)]
        self.transform = None  # necessary for inheritance of __getitem__ in Dataset
        self.__indices__ = None
        self.base_idx = [0] + [len(ds) for ds in self.datasets]
        self.base_idx = np.cumsum(self.base_idx)
        self.base_task_idx = [0] + [ds.num_tasks for ds in self.datasets]
        self.base_task_idx = np.cumsum(self.base_task_idx)
        self.num_tasks = np.sum([ds.num_tasks for ds in self.datasets])
        self.real_len = len(self)

    def get(self, idx):
        idx = idx % self.real_len
        for i, base_idx in enumerate(self.base_idx[1:]):
            if base_idx > idx:
                break
        data = self.datasets[i][int(idx - self.base_idx[i])]

        # the cumulated y is a vector of nans, except in the slice corresponding to the ds
        y = float('nan') * torch.empty(self.num_tasks)
        y[self.base_task_idx[i]:self.base_task_idx[i + 1]] = data.y
        data.y = y.unsqueeze(0)
        data.dataset_idx = torch.tensor([i])
        return data

    def get_sub_dataset(self, name):
        try:
            idx = self.names.index(name)
        except ValueError:
            raise ValueError("name not in allowed datasets: {}".format(self.names))
        indices = torch.tensor(self.indices())
        in_dataset = (self.base_idx[idx] <= indices) & (indices < self.base_idx[idx + 1])
        restricted_ds = self[in_dataset]

        def slice_y(data):
            data.y = data.y[..., self.base_task_idx[idx]: self.base_task_idx[idx + 1]]
            return data

        restricted_ds.transform = slice_y
        return restricted_ds

    def get_idx_split(self):
        splits = [ds.get_idx_split() for ds in self.datasets]
        idx_split = {}
        for k in splits[0]:
            idx_split[k] = torch.cat([
                sp[k] + base_idx for base_idx, sp in zip(self.base_idx, splits)
            ])
        return idx_split

    def __len__(self):
        if self.__indices__ is not None:
            return len(self.__indices__)
        return self.base_idx[-1]

    def __str__(self):
        return "Combined OGB datasets: {}".format(self.names)
