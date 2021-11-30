import os
import os.path as osp
import shutil
import sys

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset, Planetoid, KarateClub

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader


class DynamicDatasetInterface(DatasetInterface):
    name = None

    def __len__(self):
        raise NotImplementedError("You should subclass DynamicDatasetInterface and implement this method")

    def __getitem__(self):
        raise NotImplementedError("You should subclass DynamicDatasetInterface and implement this method")


class ChickenpoxDatasetInterface(ChickenpoxDatasetLoader, DynamicDatasetInterface):
    def __init__(self, root, name, lags=4):
        self.root = root
        self.name = name
        self.lags = lags

        self.dataset = ChickenpoxDatasetLoader().get_dataset(lags=lags)

    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        # node classification: each time step is a tuple
        return 1

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        return self.dataset[time_index]
