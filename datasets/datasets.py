import torch
import os
import json
import shutil
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_tu_data


class ZipDataset(torch.utils.data.Dataset):
    """
    This Dataset takes n datasets and "zips" them. When asked for the i-th element, it returns the i-th element of
    all n datasets. The lenght of all datasets must be the same.
    """
    def __init__(self, *datasets):
        """
        Stores all datasets in an internal variable.
        :param datasets: An iterable with PyTorch Datasets
        """
        self.datasets = datasets

        assert len(set(len(d) for d in self.datasets)) == 1

    def __getitem__(self, index):
        """
        Returns the i-th element of all datasets
        :param index: the index of the data point to retrieve
        :return: a list containing one element for each dataset in the ZipDataset
        """
        return [d[index] for d in self.datasets]

    def __len__(self):
        return len(self.datasets[0])


class DatasetInterface:

    @property
    def dim_node_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def dim_edge_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")


class TUDatasetInterface(TUDataset, DatasetInterface):

    # Do not implement a dummy init function that calls super().__init__, ow it breaks

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()
