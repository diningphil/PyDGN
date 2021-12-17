import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import download_url, extract_zip, Data
from torch_geometric_temporal import DynamicGraphTemporalSignal
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

from pydgn.data.dataset import DatasetInterface


class TemporalDatasetInterface(DatasetInterface):
    name = None

    def get_mask(self):
        raise NotImplementedError("You should subclass DynamicDatasetInterface and implement this method")

    def __len__(self):
        raise NotImplementedError("You should subclass DynamicDatasetInterface and implement this method")

    def __getitem__(self, time_index):
        raise NotImplementedError("You should subclass DynamicDatasetInterface and implement this method")


class ChickenpoxDatasetInterface(TemporalDatasetInterface):
    def __init__(self, root, name, lags=4):
        self.root = root
        self.name = name
        self.lags = lags

        self.dataset = ChickenpoxDatasetLoader().get_dataset(lags=lags)

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = torch.ones(1,1)  #  time_steps x 1
        return mask

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

    def __getitem__(self, time_index):
        # TODO WARNING: __get_item__ is specific of Pytorch Geometric Temporal! This should be addressed in next versions
        # TODO by replacing it with __getitem__
        data = self.dataset.__get_item__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        return data

    def __len__(self):
        return len(self.dataset.features)  # see DynamicGraphTemporalSignal


class TUTemporalDatasetInterface(TemporalDatasetInterface):
    """
    TUDataset interface for dynamic graphs.
    It contains dynamic graphs for graph classification.
    Taken from:
    Oettershagen, Kriege, Morris, Mutzel (2020). Temporal Graph Kernels for Classifying Dissemination Processes.
    Proceedings of the 2020 SIAM International Conference on Data Mining, pp. 496–504.
    Available at http://arxiv.org/abs/1911.05496
    """

    def __init__(self, root, name):
        self.name = name
        self.root = root
        path = self._check_and_download()
        self.dataset = self._load_data(path)

    def _check_and_download(self) -> str:
        path = os.path.join(self.root, self.name)
        if not os.path.isdir(path):
            zip_file = download_url(f'https://www.chrsmrrs.com/graphkerneldatasets/{self.name}.zip', self.root)
            extract_zip(zip_file, self.root)
            os.unlink(zip_file)
        return path

    # Adapted from https://github.com/dtortorella/graph-esn/blob/main/src/graphesn/dataset.py
    def _load_data(self, path):
        data_list = []

        edges = np.loadtxt(os.path.join(path, f'{self.name}_A.txt'), delimiter=',', dtype=np.int64) - 1
        indicator = np.loadtxt(os.path.join(path, f'{self.name}_graph_indicator.txt'), dtype=np.int64) - 1
        timestamps = np.loadtxt(os.path.join(path, f'{self.name}_edge_attributes.txt'), dtype=np.int64)
        y = np.loadtxt(os.path.join(path, f'{self.name}_graph_labels.txt'), dtype=np.int64)

        df = pd.read_csv(os.path.join(path, f'{self.name}_node_labels.txt'), names=['t0', 'x0', 't1', 'x1'],
                         dtype={'t0': np.int64, 'x0': np.float32, 't1': 'Int64', 'x1': np.float32})

        t1, x1 = df.t1.to_numpy(dtype=np.int64, na_value=-1), df.x1.to_numpy(dtype=np.float32)
        T = max(t1.max(), timestamps.max()) + 1
        change_mask = (t1 >= 0)

        x = np.zeros((len(t1), T), dtype=np.float32)
        x[change_mask, t1[change_mask]] = x1[change_mask]
        x = np.cumsum(x, axis=1)

        pos = np.cumsum(np.bincount(indicator + 1))

        empty_adj = torch.empty((2, 0), dtype=torch.int64)
        for sample in np.unique(indicator):
            edge_slice = np.bitwise_and(edges[:, 0] >= pos[sample], edges[:, 0] < pos[sample + 1])
            edge_index = [empty_adj] * T
            for t in np.unique(timestamps[edge_slice]):
                time_edge_slice = np.bitwise_and(edge_slice, timestamps == t)
                edge_index[t] = torch.tensor(edges[time_edge_slice] - pos[sample]).t()
            unmasked = np.unique(
                np.concatenate([timestamps[edge_slice], t1[[sample]]]) if t1[sample] >= 0 else timestamps[edge_slice])
            mask = np.zeros((T, pos[sample + 1] - pos[sample]), dtype=bool)
            mask[unmasked, :] = True

            targets = [torch.tensor([y[sample]]) for _ in range(len(edge_index))]
            features = torch.tensor(x[pos[sample]:pos[sample + 1]].T).unsqueeze(-1)
            features = [x[t] for t in range(features.shape[0])]
            # We add the y field for stratification purposes when using pydgn standard Splitter
            data_list.append(DynamicGraphTemporalSignal(edge_indices=edge_index, edge_weights=[None]*len(edge_index), targets=targets, features=features))
        return data_list

    def __getitem__(self, idx):
        data = self.dataset[idx]
        setattr(data, 'mask', self.get_mask(data))
        return data

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = torch.zeros(len(data.features), 1)  #  time_steps x 1
        mask[-1, 0] = 1  # last time step prediction only
        return mask

    @property
    def dim_node_features(self):
        return self.dataset[0].features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        # binary graph classification
        return 1

    def __len__(self):
        return len(self.dataset)