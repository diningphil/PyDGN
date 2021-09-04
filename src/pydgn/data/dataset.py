import os
import os.path as osp
import shutil
import sys

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset, Planetoid, KarateClub

# Trying to suppress that "Outdated version" message caused by OBG
stderr_tmp = sys.stderr
null = open(os.devnull, 'w')
sys.stderr = null
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils.url import decide_download, download_url, extract_zip

sys.stderr = stderr_tmp


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


class ConcatFromListDataset(InMemoryDataset):
    """Create a dataset from a `torch_geometric.Data` list.
    Args:
        data_list (list): List of graphs.
    """

    def __init__(self, data_list):
        super(ConcatFromListDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class DatasetInterface:
    name = None

    @property
    def dim_node_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def dim_edge_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")


class TUDatasetInterface(TUDataset, DatasetInterface):

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, use_node_attr=False,
                 use_edge_attr=False, cleaned=False):
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        if 'alchemy_full' in self.name:
            return self.data.y.shape[1]
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class KarateClubDatasetInterface(KarateClub, DatasetInterface):

    def __init__(self, root, name, transform=None, pre_transform=None):
        super().__init__()
        self.root = root
        self.name = name
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.data.x = torch.ones(self.data.x.shape[0], 1)
        torch.save((self.data, self.slices), osp.join(self.processed_dir, 'data.pt'))

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def dim_node_features(self):
        return 1

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 2

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class LinkPredictionKarateClub(KarateClubDatasetInterface):

    @property
    def dim_target(self):
        return 1

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class PlanetoidDatasetInterface(Planetoid, DatasetInterface):

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


class LinkPredictionPlanetoid(PlanetoidDatasetInterface):

    @property
    def dim_target(self):
        return 1

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class OGBG(PygGraphPropPredDataset, DatasetInterface):
    def __init__(self, root, name, transform=None,
                 pre_transform=None, pre_filter=None, meta_dict=None):
        super().__init__('-'.join(name.split('_')), root, transform, pre_transform, meta_dict)  #
        self.name = name
        self.data.y = self.data.y.squeeze()

    @property
    def dim_node_features(self):
        return 1

    @property
    def dim_edge_features(self):
        if self.data.edge_attr is not None:
            return self.data.edge_attr.shape[1]
        else:
            return 0

    @property
    def dim_target(self):
        return 37

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            print(f'Removing {path}')
            os.unlink(path)
            print(f'Removing {self.root}')
            shutil.rmtree(self.root)
            print(f'Moving {osp.join(self.original_root, self.download_name)} to {self.root}')
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

    def process(self):
        super().process()
