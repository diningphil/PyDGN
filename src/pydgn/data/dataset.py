from typing import List

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data.dataset import Dataset


class ZipDataset(Dataset):
    r"""
    This Dataset takes `n` datasets and "zips" them. When asked for the `i`-th element, it returns the `i`-th element of
    all `n` datasets.

    Args:
        datasets (List[Dataset]): An iterable with PyTorch Datasets

    Precondition:
        The length of all datasets must be the same

    """
    def __init__(self, *datasets: List[Dataset]):
        self.datasets = datasets
        assert len(set(len(d) for d in self.datasets)) == 1

    def __getitem__(self, index: int) -> List[object]:
        r"""
        Returns the `i`-th element of all datasets in a list

        Args:
            index (int): the index of the data point to retrieve from each dataset

        Returns:
            A list containing one element for each dataset in the ZipDataset
        """
        return [d[index] for d in self.datasets]

    def __len__(self) -> int:
        r"""
        Returns the length of the datasets (all of them have the same length)

        Returns:
            The unique number of samples of all datasets
        """
        return len(self.datasets[0])


class ConcatFromListDataset(InMemoryDataset):
    r"""
    Create a dataset from a list of :class:`torch_geometric.data.Data` objects. Inherits from
    :class:`torch_geometric.data.InMemoryDataset`

    Args:
        data_list (list): List of graphs.
    """
    def __init__(self, data_list: List[Data]):
        super(ConcatFromListDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class DatasetInterface:
    r"""
    Class that defines a number of properties essential to all datasets implementations inside PyDGN. These properties
    are used by the training engine and forwarded to the model to be trained. For some datasets, e.g.,
    :class:`torch_geometric.datasets.TUDataset`, implementing this interface is straightforward.
    """
    name = None

    @property
    def dim_node_features(self):
        r"""
        Specifies the number of node features (after pre-processing, but in the end it depends on the model that is
        implemented).
        """
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def dim_edge_features(self):
        r"""
        Specifies the number of edge features (after pre-processing, but in the end it depends on the model that is
        implemented).
        """
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def dim_target(self):
        r"""
        Specifies the dimension of each target vector.
        """
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")


class TUDatasetInterface(TUDataset, DatasetInterface):
    r"""
    Class that wraps the :class:`torch_geometric.datasets.TUDataset` class to provide aliases of some fields.
    """
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, use_node_attr=False,
                 use_edge_attr=False, cleaned=False):
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)

    @property
    def dim_node_features(self):
        return self.num_node_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return self.num_classes

    def download(self):
        super().download()

    def process(self):
        super().process()


class PlanetoidDatasetInterface(Planetoid, DatasetInterface):
    r"""
    Class that wraps the :class:`torch_geometric.datasets.Planetoid` class to provide aliases of some fields.
    """

    @property
    def dim_node_features(self):
        return self.num_node_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return self.num_classes

    def download(self):
        super().download()

    def process(self):
        super().process()