import os
import shutil
from typing import List, Union, Tuple, Optional, Callable

import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import TUDataset, Planetoid


class ZipDataset(torch.utils.data.Dataset):
    r"""
    This Dataset takes `n` datasets and "zips" them. When asked for the `i`-th element, it returns the `i`-th element of
    all `n` datasets.

    Args:
        datasets (List[torch.utils.data.Dataset]): An iterable with PyTorch Datasets

    Precondition:
        The length of all datasets must be the same

    """
    def __init__(self, datasets: List[torch.utils.data.Dataset]):
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

    def download(self):
        pass

    def process(self):
        pass

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []



class DatasetInterface(torch_geometric.data.dataset.Dataset):
    r"""
    Class that defines a number of properties essential to all datasets implementations inside PyDGN. These properties
    are used by the training engine and forwarded to the model to be trained. For some datasets, e.g.,
    :class:`torch_geometric.datasets.TUDataset`, implementing this interface is straightforward.

    Args:
        root (str): root folder where to store the dataset
        name (str): name of the dataset
        transform (Optional[Callable]): transformations to apply to each ``Data`` object at run time
        pre_transform (Optional[Callable]): transformations to apply to each ``Data`` object at dataset creation time
        pre_filter (Optional[Callable]): sample filtering to apply to each ``Data`` object at dataset creation time
    """
    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable]=None,
                 pre_transform: Optional[Callable]=None,
                 pre_filter: Optional[Callable]=None):

        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    def download(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    def process(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    def len(self) -> int:
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    def get(self, idx: int) -> Data:
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

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

    def __len__(self) -> int:
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")


class TUDatasetInterface(TUDataset):
    r"""
    Class that wraps the :class:`torch_geometric.datasets.TUDataset` class to provide aliases of some fields.
    It implements the interface ``DatasetInterface`` but does not extend directly to avoid clashes of ``__init__`` methods
    """
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.name = name
        # Do not call DatasetInterface __init__ method in this case, because otherwise it will break
        super().__init__(root=root, name=name,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
                         **kwargs)

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

    def __len__(self) -> int:
        return len(self.data)


class PlanetoidDatasetInterface(Planetoid):
    r"""
    Class that wraps the :class:`torch_geometric.datasets.Planetoid` class to provide aliases of some fields.
    It implements the interface ``DatasetInterface`` but does not extend directly to avoid clashes of ``__init__`` methods
    """
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.name = name
        # Do not call DatasetInterface __init__ method in this case, because otherwise it will break
        super().__init__(root=root, name=name,
                         transform=transform, pre_transform=pre_transform,
                         **kwargs)
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

    def __len__(self) -> int:
        if isinstance(self.data, Data):
            return 1
        return len(self.data)


class OGBGDatasetInterface(PygGraphPropPredDataset):
    r"""
    Class that wraps the :class:`ogb.graphproppred.PygGraphPropPredDataset` class to provide aliases of some fields.
    It implements the interface ``DatasetInterface`` but does not extend directly to avoid clashes of ``__init__`` methods
    """
    def __init__(self, root, name, transform=None,
                 pre_transform=None, pre_filter=None, meta_dict=None):
        super().__init__('-'.join(name.split('_')), root, transform, pre_transform, meta_dict)  #
        self.name = name
        self.data.y = self.data.y.squeeze()

    def get_idx_split(self, split_type: str = None) -> dict:
        return self.dataset.get_idx_split(split_type=split_type)

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
            print(f'Moving {os.path.join(self.original_root, self.download_name)} to {self.root}')
            shutil.move(os.path.join(self.original_root, self.download_name), self.root)

    def process(self):
        super().process()

    def __len__(self) -> int:
        return len(self.data)
