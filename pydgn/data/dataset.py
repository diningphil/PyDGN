import os
import shutil
from pathlib import Path
from random import shuffle
from typing import List, Union, Tuple, Optional, Callable

import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric_temporal import ChickenpoxDatasetLoader


class ZipDataset(torch.utils.data.Dataset):
    r"""
    This Dataset takes `n` datasets and "zips" them. When asked for the `i`-th
    element, it returns the `i`-th element of all `n` datasets.

    Args:
        datasets (List[torch.utils.data.Dataset]): An iterable with
            PyTorch Datasets

    Precondition:
        The length of all datasets must be the same

    """

    def __init__(self, datasets: List[torch.utils.data.Dataset], **kwargs):
        self.datasets = datasets
        assert len(set(len(d) for d in self.datasets)) == 1

    def __getitem__(self, index: int) -> List[object]:
        r"""
        Returns the `i`-th element of all datasets in a list

        Args:
            index (int): the index of the data point to retrieve from each
                dataset

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
    Create a dataset from a list of :class:`torch_geometric.data.Data` objects.
    Inherits from :class:`torch_geometric.data.InMemoryDataset`

    Args:
        data_list (list): List of graphs.
    """

    def __init__(self, data_list: List[Data], **kwargs):
        super(ConcatFromListDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)

    def download(self):
        r"""
        Does nothing, the data list is already provided
        """
        pass

    def process(self):
        r"""
        Does nothing, the data list is already provided
        """
        pass

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        Does nothing, the data list is already provided
        """
        return []

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        Does nothing, the data list is already provided
        """
        return []


class DatasetInterface(torch_geometric.data.dataset.Dataset):
    r"""
    Class that defines a number of properties essential to all datasets
    implementations inside PyDGN. These properties are used by the training
    engine and forwarded to the model to be trained. For some datasets, e.g.,
    :class:`torch_geometric.datasets.TUDataset`, implementing this interface
    is straightforward.

    Args:
        root (str): root folder where to store the dataset
        name (str): name of the dataset
        transform (Optional[Callable]): transformations to apply to each
            ``Data`` object at run time
        pre_transform (Optional[Callable]): transformations to apply to each
            ``Data`` object at dataset creation time
        pre_filter (Optional[Callable]): sample filtering to apply to each
            ``Data`` object at dataset creation time
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    def download(self):
        r"""
        Downloads the dataset to the :obj:`self.raw_dir` folder.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    def process(self):
        r"""
        Processes the dataset to the :obj:`self.processed_dir` folder.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    def get(self, idx: int) -> Data:
        r"""
        Gets the data object at index :obj:`idx`.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )

    def len(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self)

    def __len__(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        raise NotImplementedError(
            "You should subclass DatasetInterface and implement this method"
        )


class TemporalDatasetInterface(DatasetInterface):
    """
    Extension of DatasetInterface to the temporal scenario.
    """

    def get_mask(self, data: Union[Batch, Data]) -> torch.Tensor:
        """
        Computes the mask of time steps for which we need to make a prediction.

        Args:
            data: the data object

        Returns:
            A tensor indicating the time-steps at which we expect predictions
        """
        raise NotImplementedError(
            "You should subclass TemporalDatasetInterface and implement this "
            "method"
        )

    def get(self, idx: int) -> Data:
        """
        Gets element ``idx`` from object self.dataset

        Args:
            idx (int): the sample index

        Returns:

        """
        data = self.dataset[idx]
        data.time_prediction_mask = self.get_mask(data)
        return data


class IterableDatasetInterface(torch.utils.data.IterableDataset):
    r"""
    Class that implements the Iterable-style dataset, including multi-process
    data loading
    (https://pytorch.org/docs/stable/data.html#iterable-style-datasets).
    Useful when the dataset is too big and split in chunks of files to be
    stored on disk. Each chunk can hold a single sample or a set of samples,
    and there is the chance to shuffle sample-wise or chunk-wise.
    To get a subset of this dataset, just provide an argument `url_indices`
    specifying which chunks you want to use. Must be combined with an
    appropriate :class:`pydgn.data.provider.IterableDataProvider`.

    NOTE 1: We assume the splitter will split the dataset with respect to to
    the number of files stored on disk, so be sure that the length of your
    dataset reflects that number. Then, examples will be provided sequentially,
    so if each file holds more than one sample, we will still be able to create
    a batch of samples from one or multiple files.

    NOTE 2: NEVER override the __len__() method, as it varies dynamically with
    the ``url_indices`` argument.

    Args:
        root (str): root folder where to store the dataset
        name (str): name of the dataset
        transform (Optional[Callable]): transformations to apply to each
            ``Data`` object at run time
        pre_transform (Optional[Callable]): transformations to apply to each
            ``Data`` object at dataset creation time
        url_indices (Optional[List]): list of indices used to extract
            a portion of the dataset
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        url_indices: Optional[List] = None,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.name = name

        # This is needed to compute the a subset of the entire dataset
        if url_indices is None:
            processed_file_names = self.processed_file_names
        else:
            processed_file_names = [
                self.processed_file_names[i] for i in url_indices
            ]

        # This is needed to compute the length of a subset of the entire
        # dataset
        self.urls = processed_file_names
        self.shuffled_urls = self.urls

        self.start_index = 0
        self.end_index = len(self.shuffled_urls)

        # This information allows us to shuffle between urls and sub-patches
        # inside each url
        self._shuffle_urls = False
        self._shuffle_subpatches = False

        self.pre_transform = pre_transform
        self.transform = transform

        for p in self.raw_file_names:
            if not os.path.exists(os.path.join(self.raw_dir, p)):
                self.download()
                break

        if not os.path.exists(self.processed_dir):
            os.makedirs(os.path.join(self.processed_dir))

        for p in processed_file_names:
            if not os.path.exists(os.path.join(self.processed_dir, p)):
                self.process()
                break

    def shuffle_urls_elements(self, value: bool):
        r"""
        Shuffles elements contained in each file (associated with an url).
        Use this method when a single file stores multiple samples and you want
        to provide them in shuffled order.
        IMPORTANT: in this case we assume that each file contains a
        list of Data objects!

        Args:
            value (bool): whether or not to shuffle urls
        """
        self._shuffle_subpatches = value

    def shuffle_urls(self, value: bool):
        r"""
        Shuffles urls associated to individual files stored on disk

        Args:
            value (bool): whether or not to shuffle urls
        """
        self._shuffle_urls = value

        # Needed for multiple dataloader workers
        if self._shuffle_urls:
            shuffle(self.shuffled_urls)

    def splice(self, start: int, end: int):
        r"""
        Use this method to assign portions of the dataset to load to different
        workers, otherwise they will load the same samples.

        Args:
            start (int): the index where to start
            end (int): the index where to stop
        """
        self.start_index = start
        self.end_index = end

    def __iter__(self):
        r"""
        Generator that returns individual Data objects. If each files contains
        a list of data objects, these can be shuffled using the
        method :func:`shuffle_urls_elements`.

        Returns:
            a Data object with the next element to process
        """
        end_index = (
            self.end_index
            if self.end_index <= len(self.shuffled_urls)
            else len(self.shuffled_urls)
        )
        for url in self.shuffled_urls[self.start_index : end_index]:
            sample = torch.load(os.path.join(self.processed_dir, url))
            if isinstance(sample, list):
                if self._shuffle_subpatches:
                    shuffle(sample)

                data = [
                    self.transform(d) if self.transform is not None else d
                    for d in sample
                ]
            else:
                data = [
                    self.transform(sample)
                    if self.transform is not None
                    else sample
                ]

            for i in range(len(data)):
                yield data[i]

    @property
    def raw_file_names(self) -> List[Union[str, Path]]:
        r"""
        The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    @property
    def raw_dir(self) -> Union[str, Path]:
        r"""
        The path where the raw data should be downloaded
        :return: a string
        """
        return os.path.join(self.root, self.name, "raw")

    @property
    def raw_paths(self) -> List[Union[str, Path]]:
        r"""
        The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The list of file names that must be present in order to skip
        downloading.
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    @property
    def processed_dir(self) -> Union[str, Path]:
        r"""
        The folder where to store processed data files.
        """
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_paths(self) -> List[Union[str, Path]]:
        r"""
        The absolute filepaths that must be present in order to skip
        processing.
        """
        files = self.processed_file_names
        return [os.path.join(self.processed_dir, f) for f in files]

    def download(self):
        r"""
        Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    def process(self):
        r"""
        Processes the dataset to the :obj:`self.processed_dir` folder.
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    def get(self, idx: int) -> Data:
        r"""
        Gets the data object at index :obj:`idx`.
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing,
        but in the end it depends on the model that is implemented).
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing,
        but in the end it depends on the model that is implemented).
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        raise NotImplementedError(
            "You should subclass IterableDatasetInterface "
            "and implement this method"
        )

    def __len__(self):
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply
        with PyG interface
        """
        # It's important it stays dynamic, because
        # self.urls depends on url_indices
        return len(self.urls)


class TUDatasetInterface(TUDataset):
    r"""
    Class that wraps the :class:`torch_geometric.datasets.TUDataset` class to
    provide aliases of some fields. It implements the interface
    ``DatasetInterface`` but does not extend directly to avoid
    clashes of ``__init__`` methods
    """

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        self.name = name
        # Do not call DatasetInterface __init__ method in this case, because
        # otherwise it will break
        super().__init__(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.num_node_features

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.num_edge_features

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return self.num_classes

    def download(self):
        r"""
        Downloads the TUDataset dataset to the :obj:`self.raw_dir` folder."""
        super().download()

    def process(self):
        r"""
        Processes the TUDataset dataset to the :obj:`self.processed_dir` folder
        """
        super().process()


class PlanetoidDatasetInterface(Planetoid):
    r"""
    Class that wraps the :class:`torch_geometric.datasets.Planetoid` class to
    provide aliases of some fields. It implements the interface
    ``DatasetInterface`` but does not extend directly to avoid clashes
    of ``__init__`` methods
    """

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        self.name = name
        # Do not call DatasetInterface __init__ method in this case, because
        # otherwise it will break
        super().__init__(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
        )

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.num_node_features

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.num_edge_features

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return self.num_classes

    def download(self):
        r"""
        Downloads the Planetoid dataset to the :obj:`self.raw_dir` folder."""
        super().download()

    def process(self):
        r"""
        Processes the Planetoid dataset to the :obj:`self.processed_dir` folder
        """
        super().process()

    def __len__(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        if isinstance(self.data, Data):
            return 1
        return len(self.data)


class OGBGDatasetInterface(PygGraphPropPredDataset):
    r"""
    Class that wraps the :class:`ogb.graphproppred.PygGraphPropPredDataset`
    class to provide aliases of some fields. It implements the interface
    ``DatasetInterface`` but does not extend directly to avoid clashes
    of ``__init__`` methods
    """

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        meta_dict=None,
        **kwargs,
    ):
        super().__init__(
            "-".join(name.split("_")),
            root,
            transform,
            pre_transform,
            meta_dict,
        )
        self.name = name
        self.data.y = self.data.y.squeeze()

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.data.x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        if self.data.edge_attr is not None:
            return self.data.edge_attr.shape[1]
        else:
            return 0

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return self.data.y.shape[1]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The list of file names that must be present in order to skip
        downloading.
        """
        return ["data.pt"]

    def download(self):
        r"""
        Downloads the OGB dataset to the :obj:`self.raw_dir` folder.
        """
        url = self.meta_info["url"]
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            print(f"Removing {path}")
            os.unlink(path)
            print(f"Removing {self.root}")
            shutil.rmtree(self.root)
            print(
                f"Moving "
                f"{os.path.join(self.original_root, self.download_name)} "
                f"to {self.root}"
            )
            shutil.move(
                os.path.join(self.original_root, self.download_name), self.root
            )

    def process(self):
        r"""
        Processes the OGB dataset to the :obj:`self.processed_dir` folder.
        """
        super().process()

    def __len__(self) -> int:
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return self.data.y.shape[0]


class ToyIterableDataset(IterableDatasetInterface):
    r"""
    Class that implements the Iterable-style dataset, including multi-process
    data loading
    (https://pytorch.org/docs/stable/data.html#iterable-style-datasets).
    Useful when the dataset is too big and split in chunks of files to be
    stored on disk. Each chunk can hold a single sample or a set of samples,
    and there is the chance to shuffle sample-wise or chunk-wise. To get a
    subset of this dataset, just provide an argument `url_indices` specifying
    which chunks you want to use. Must be combined with an appropriate
    :class:`pydgn.data.provider.IterableDataProvider`.

    Args:
        root (str): root folder where to store the dataset
        name (str): name of the dataset
        transform (Optional[Callable]): transformations to apply to each
            ``Data`` object at run time
        pre_transform (Optional[Callable]): transformations to apply to each
            ``Data`` object at dataset creation time
        url_indices (Optional[List]): list of indices used to extract a
            portion of the dataset
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        url_indices: Optional[List] = None,
        **kwargs,
    ):
        self.num_files = 5
        self.url_indices = [f"{i}" for i in range(self.num_files)]
        super().__init__(
            root, name, transform, pre_transform, url_indices, **kwargs
        )

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""
        The list of file names that must be present in order to skip
        downloading.
        """
        return [f"fake_processed_{i}.pt" for i in range(self.num_files)]

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 5

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 0

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return 1

    def download(self):
        r"""
        Does nothing, the data list is already provided
        """
        pass

    def process(self):
        r"""
        Creates a fake dataset and stores it to the :obj:`self.processed_dir`
        folder. Each file will contain a list of 10 fake graphs.
        """
        for i in range(len(self)):
            fake_graphs = []
            for g in range(10):
                fake_graphs.append(
                    Data(
                        x=torch.zeros(20, 5),
                        y=torch.zeros(1, 1),
                        edge_index=torch.zeros(2, 1).long(),
                    )
                )
            torch.save(fake_graphs, self.processed_paths[i])


class ChickenpoxDatasetInterface(TemporalDatasetInterface):
    r"""
    Class that acts as wrapper to Pytorch Geometric Temporal Chickenpox dataset

    Args:
        root (Union[str,Path]): root data folder
        name (str): name of the dataset
        lags: (how many original timesteps to aggregate into
            a single time step)
        kwargs: optional additional parameters to `ChickenpoxDatasetLoader`
            (from PyG Temporal)
    """

    def __init__(self, root, name, lags=4, **kwargs):
        super().__init__(root, name, **kwargs)

        self.lags = lags
        self.dataset = ChickenpoxDatasetLoader().get_dataset(lags=lags)

    def get_mask(self, data):
        """
        Computes the mask of time steps for which we need to make a prediction.
        In this case data is a Data object containing a snapshot of a
        single graph sequence, and the task is node classification
        at each time step

        Args:
            data: the data object

        Returns:
            A tensor indicating the time-steps at which we expect predictions
        """
        mask = torch.ones((1, 1))  # time_steps x 1
        return mask

    @property
    def dim_node_features(self) -> int:
        r"""
        Specifies the number of node features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self) -> int:
        r"""
        Specifies the number of edge features (after pre-processing, but in
        the end it depends on the model that is implemented).
        """
        return 0

    @property
    def dim_target(self) -> int:
        r"""
        Specifies the dimension of each target vector.
        """
        return 1

    def len(self):
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self)

    def __len__(self):
        r"""
        Returns the number of graphs stored in the dataset.
        Note: we need to implement both `len` and `__len__` to comply with
        PyG interface
        """
        return len(self.dataset.features)  # see DynamicGraphTemporalSignal
