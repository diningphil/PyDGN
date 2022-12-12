import math
import random
from typing import Union, Callable, List

import numpy as np
import torch
import torch_geometric.loader
from torch.utils.data import Subset
from torch_geometric.data import Data, Batch
from torch_geometric.loader.dataloader import Collater

import pydgn.data.dataset
from pydgn.data.dataset import DatasetInterface
from pydgn.data.dataset import TemporalDatasetInterface
from pydgn.data.sampler import RandomSampler
from pydgn.data.splitter import (
    Splitter,
    LinkPredictionSingleGraphSplitter,
    SingleGraphSplitter,
)
from pydgn.data.util import load_dataset


def seed_worker(exp_seed, worker_id):
    r"""
    Used to set a different, but reproducible, seed for all data-retriever
    workers. Without this, all workers will retrieve the data in the same
    order (important for Iterable-style datasets).

    Args:
        exp_seed (int): base seed to be used for reproducibility
        worker_id (int): id number of the worker
    """
    np.random.seed(exp_seed + worker_id)
    random.seed(exp_seed + worker_id)
    torch.manual_seed(exp_seed + worker_id)
    torch.cuda.manual_seed(exp_seed + worker_id)


class DataProvider:
    r"""
    A DataProvider object retrieves the correct data according to the external
    and internal data splits. It can be additionally used to augment the data,
    or to create a specific type of data loader. The base class does nothing
    special, but here is where the i-th element of a dataset could be
    pre-processed before constructing the mini-batches.

    IMPORTANT: if the dataset is to be shuffled, you MUST use a
    :class:`pydgn.data.sampler.RandomSampler` object to determine the
    permutation.

    Args:
        data_root (str): the path of the root folder in which data is stored
        splits_filepath (str): the filepath of the splits. with additional
            metadata
        dataset_class
            (Callable[...,:class:`pydgn.data.dataset.DatasetInterface`]):
            the class of the dataset
        data_loader_class
            (Union[Callable[...,:class:`torch.utils.data.DataLoader`],
            Callable[...,:class:`torch_geometric.loader.DataLoader`]]):
            the class of the data loader to use
        data_loader_args (dict): the arguments of the data loader
        dataset_name (str): the name of the dataset
        outer_folds (int): the number of outer folds for risk assessment.
            1 means hold-out, >1 means k-fold
        inner_folds (int): the number of outer folds for model selection.
            1 means hold-out, >1 means k-fold

    """

    def __init__(
        self,
        data_root: str,
        splits_filepath: str,
        dataset_class: Callable[..., pydgn.data.dataset.DatasetInterface],
        dataset_name: str,
        data_loader_class: Union[
            Callable[..., torch.utils.data.DataLoader],
            Callable[..., torch_geometric.loader.DataLoader],
        ],
        data_loader_args: dict,
        outer_folds: int,
        inner_folds: int,
    ):

        self.exp_seed = None

        self.data_root = data_root
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name

        self.data_loader_class = data_loader_class
        self.data_loader_args = data_loader_args

        self.outer_folds = outer_folds
        self.inner_folds = inner_folds

        self.outer_k = None
        self.inner_k = None

        self.splits_filepath = splits_filepath
        self.splitter = None

        # use this to avoid instantiating multiple versions of the same
        # dataset when no run-time
        # specific arguments are needed
        self.dataset = None

        self.dim_node_features = None
        self.dim_edge_features = None
        self.dim_target = None

    def set_exp_seed(self, seed: int):
        r"""
        Sets the experiment seed to give to the DataLoader.
        Helps with reproducibility.

        Args:
            seed (int): id of the seed
        """
        self.exp_seed = seed

    def set_outer_k(self, k: int):
        r"""
        Sets the parameter k of the `risk assessment` procedure.
        Called by the evaluation modules to load the correct data subset.

        Args:
            k (int): the id of the fold, ranging from 0 to K-1.
        """
        self.outer_k = k

    def set_inner_k(self, k):
        r"""
        Sets the parameter k of the `model selection` procedure. Called by the
        evaluation modules to load the correct subset of the data.

        Args:
            k (int): the id of the fold, ranging from 0 to K-1.
        """
        self.inner_k = k

    def _get_splitter(self) -> Splitter:
        """
        Instantiates the splitter with the parameters stored
        in the file ``self.splits_filepath``

        Returns:
            a :class:`~pydgn.data.splitter.Splitter` object
        """
        if self.splitter is None:
            self.splitter = Splitter.load(self.splits_filepath)
        return self.splitter

    def _get_dataset(self, **kwargs: dict) -> DatasetInterface:
        """
        Instantiates the dataset. Relies on the parameters stored in
        the ``dataset_kwargs.pt`` file.

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset. Not used in the base version

        Returns:
            a :class:`~pydgn.data.dataset.DatasetInterface` object
        """
        assert "shuffle" not in kwargs, (
            "Your implementation of _get_loader should remove `shuffle` "
            "from kwargs before calling _get_dataset"
        )
        assert "batch_size" not in kwargs, (
            "Your implementation of _get_loader should remove `batch_size` "
            "from kwargs before calling _get_dataset"
        )

        if kwargs is not None and len(kwargs) != 0:
            # we probably need to pass run-time specific parameters,
            # so load the dataset in memory again
            # an example is the subset of urls in Iterable style datasets
            dataset = load_dataset(
                self.data_root, self.dataset_name, self.dataset_class, **kwargs
            )
        else:
            if self.dataset is None:
                dataset = load_dataset(
                    self.data_root, self.dataset_name, self.dataset_class
                )
                self.dataset = dataset
            else:
                dataset = self.dataset

        self.dim_node_features = dataset.dim_node_features
        self.dim_edge_features = dataset.dim_edge_features
        self.dim_target = dataset.dim_target

        return dataset

    def _get_loader(
        self, indices: list, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Instantiates the data loader.

        Args:
            indices (sequence): Indices in the whole set selected for subset
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        shuffle = kwargs.pop("shuffle", False)
        batch_size = kwargs.pop("batch_size", 1)

        dataset: DatasetInterface = self._get_dataset(**kwargs)
        dataset = Subset(dataset, indices)

        assert (
            self.exp_seed is not None
        ), "DataLoader's seed has not been specified! Is this a bug?"

        # no need to set worker seed in map-stye dataset, see pytorch doc about reproducibility
        # this would also cause transforms based on random sampling to behave in the same way every time
        # kwargs["worker_init_fn"] = lambda worker_id: seed_worker(
        #     worker_id, self.exp_seed
        # )

        kwargs.update(self.data_loader_args)

        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = self.data_loader_class(
                dataset, sampler=sampler, batch_size=batch_size, **kwargs
            )
        else:
            dataloader = self.data_loader_class(
                dataset, shuffle=False, batch_size=batch_size, **kwargs
            )

        return dataloader

    def get_inner_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for model selection associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        indices = splitter.inner_folds[self.outer_k][self.inner_k].train_idxs
        return self._get_loader(indices, **kwargs)

    def get_inner_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for model selection associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        indices = splitter.inner_folds[self.outer_k][self.inner_k].val_idxs
        return self._get_loader(indices, **kwargs)

    def get_outer_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        return self._get_loader(train_indices, **kwargs)

    def get_outer_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for risk assessment associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        val_indices = splitter.outer_folds[self.outer_k].val_idxs
        return self._get_loader(val_indices, **kwargs)

    def get_outer_test(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the test set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        indices = splitter.outer_folds[self.outer_k].test_idxs
        return self._get_loader(indices, **kwargs)

    def get_dim_node_features(self) -> int:
        r"""
        Returns the number of node features of the dataset

        Returns:
            the value of the property ``dim_node_features`` in the dataset

        """
        if self.dim_node_features is None:
            raise Exception(
                "You should first initialize the dataset "
                "by creating a data loader!"
            )
        return self.dim_node_features

    def get_dim_edge_features(self) -> int:
        r"""
        Returns the number of node features of the dataset

        Returns:
            the value of the property ``dim_edge_features`` in the dataset

        """
        if self.dim_edge_features is None:
            raise Exception(
                "You should first initialize the dataset "
                "by creating a data loader!"
            )
        return self.dim_edge_features

    def get_dim_target(self) -> int:
        r"""
        Returns the dimension of the target for the task

        Returns:
            the value of the property ``dim_target`` in the dataset

        """
        if self.dim_target is None:
            raise Exception(
                "You should first initialize the dataset "
                "by creating a data loader!"
            )
        return self.dim_target


class IterableDataProvider(DataProvider):
    r"""
    A DataProvider object that allows to fetch data from an Iterable-style
    Dataset (see :class:`pydgn.data.dataset.IterableDatasetInterface`).
    """

    def _get_loader(
        self, indices: list, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Instantiates the data loader, passing to the dataset an additional
        `url_indices` argument with the indices to fetch. This is because
        each time this method is called with different indices a separate
        instance of the dataset is called.

        Args:
            indices (sequence): Indices in the whole set selected for subset
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        shuffle = kwargs.pop("shuffle", False)
        batch_size = kwargs.pop("batch_size", 1)

        # we will overwrite the dataset each time the loader is called
        dataset = self._get_dataset(**{"url_indices": indices})

        if shuffle:
            dataset.shuffle_urls(True)
            dataset.shuffle_urls_elements(True)

        assert (
            self.exp_seed is not None
        ), "DataLoader's seed has not been specified! Is this a bug?"

        # Define a worker_init_fn that configures each dataset copy differently
        # this is called only when num_workers is set to a value > 0
        def worker_init_fn(worker_id: int):
            """
            Set the seeds for the worker and computes the range of
            samples ids to fetch.
            """
            worker_info = torch.utils.data.get_worker_info()
            num_workers = worker_info.num_workers
            assert num_workers > 0

            # Set the random seed
            seed_worker(worker_id, self.exp_seed)

            # Get the dataset and overall length
            dataset = (
                worker_info.dataset
            )  # the dataset copy in this worker process
            dataset_length = len(
                dataset
            )  # dynamic, already refers to the subset of urls!

            per_worker = int(
                math.ceil((dataset_length) / float(worker_info.num_workers))
            )

            start = worker_id * per_worker
            end = worker_id * per_worker + per_worker

            # configure the dataset to only process the split workload
            dataset.splice(start, end)

        kwargs.update(self.data_loader_args)

        dataloader = self.data_loader_class(
            dataset,
            sampler=None,
            collate_fn=Collater(None, None),
            worker_init_fn=worker_init_fn,
            batch_size=batch_size,
            **kwargs,
        )
        return dataloader


class SingleGraphSequenceDataProvider(DataProvider):
    """
    This class is responsible for building the dynamic dataset at runtime.
    """

    @classmethod
    def collate_fn(cls, samples_list: List[Data]) -> List[Batch]:
        r"""
        Creates a Batch object for each sample in the data list.

        Args:
            samples_list (`List[Data]`): the list of graphs to batch

        Returns:
            a list of Batch objects
        """
        return [Batch.from_data_list([d]) for d in samples_list]

    def _get_loader(
        self, indices: list, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Instantiates the data loader.
        Only works with torch.utils.data.DataLoader.

        Args:
            indices (sequence): Indices in the whole set selected for subset
            kwargs (dict): a dictionary of additional arguments to be
                passed to the dataset being loaded.
                Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        shuffle = kwargs.pop("shuffle", False)
        assert not shuffle, (
            "Shuffle cannot be True with a single graph " "sequence dataset."
        )
        batch_size = kwargs.pop("batch_size", 1)

        dataset: TemporalDatasetInterface = self._get_dataset(**kwargs)
        dataset = Subset(dataset, indices)

        assert (
            self.exp_seed is not None
        ), "DataLoader seed has not been specified! Is this a bug?"

        # no need to set worker seed in map-stye dataset, see pytorch doc about reproducibility
        # this would also cause transforms based on random sampling to behave in the same way every time
        # kwargs["worker_init_fn"] = lambda worker_id: seed_worker(
        #     worker_id, self.exp_seed
        # )

        # Using Pytorch default DataLoader instead of PyG, to return list of
        # graphs
        assert (
            self.data_loader_class == torch.utils.data.DataLoader
        ), "SingleGraphSequenceDataProvider accepts a torch DataLoader only!"
        dataloader = self.data_loader_class(
            dataset,
            collate_fn=SingleGraphSequenceDataProvider.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            **kwargs,
        )

        kwargs.update(self.data_loader_args)
        return dataloader


class SingleGraphDataProvider(DataProvider):
    r"""
    A DataProvider subclass that only works with
    :class:`pydgn.data.splitter.SingleGraphSplitter`.

    Args:
        data_root (str): the path of the root folder in which data is stored
        splits_filepath (str): the filepath of the splits. with additional
            metadata
        dataset_class
            (Callable[...,:class:`pydgn.data.dataset.DatasetInterface`]):
            the class of the dataset
        data_loader_class
            (Union[Callable[...,:class:`torch.utils.data.DataLoader`],
            Callable[...,:class:`torch_geometric.loader.DataLoader`]]):
            the class of the data loader to use
        data_loader_args (dict): the arguments of the data loader
        dataset_name (str): the name of the dataset
        outer_folds (int): the number of outer folds for risk assessment.
            1 means hold-out, >1 means k-fold
        inner_folds (int): the number of outer folds for model selection.
            1 means hold-out, >1 means k-fold

    """

    def _get_splitter(self):
        """
        Instantiates the splitter with the parameters stored in the file
        ``self.splits_filepath``. Only works
        with `~pydgn.data.splitter.SingleGraphSplitter`.

        Returns:
            a :class:`~pydgn.data.splitter.Splitter` object
        """
        super()._get_splitter()  # loads splitter into self.splitter
        assert isinstance(
            self.splitter, SingleGraphSplitter
        ), "This class only works with a SingleGraphNodeSplitter splitter."
        return self.splitter

    def _get_dataset(self, **kwargs: dict) -> DatasetInterface:
        """
        Compared to superclass method, this always returns a new instance of
        the dataset, optionally passing extra arguments specified at runtime.

        Args:
            kwargs (dict): a dictionary of additional arguments to be
                passed to the dataset. Not used in the base version

        Returns:
            a :class:`~pydgn.data.dataset.DatasetInterface` object
        """
        assert "shuffle" not in kwargs, (
            "Your implementation of _get_loader should remove `shuffle` "
            "from kwargs before calling _get_dataset"
        )
        assert "batch_size" not in kwargs, (
            "Your implementation of _get_loader should remove `batch_size` "
            "from kwargs before calling _get_dataset"
        )

        # we probably need to pass run-time specific parameters, so load the
        # dataset in memory again
        # an example is the subset of urls in Iterable style datasets
        dataset = load_dataset(
            self.data_root, self.dataset_name, self.dataset_class, **kwargs
        )

        self.dim_node_features = dataset.dim_node_features
        self.dim_edge_features = dataset.dim_edge_features
        self.dim_target = dataset.dim_target

        return dataset

    def _get_loader(
        self, eval_indices: list, training_indices: list, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Compared to superclass method, returns a dataloader with the single
        graph augmented with additional fields. These are `training_indices`
        with the indices that refer to training nodes (usually always
        available) and `eval_indices`, which specify which are the indices
        on which to evaluate (can be validation or test).

        Args:
            indices (sequence): Indices in the whole set selected for subset
            eval_set (bool): whether or not indices refer to eval set
                (validation or test) or to training
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version
        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        shuffle = kwargs.pop("shuffle", False)
        batch_size = kwargs.pop("batch_size", 1)

        dataset: DatasetInterface = self._get_dataset(**kwargs)

        dataset.data.training_indices = torch.tensor(training_indices)
        dataset.data.eval_indices = torch.tensor(eval_indices)

        assert (
            self.exp_seed is not None
        ), "DataLoader's seed has not been specified! Is this a bug?"

        # no need to set worker seed in map-stye dataset, see pytorch doc about reproducibility
        # this would also cause transforms based on random sampling to behave in the same way every time
        # kwargs["worker_init_fn"] = lambda worker_id: seed_worker(
        #     worker_id, self.exp_seed
        # )

        kwargs.update(self.data_loader_args)

        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = self.data_loader_class(
                dataset, sampler=sampler, batch_size=batch_size, **kwargs
            )
        else:
            dataloader = self.data_loader_class(
                dataset, shuffle=False, batch_size=batch_size, **kwargs
            )

        return dataloader

    def get_inner_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for model selection associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][
            self.inner_k
        ].train_idxs
        return self._get_loader(
            train_indices, training_indices=train_indices, **kwargs
        )

    def get_inner_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for model selection associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][
            self.inner_k
        ].train_idxs
        val_indices = splitter.inner_folds[self.outer_k][self.inner_k].val_idxs
        return self._get_loader(
            val_indices, training_indices=train_indices, **kwargs
        )

    def get_outer_train(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the training set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        return self._get_loader(
            train_indices, training_indices=train_indices, **kwargs
        )

    def get_outer_val(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the validation set for risk assessment associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded.Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        val_indices = splitter.outer_folds[self.outer_k].val_idxs
        return self._get_loader(
            val_indices, training_indices=train_indices, **kwargs
        )

    def get_outer_test(
        self, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        Returns the test set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        test_indices = splitter.outer_folds[self.outer_k].test_idxs
        return self._get_loader(
            test_indices, training_indices=train_indices, **kwargs
        )


class LinkPredictionSingleGraphDataProvider(DataProvider):
    r"""
    An extension of the DataProvider class to deal with link prediction
    on a single graph. Designed to work with
    :class:`~pydgn.data.splitter.LinkPredictionSingleGraphSplitter`.
    We also assume the single-graph dataset can fit in memory
    **WARNING**: this class **modifies** the dataset by creating copies.
    It may not work if a "shared dataset" feature is added to PyDGN.
    """

    def _get_dataset(self, **kwargs):
        """
        Compared to superclass method, this always returns a new instance of
        the dataset, optionally passing extra arguments specified at runtime.

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset. Not used in the base version

        Returns:
            a :class:`~pydgn.data.dataset.DatasetInterface` object
        """
        assert "shuffle" not in kwargs, (
            "Your implementation of _get_loader should remove `shuffle` "
            "from kwargs before calling _get_dataset"
        )
        assert "batch_size" not in kwargs, (
            "Your implementation of _get_loader should remove `batch_size` "
            "from kwargs before calling _get_dataset"
        )

        # Since we modify the dataset, we need different istances of the same
        # graph
        return load_dataset(
            self.data_root, self.dataset_name, self.dataset_class, **kwargs
        )

    def _get_splitter(self):
        """
        Instantiates the splitter with the parameters stored in the file
            ``self.splits_filepath``. Only works with
            `~pydgn.data.splitter.LinkPredictionSingleGraphSplitter`.

        Returns:
            a :class:`~pydgn.data.splitter.Splitter` object
        """
        super()._get_splitter()  # loads splitter into self.splitter
        assert isinstance(self.splitter, LinkPredictionSingleGraphSplitter), (
            "This class only work with a "
            "LinkPredictionSingleGraphSplitter splitter."
        )
        return self.splitter

    def _get_loader(
        self, indices: list, **kwargs: dict
    ) -> Union[torch.utils.data.DataLoader, torch_geometric.loader.DataLoader]:
        r"""
        This method returns a data loader for a **single** graph augmented with
        additional fields. The `y` field becomes (y, positive EVAL edges,
        negative EVAL edges), where eval means these are the edges on which
        to evaluate losses and scores (in fact, eval could also mean
        training!). A list of different Data objects is created, where
        the evaluation edges are randomly permuted. This depends on the size
        of the batch that is specified.

        Args:
            indices (sequence): Indices in the whole set selected for subset
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version
        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object
        """
        # We may want to shuffle the edges of our single graph and take edge
        # batches
        # NOTE: EVAL edges can be TRAINING/VAL/TEST. It is on "eval" edges
        # that we compute the loss (and eventually do training)
        # changing names may be good
        shuffle = kwargs.pop("shuffle", False)
        batch_size = kwargs.pop("batch_size", 1)

        dataset = self._get_dataset(
            **kwargs
        )  # Custom implementation, we need a copy of the dataset every time
        assert len(dataset) == 1, (
            f"Provider accepts a single-graph dataset only, "
            f"but I see {len(dataset)} graphs"
        )
        y = dataset.data.y

        # Use indices to change edge_index and provide a y
        train, eval = indices
        pos_train_edges, train_attr, neg_train_edges = train
        pos_train_edges = torch.tensor(pos_train_edges, dtype=torch.long)
        train_attr = (
            torch.tensor(train_attr) if train_attr is not None else None
        )
        neg_train_edges = torch.tensor(neg_train_edges, dtype=torch.long)

        pos_eval_edges, eval_attr, neg_eval_edges = eval
        pos_eval_edges = torch.tensor(pos_eval_edges, dtype=torch.long)
        eval_attr = torch.tensor(eval_attr) if eval_attr is not None else None
        neg_eval_edges = torch.tensor(neg_eval_edges, dtype=torch.long)

        # The code below works because we are working with a single graph!
        dataset.data.edge_index = pos_train_edges
        dataset.data.edge_attr = train_attr
        # Eval can be training/val/test
        dataset.data.y = (y, pos_eval_edges, neg_eval_edges)

        if shuffle is True:
            pos_edge_indices = torch.randperm(pos_eval_edges.shape[1])
            neg_edge_indices = torch.randperm(neg_eval_edges.shape[1])
        else:
            pos_edge_indices = torch.arange(pos_eval_edges.shape[1])
            neg_edge_indices = torch.arange(neg_eval_edges.shape[1])

        # create batch of edges here, leaving other fields as is
        batched_edge_dataset = []

        permuted_pos_eval_edges = pos_eval_edges[:, pos_edge_indices]
        permuted_neg_eval_edges = neg_eval_edges[:, neg_edge_indices]
        permuted_edge_attr = (
            eval_attr[pos_edge_indices] if eval_attr is not None else None
        )
        done = False
        batch_start = 0
        while not done:
            if batch_size == 0:
                # Full batch
                pos_batch_end = permuted_pos_eval_edges.shape[1]
                neg_batch_end = permuted_neg_eval_edges.shape[1]
            else:
                # Create subgraph
                pos_batch_end = (
                    batch_start + batch_size
                    if (batch_start + batch_size)
                    < permuted_pos_eval_edges.shape[1]
                    else permuted_pos_eval_edges.shape[1]
                )
                neg_batch_end = (
                    batch_start + batch_size
                    if (batch_start + batch_size)
                    < permuted_neg_eval_edges.shape[1]
                    else permuted_neg_eval_edges.shape[1]
                )

            batch_pos_edge_indices = permuted_pos_eval_edges[
                :, batch_start:pos_batch_end
            ]
            batch_neg_edge_indices = permuted_neg_eval_edges[
                :, batch_start:neg_batch_end
            ]
            batch_edge_attr = (
                permuted_edge_attr[batch_start:pos_batch_end]
                if eval_attr is not None
                else None
            )
            batch_start += batch_size

            if (
                pos_batch_end == permuted_pos_eval_edges.shape[1]
                and neg_batch_end == permuted_neg_eval_edges.shape[1]
            ):
                done = True

            # create data object and append to list
            subgraph = Data(
                x=dataset.data.x,
                edge_index=dataset.data.edge_index,
                edge_attr=dataset.data.edge_attr,
                y=(y, batch_pos_edge_indices, batch_neg_edge_indices),
            )
            batched_edge_dataset.append(subgraph)

        assert (
            self.exp_seed is not None
        ), "DataLoader seed has not been specified! Is this a bug?"

        # no need to set worker seed in map-stye dataset, see pytorch doc about reproducibility
        # this would also cause transforms based on random sampling to behave in the same way every time
        # kwargs["worker_init_fn"] = lambda worker_id: seed_worker(
        #     worker_id, self.exp_seed
        # )

        kwargs.update(self.data_loader_args)

        # Single graph dataset, shuffle does not make sense (unless we know how
        # to do mini-batch training with nodes)
        dataloader = self.data_loader_class(
            batched_edge_dataset, batch_size=1, shuffle=False, **kwargs
        )

        return dataloader

    def get_inner_train(self, **kwargs):
        r"""
        Returns the training set for model selection associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][
            self.inner_k
        ].train_idxs
        eval_indices = train_indices
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_inner_val(self, **kwargs):
        r"""
        Returns the validation set for model selection associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][
            self.inner_k
        ].train_idxs
        eval_indices = splitter.inner_folds[self.outer_k][
            self.inner_k
        ].val_idxs
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_outer_train(self, **kwargs):
        r"""
        Returns the training set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        eval_indices = train_indices
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_outer_val(self, **kwargs):
        r"""
        Returns the validation set for risk assessment associated with
        specific outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        eval_indices = splitter.outer_folds[self.outer_k].val_idxs

        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_outer_test(self, **kwargs):
        r"""
        Returns the test set for risk assessment associated with specific
        outer and inner folds

        Args:
            kwargs (dict): a dictionary of additional arguments to be passed
                to the dataset being loaded. Not used in the base version

        Returns:
            a Union[:class:`torch.utils.data.DataLoader`,
            :class:`torch_geometric.loader.DataLoader`] object

        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        eval_indices = splitter.outer_folds[self.outer_k].test_idxs
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)
