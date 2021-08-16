import random

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, DataLoader

from pydgn.data.dataset import ZipDataset
from pydgn.data.sampler import RandomSampler
from pydgn.data.splitter import Splitter, LinkPredictionSingleGraphSplitter
from pydgn.data.util import load_dataset


def seed_worker(exp_seed, worker_id):
    np.random.seed(exp_seed + worker_id)
    random.seed(exp_seed + worker_id)


class DataProvider:
    """
    This class is responsible for building the dataset at runtime.
    """

    def __init__(self, data_root, splits_root, splits_filepath, dataset_class, dataset_name, outer_folds, inner_folds,
                 num_workers, pin_memory):
        """
        Initializes the object with all the relevant information
        :param data_root: the path of the root folder in which data is stored
        :param splits_root: the path of the splits folder in which data splits are stored
        :param splits_filepath: the filepath of the splits. with additional metadata
        :param dataset_class: the class of the dataset
        :param dataset_name: the name of the dataset
        :param outer_folds: the number of outer folds for risk assessment. 1 means hold-out, >1 means k-fold
        :param inner_folds: the number of outer folds for model selection. 1 means hold-out, >1 means k-fold
        :param num_workers: the number of workers to use in the DataLoader. A value > 0 triggers multiprocessing. Useful to prefetch data from disk to GPU.
        :param pin_memory: should be True when working on GPU.
        """
        self.exp_seed = None

        self.data_root = data_root
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name

        self.outer_folds = outer_folds
        self.inner_folds = inner_folds

        self.outer_k = None
        self.inner_k = None

        self.splits_root = splits_root
        self.splits_filepath = splits_filepath
        self.splitter = None
        self.dataset = None
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def set_exp_seed(self, seed):
        """
        Sets the experiment seed to give to the DataLoader. Helps with reproducibility
        :param seed: id of the seed
        :return:
        """
        self.exp_seed = seed

    def set_outer_k(self, k):
        """
        Sets the parameter k of the risk assessment procedure. Called by the evaluation sub-package.
        :param k: the id of the fold, ranging from 0 to K-1
        :return:
        """
        self.outer_k = k

    def set_inner_k(self, k):
        """
        Sets the parameter k of the model selection procedure. Called by the evaluation sub-package.
        :param k: the id of the fold, ranging from 0 to K-1.
        :return:
        """
        self.inner_k = k

    def _get_splitter(self):
        if self.splitter is None:
            self.splitter = Splitter.load(self.splits_filepath)
        return self.splitter

    def _get_dataset(self, **kwargs):
        if self.dataset is None:
            self.dataset = load_dataset(self.data_root, self.dataset_name, self.dataset_class)
        return self.dataset

    def _get_loader(self, indices, **kwargs):
        dataset = self._get_dataset(**kwargs)
        dataset = Subset(dataset, indices)
        shuffle = kwargs.pop("shuffle", False)

        assert self.exp_seed is not None, 'DataLoader seed has not been specified! Is this a bug?'
        kwargs['worker_init_fn'] = lambda worker_id: seed_worker(worker_id, self.exp_seed)

        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    **kwargs)
        else:
            dataloader = DataLoader(dataset, shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    **kwargs)

        return dataloader

    def get_inner_train(self, **kwargs):
        """
        Returns the training set for model selection associated with specific outer and inner folds
        :param kwargs: any extra information that has to be passed to the _get_loader function
        :return: a Dataloader
        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        indices = splitter.inner_folds[self.outer_k][self.inner_k].train_idxs
        return self._get_loader(indices, **kwargs)

    def get_inner_val(self, **kwargs):
        """
        Returns the validation set for model selection associated with specific outer and inner folds
        :param kwargs: any extra information that has to be passed to the _get_loader function
        :return: a Dataloader
        """
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        indices = splitter.inner_folds[self.outer_k][self.inner_k].val_idxs
        return self._get_loader(indices, **kwargs)

    def get_outer_train(self, train_perc=None, **kwargs):
        """
        Returns the training set for risk assessment associated with specific outer and inner folds
        :param train_perc: the percentage of the outer training set that has to be used for training. If None, it uses (1 - inner validation_ratio)
        :param kwargs: any extra information that has to be passed to the _get_loader function
        :return: a Dataloader
        """
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs

        # Backward compatibility
        if not hasattr(splitter.outer_folds[self.outer_k], 'val_idxs') or splitter.outer_folds[
            self.outer_k].val_idxs is None:
            if train_perc is None:
                # Use the same percentage of validation samples as in model select.
                train_perc = 1 - splitter.val_ratio
            train_indices = train_indices[:round(train_perc * len(train_indices))]
        return self._get_loader(train_indices, **kwargs)

    def get_outer_val(self, train_perc=None, **kwargs):
        """
        Returns the validation set for risk assessment associated with specific outer and inner folds
        :param train_perc: the percentage of the outer training set that has to be used for training. If None, it uses (1 - inner validation_ratio)
        :param kwargs: any extra information that has to be passed to the _get_loader function
        :return: a Dataloader
        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs

        # Backward compatibility
        if not hasattr(splitter.outer_folds[self.outer_k], 'val_idxs') or splitter.outer_folds[
            self.outer_k].val_idxs is None:
            if train_perc is None:
                # Use the same percentage of validation samples as in model select.
                train_perc = 1 - splitter.val_ratio
            val_indices = train_indices[round(train_perc * len(train_indices)):]
        else:
            val_indices = splitter.outer_folds[self.outer_k].val_idxs

        return self._get_loader(val_indices, **kwargs)

    def get_outer_test(self, **kwargs):
        """
        Returns the test set for risk assessment associated with specific outer and inner folds
        :param kwargs: any extra information that has to be passed to the _get_loader function
        :return: a Dataloader
        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        indices = splitter.outer_folds[self.outer_k].test_idxs
        return self._get_loader(indices, **kwargs)

    def get_dim_node_features(self):
        """
        Returns the number of node features of the dataset
        :return: an arbitrary object that depends on the implementation of the dataset
        """
        return self._get_dataset().dim_node_features

    def get_dim_edge_features(self):
        """
        Returns the number of node features of the dataset
        :return: an arbitrary object that depends on the implementation of the dataset
        """
        return self._get_dataset().dim_edge_features

    def get_dim_target(self):
        """
        Returns the dimension of the target for the task
        :return: an arbitrary object that depends on the implementation of the dataset num_classes property
        """
        return self._get_dataset().dim_target


class LinkPredictionSingleGraphDataProvider(DataProvider):
    """
    An extension of the DataProvider class to deal with link prediction on a single graph
    Designed to work with LinkPredictionSingleGraphSplitter
    We assume the single-graph dataset can fit in memory
    # WARNING: BE CAREFUL IF SHARED DATA IS IMPLEMENTED IN THE FUTURE: EXTENDING THE DATASET MAY NOT WORK ANYMORE
    """

    # Since we modify the dataset, we need different istances of the same graph
    def _get_dataset(self, **kwargs):
        return load_dataset(self.data_root, self.dataset_name, self.dataset_class)

    def _get_splitter(self):
        super()._get_splitter()  # loads splitter into self.splitter
        assert isinstance(self.splitter,
                          LinkPredictionSingleGraphSplitter), "This class only work with a LinkPredictionSingleGraphSplitter splitter."
        return self.splitter

    def _get_loader(self, indices, **kwargs):
        dataset = self._get_dataset()  # Custom implementation, we need a copy of the dataset every time
        assert len(dataset) == 1, f"Provider accepts a single-graph dataset only, but I see {len(dataset)} graphs"
        y = dataset.data.y

        # Use indices to change edge_index and provide an y
        train, eval = indices
        pos_train_edges, train_attr, neg_train_edges = train
        pos_train_edges = torch.tensor(pos_train_edges, dtype=torch.long)
        train_attr = torch.tensor(train_attr) if train_attr is not None else None
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

        # We may want to shuffle the edges of our single graph and take edge batches
        # NOTE: EVAL edges can be TRAINING/VAL/TEST. It is on "eval" edges
        # that we compute the loss (and eventually do training)
        # TODO changing names may be good
        shuffle = kwargs.pop("shuffle")
        if shuffle is True:
            pos_edge_indices = torch.randperm(pos_eval_edges.shape[1])
            neg_edge_indices = torch.randperm(neg_eval_edges.shape[1])
        else:
            pos_edge_indices = torch.arange(pos_eval_edges.shape[1])
            neg_edge_indices = torch.arange(neg_eval_edges.shape[1])

        batch_size = kwargs.pop("batch_size")

        # create batch of edges here, leaving other fields as is
        batched_edge_dataset = []

        permuted_pos_eval_edges = pos_eval_edges[:, pos_edge_indices]
        permuted_neg_eval_edges = neg_eval_edges[:, neg_edge_indices]
        permuted_edge_attr = eval_attr[pos_edge_indices] if eval_attr is not None else None
        done = False
        batch_start = 0
        while not done:
            if batch_size == 0:
                # Full batch
                pos_batch_end = -1
                neg_batch_end = -1
            else:
                # Create subgraph
                pos_batch_end = batch_start + batch_size if (batch_start + batch_size) < permuted_pos_eval_edges.shape[
                    1] else -1
                neg_batch_end = batch_start + batch_size if (batch_start + batch_size) < permuted_neg_eval_edges.shape[
                    1] else -1

            batch_pos_edge_indices = permuted_pos_eval_edges[:, batch_start:pos_batch_end]
            batch_neg_edge_indices = permuted_neg_eval_edges[:, batch_start:neg_batch_end]
            batch_edge_attr = permuted_edge_attr[batch_start:pos_batch_end] if eval_attr is not None else None
            batch_start += batch_size

            if pos_batch_end == -1 or neg_batch_end == -1:  # to ensure balancing
                done = True

            # create data object and append to list
            subgraph = Data(x=dataset.data.x, edge_index=dataset.data.edge_index,
                            edge_attr=dataset.data.edge_attr,
                            y=(y, batch_pos_edge_indices, batch_neg_edge_indices))
            batched_edge_dataset.append(subgraph)

        assert self.exp_seed is not None, 'DataLoader seed has not been specified! Is this a bug?'
        kwargs['worker_init_fn'] = lambda worker_id: seed_worker(worker_id, self.exp_seed)

        # Single graph dataset, shuffle does not make sense (unless we know how to do mini-batch training with nodes)
        dataloader = DataLoader(batched_edge_dataset, batch_size=1,
                                shuffle=False, num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                **kwargs)

        return dataloader

    def get_inner_train(self, **kwargs):
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][self.inner_k].train_idxs
        eval_indices = train_indices
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_inner_val(self, **kwargs):
        assert self.outer_k is not None and self.inner_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.inner_folds[self.outer_k][self.inner_k].train_idxs
        eval_indices = splitter.inner_folds[self.outer_k][self.inner_k].val_idxs
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_outer_train(self, **kwargs):
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        eval_indices = train_indices
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_outer_val(self, **kwargs):
        assert self.outer_k is not None
        splitter = self._get_splitter()

        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        eval_indices = splitter.outer_folds[self.outer_k].val_idxs

        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)

    def get_outer_test(self, **kwargs):
        assert self.outer_k is not None
        splitter = self._get_splitter()
        train_indices = splitter.outer_folds[self.outer_k].train_idxs
        eval_indices = splitter.outer_folds[self.outer_k].test_idxs
        indices = train_indices, eval_indices
        return self._get_loader(indices, **kwargs)


class IncrementalDataProvider(DataProvider):
    """
    An extension of the DataProvider class to deal with the intermediate outputs produced by incremental architectures
    Used by CGMM to deal with node/graph classification/regression.
    """

    def _get_loader(self, indices, **kwargs):
        """
        Takes the "extra" argument from kwargs and zips it together with the original data into a ZipDataset
        :param indices: indices of the subset of the data to be extracted
        :param kwargs: an arbitrary dictionary
        :return: a DataLoader
        """
        dataset = self._get_dataset()
        dataset = Subset(dataset, indices)
        dataset_extra = kwargs.pop("extra", None)

        if dataset_extra is not None and isinstance(dataset_extra, list) and len(dataset_extra) > 0:
            assert len(dataset) == len(dataset_extra), (dataset, dataset_extra)
            datasets = [dataset, dataset_extra]
            dataset = ZipDataset(*datasets)
        elif dataset_extra is None or (isinstance(dataset_extra, list) and len(dataset_extra) == 0):
            pass
        else:
            raise NotImplementedError("Check that extra is None, an empty list or a non-empty list")

        assert self.exp_seed is not None, 'DataLoader seed has not been specified! Is this a bug?'
        kwargs['worker_init_fn'] = lambda worker_id: seed_worker(worker_id, self.exp_seed)

        shuffle = kwargs.pop("shuffle", False)
        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    **kwargs)
        else:
            dataloader = DataLoader(dataset, shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    **kwargs)

        return dataloader


class ContinualDataProvider(DataProvider):

    def _get_loader(self, indices, **kwargs):
        dataset = self._get_dataset(**kwargs)
        dataset = Subset(dataset, indices)
        shuffle = kwargs.pop("shuffle", False)
        kwargs.pop('task_id', None)
        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, num_workers=self.num_workers, pin_memory=self.pin_memory,
                                    **kwargs)
        else:
            dataloader = DataLoader(dataset, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory,
                                    **kwargs)

        return dataloader
