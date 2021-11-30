import random

import numpy as np
import torch
from pydgn.data.dataset import ZipDataset
from pydgn.data.sampler import RandomSampler
from pydgn.data.splitter import Splitter, LinkPredictionSingleGraphSplitter
from pydgn.data.util import load_dataset
from torch.utils.data import Subset
from torch_geometric.data import Data, DataLoader
from pydgn.data.provider import seed_worker, DataProvider


class SingleGraphSequenceDataProvider(DataProvider):
    """
    This class is responsible for building the dynamic dataset at runtime.
    """

    def __init__(self, data_root, splits_root, splits_filepath, dataset_class, dataset_name, outer_folds, inner_folds,
                 num_workers, pin_memory):
        super(SingleGraphSequenceDataProvider).__init__(data_root, splits_root,
                                                        splits_filepath,
                                                        dataset_class,
                                                        dataset_name,
                                                        outer_folds, inner_folds,
                                                        num_workers, pin_memory)

    def _get_loader(self, indices, **kwargs):
        dataset = self._get_dataset(**kwargs)
        dataset = Subset(dataset, indices)

        assert self.exp_seed is not None, 'DataLoader seed has not been specified! Is this a bug?'
        kwargs['worker_init_fn'] = lambda worker_id: seed_worker(worker_id, self.exp_seed)

        dataloader = DataLoader(dataset, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, **kwargs)

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
