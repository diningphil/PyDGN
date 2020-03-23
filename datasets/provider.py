import torch
from torch_geometric.data import DataLoader
from torch.utils.data import Subset

from datasets.datasets import ZipDataset
from datasets.utils import load_dataset, load_splitter
from datasets.sampler import RandomSampler


class DataProvider:
    """
    This class is responsible for building the dataset at runtime.
    """
    def __init__(self, data_root, dataset_class, dataset_name, outer_folds=10, inner_folds=1):
        """
        Initializes the object with all the relevant information
        :param data_root: the path of the root folder in which data is stored
        :param dataset_class: the class of the dataset
        :param dataset_name: the name of the dataset
        :param outer_folds: the number of outer folds for risk assessment. 1 means hold-out, >1 means k-fold
        :param inner_folds: the number of outer folds for model selection. 1 means hold-out, >1 means k-fold
        """
        self.data_root = data_root
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name

        self.outer_folds = outer_folds
        self.inner_folds = inner_folds

        self.outer_k = None
        self.inner_k = None

        self.splitter = None
        self.dataset = None

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
            self.splitter = load_splitter(self.dataset_name, self.outer_folds, self.inner_folds)
        return self.splitter

    def _get_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset(self.data_root, self.dataset_name, self.dataset_class)
        return self.dataset

    def _get_loader(self, indices, **kwargs):
        dataset = self._get_dataset()
        dataset = Subset(dataset, indices)
        shuffle = kwargs.pop("shuffle", False)
        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        else:
            dataloader = DataLoader(dataset, shuffle=False, **kwargs)

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
        indices = splitter.inner_folds[self.outer_k][self.inner_k].test_idxs
        return self._get_loader(indices, **kwargs)

    def get_outer_train(self, train_perc=0.9, **kwargs):
        """
        Returns the training set for risk assessment associated with specific outer and inner folds
        :param train_perc: the percentage of the outer training set that has to be used for training
        :param kwargs: any extra information that has to be passed to the _get_loader function
        :return: a Dataloader
        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        indices = splitter.outer_folds[self.outer_k].train_idxs
        train_indices = indices[:round(train_perc * len(indices))]
        return self._get_loader(train_indices, **kwargs)

    def get_outer_val(self, train_perc=0.9, **kwargs):
        """
        Returns the validation set for risk assessment associated with specific outer and inner folds
        :param train_perc: the percentage of the outer training set that has to be used for trainin
        :param kwargs: any extra information that has to be passed to the _get_loader function
        :return: a Dataloader
        """
        assert self.outer_k is not None
        splitter = self._get_splitter()
        indices = splitter.outer_folds[self.outer_k].train_idxs
        val_indices = indices[round(train_perc * len(indices)):]
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

    def get_dim_features(self):
        """
        Returns the number of features of the dataset
        :return: an arbitrary object that depends on the implementation of the dataset num_features property
        """
        dataset = self._get_dataset()
        return dataset.num_features

    def get_dim_target(self):
        """
        Returns the dimension of the target for the task
        :return: an arbitrary object that depends on the implementation of the dataset num_classes property
        """
        # TODO: what if it is a regression problem?
        dataset = self._get_dataset()
        return dataset.num_classes


class IncrementalDataProvider(DataProvider):
    """
    An extension of the DataProvider class to deal with the intermediate outputs produced by incremental architectures
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
            datasets = [dataset, dataset_extra]
            dataset = ZipDataset(*datasets)
        elif dataset_extra is None or (isinstance(dataset_extra, list) and len(dataset_extra) == 0):
            pass
        else:
            raise NotImplementedError("Check that extra is None, an empty list or a non-empty list")

        shuffle = kwargs.pop("shuffle", False)
        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        else:
            dataloader = DataLoader(dataset, shuffle=False, **kwargs)

        return dataloader


