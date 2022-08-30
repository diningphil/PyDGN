import torch
from torch.utils.data import sampler

from pydgn.data.dataset import DatasetInterface


class RandomSampler(sampler.RandomSampler):
    """
    This sampler wraps the dataset and saves the random permutation applied
    to the samples, so that it will be available
    for further use (e.g. for saving graph embeddings in the original order).
    The permutation is saved in the 'permutation' attribute.

    Args:
        data_source (:class:`pydgn.data.DatasetInterface`): the dataset object
    """

    def __init__(self, data_source: DatasetInterface):
        super().__init__(data_source)
        self.permutation = None

    def __iter__(self):
        """
        Iterates over the samples according to a pre-determined permutation.

        Returns:
             An iterable with permuted indices.

        """
        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        return iter(self.permutation)
