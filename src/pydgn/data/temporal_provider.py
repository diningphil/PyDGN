from torch.utils.data import Subset, DataLoader
from pydgn.data.provider import seed_worker, DataProvider


class SingleGraphSequenceDataProvider(DataProvider):
    """
    This class is responsible for building the dynamic dataset at runtime.
    """

    def __init__(self, data_root, splits_root, splits_filepath, dataset_class, dataset_name, outer_folds, inner_folds,
                 num_workers, pin_memory):
        super().__init__(data_root, splits_root,
                                                        splits_filepath,
                                                        dataset_class,
                                                        dataset_name,
                                                        outer_folds, inner_folds,
                                                        num_workers, pin_memory)

    @classmethod
    def collate_fn(cls, samples_list):
        return samples_list

    def _get_loader(self, indices, **kwargs):
        dataset = self._get_dataset(**kwargs)
        dataset = Subset(dataset, indices)

        assert self.exp_seed is not None, 'DataLoader seed has not been specified! Is this a bug?'
        kwargs['worker_init_fn'] = lambda worker_id: seed_worker(worker_id, self.exp_seed)

        # Using Pytorch default DataLoader instead of PyG, to return list of graphs
        dataloader = DataLoader(dataset, num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                collate_fn=SingleGraphSequenceDataProvider.collate_fn,
                                **kwargs)
        return dataloader
