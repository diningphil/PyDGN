from pydgn.data.sampler import RandomSampler
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from pydgn.data.provider import seed_worker, DataProvider


class SingleGraphSequenceDataProvider(DataProvider):
    """
    This class is responsible for building the dynamic dataset at runtime.
    """

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


class MultipleGraphSequenceDataProvider(DataProvider):
    """
    IMPORTANT: This provider assumes all graph sequences have the same length!
    """

    @classmethod
    def collate_fn(cls, samples_list):
        # Decompose by timestep and graph sequence, then batch graphs belonging
        # to the same time step

        num_sequences = len(samples_list)
        assert num_sequences >= 1

        # we should have a list of T edge index tensors, T=num_timesteps
        num_timesteps = len(samples_list[0].edge_index)

        batched_graphs_t = []
        for t in range(num_timesteps):
            graphs_t = []
            for i in range(num_sequences):
                graph_i = samples_list[i]
                # x = graph_i.x[t]
                # edge_index = graph_i.edge_index[t]
                # edge_attr = graph_i.edge_attr[t]
                # pos = graph_i.pos[t]
                # mask = graph_i.mask[t]
                # graph_it = Data(x=x,edge_index=edge_index,edge_attr=edge_attr,pos=pos,mask=mask)
                # graph_it = Data(**{k:v[t] for k,v in graph_i.to_dict().items() if type(v) in [torch.tensor, list]})
                graphs_t.append(graph_i.__get_item__(t))
            print(graphs_t)
            exit(0)
            batched_graphs_t.append(Batch.from_data_list(graphs_t))
        return batched_graphs_t

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
                                    collate_fn=MultipleGraphSequenceDataProvider.collate_fn,
                                    **kwargs)
        else:
            dataloader = DataLoader(dataset, shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    collate_fn=MultipleGraphSequenceDataProvider.collate_fn,
                                    **kwargs)

        return dataloader
