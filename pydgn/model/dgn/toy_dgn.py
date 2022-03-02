from typing import List, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv

from pydgn.model.interface import ModelInterface


class ToyDGN(ModelInterface):
    """
    A toy Deep Graph Network used to test the library
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.readout = readout_class(dim_node_features=dim_embedding * num_layers,
                                         dim_edge_features=dim_edge_features,
                                         dim_target=dim_target, config=config)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

    def forward(self, data: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        node_embs = torch.cat(x_all, dim=1)

        return self.readout(node_embs, batch, **dict(edge_index=edge_index))