import torch.nn as nn
from torch_geometric.nn import global_add_pool

from pydgn.model.interface import ReadoutInterface


class LinearGraphReadout(ReadoutInterface):
    """
    Class that implements a simple readout mapping for graph prediction
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, node_embeddings, batch, **kwargs):
        hg = global_add_pool(node_embeddings, batch)
        return self.W(hg), node_embeddings
