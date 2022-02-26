import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


class GraphPredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        raise NotImplementedError('You need to implement this method!')


class IdentityGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, x


class LinearGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.W(x), x
