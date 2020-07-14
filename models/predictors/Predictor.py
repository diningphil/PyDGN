import torch


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


class NodePredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        raise NotImplementedError('You need to implement this method!')


class LinkPredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, x, edge_index, batch):
        raise NotImplementedError('You need to implement this method!')