import torch
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool


class ToyDGN(nn.Module):
    """
    Simple Deep Graph Network that can be used to test the library
    """

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        """
        Initializes the model.
        :param dim_node_features: arbitrary object holding node feature information
        :param dim_edge_features: arbitrary object holding edge feature information
        :param dim_target: arbitrary object holding target information
        :param predictor_class: the class of the predictor that will classify node/graph embeddings produced by this DGN
        :param config: the configuration dictionary to extract further hyper-parameters
        """
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.predictor = predictor_class(dim_node_features=dim_embedding * num_layers,
                                         dim_edge_features=dim_edge_features,
                                         dim_target=dim_target, config=config)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

    def forward(self, data):
        """
        Forward pass
        :param data: A PyTorch Geometric Data object, possibly with some additional fields
        :return: a prediction
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        data.x = x

        return self.predictor(data)


class DummyDGN(nn.Module):
    """
    Simple Deep Graph Network that can be used to test the library
    """

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        """
        Initializes the model.
        :param dim_node_features: arbitrary object holding node feature information
        :param dim_edge_features: arbitrary object holding edge feature information
        :param dim_target: arbitrary object holding target information
        :param predictor_class: the class of the predictor that will classify node/graph embeddings produced by this DGN
        :param config: the configuration dictionary to extract further hyper-parameters
        """
        super().__init__()
        dim_embedding = config['dim_embedding']
        self.transf = nn.Linear(dim_node_features, dim_embedding)
        self.readout = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        """
        Forward pass
        :param data: A PyTorch Geometric Data object, possibly with some additional fields
        :return: a prediction
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb = self.transf(x)
        out = self.readout(global_mean_pool(node_emb, batch))
        return out, node_emb


class DummySoftmaxDGN(nn.Module):
    """
    Simple Deep Graph Network that can be used to test the library
    """

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        """
        Initializes the model.
        :param dim_node_features: arbitrary object holding node feature information
        :param dim_edge_features: arbitrary object holding edge feature information
        :param dim_target: arbitrary object holding target information
        :param predictor_class: the class of the predictor that will classify node/graph embeddings produced by this DGN
        :param config: the configuration dictionary to extract further hyper-parameters
        """
        super().__init__()
        dim_embedding = config['dim_embedding']
        self.transf = nn.Linear(dim_node_features, dim_embedding)
        self.readout = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        """
        Forward pass
        :param data: A PyTorch Geometric Data object, possibly with some additional fields
        :return: a prediction
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb = self.transf(x)
        exp_node_emb = torch.exp(node_emb)
        sum_nodes = global_add_pool(exp_node_emb, batch)[batch]
        softmax_node_emb = exp_node_emb / sum_nodes
        out = self.readout(global_mean_pool(softmax_node_emb, batch))
        return out, node_emb


class DummyConvDGN(nn.Module):
    """
    Simple Deep Graph Network that can be used to test the library
    """

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        """
        Initializes the model.
        :param dim_node_features: arbitrary object holding node feature information
        :param dim_edge_features: arbitrary object holding edge feature information
        :param dim_target: arbitrary object holding target information
        :param predictor_class: the class of the predictor that will classify node/graph embeddings produced by this DGN
        :param config: the configuration dictionary to extract further hyper-parameters
        """
        super().__init__()
        dim_embedding = config['dim_embedding']
        self.transf = SAGEConv(dim_node_features, dim_embedding)
        self.readout = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        """
        Forward pass
        :param data: A PyTorch Geometric Data object, possibly with some additional fields
        :return: a prediction
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb = self.transf(x, edge_index)
        out = self.readout(global_mean_pool(node_emb, batch))
        return out, node_emb
