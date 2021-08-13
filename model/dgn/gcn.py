import torch
from torch_geometric.nn import GCNConv


class GCNLinkPrediction(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        super(GCNLinkPrediction, self).__init__()

        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target

        num_layers = config['num_convolutional_layers']
        hidden_units = config['hidden_units']

        self.layers = torch.nn.ModuleList([])

        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else hidden_units
            l = GCNConv(dim_input, hidden_units)
            self.layers.append(l)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)

        return x, x
