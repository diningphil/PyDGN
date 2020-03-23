import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from models.predictors.Predictor import GraphPredictor


class IdentityGraphPredictor(GraphPredictor):

    def __init__(self, dim_features, dim_target, config):
        super().__init__(dim_features, dim_target, config)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return x


class LinearGraphPredictor(GraphPredictor):

    def __init__(self, dim_features, dim_target, config):
        super().__init__(dim_features, dim_target, config)
        self.W = nn.Linear(dim_features, dim_target, bias=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.W(x), x


class MLPGraphPredictor(nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MLPGraphPredictor, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_global = nn.Linear(dim_features, hidden_units)
        self.out = nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.out(F.relu(self.fc_global(x)))


class CGMMMLPGraphPredictor(GraphPredictor):

    def __init__(self, dim_features, dim_target, config):
        super().__init__(dim_features, dim_target, config)
        hidden_units = config['dim_embedding']

        self.fc_global = nn.Linear(dim_features, hidden_units)
        self.out = nn.Linear(hidden_units, dim_target)

    def forward(self, data):

        data, extra = data[0], data[1]

        x, edge_index, batch = extra.g_outs.squeeze().float(), data.edge_index, data.batch

        # Concat all dimensions into a single vector
        x = torch.reshape(x, (x.shape[0], -1))

        return self.out(F.relu(self.fc_global(x))), x
