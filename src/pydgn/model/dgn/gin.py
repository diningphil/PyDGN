import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):
        super(GIN, self).__init__()

        self.config = config
        self.dropout = config['dropout']

        hidden_units = [config['dim_embedding'] for _ in range(config['layers'])]

        self.embeddings_dim = [hidden_units[0]] + hidden_units
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = config['train_eps']
        if config['aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_node_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                          Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                           Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0

        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer - 1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out


class GINDEBUG(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, L, A):
        super(GINDEBUG, self).__init__()

        hidden_units = 32

        self.embeddings_dim = hidden_units
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.L = L
        self.A = A
        self.dim_target = dim_target
        self.dim_features = dim_node_features
        train_eps = False
        self.pooling = global_add_pool
        # self.pooling = global_mean_pool

        out_emb_dim = self.embeddings_dim

        self.nn = Sequential(Linear(dim_node_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                             Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
        self.convs.append(GINConv(self.nn, train_eps=train_eps))  # Eq. 4.2
        self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, x, edge_index, edge_attr):
        out = 0

        x = x[:, 0, :]

        # Layer l ("convolution" layer)
        x = self.convs[0](x, edge_index)
        out += self.linears[0](x)

        # reshape vectors from ?x(LxA)xC into ?xLxAxC
        out = torch.reshape(out, (out.shape[0], self.L, self.A, self.dim_target))

        return out
