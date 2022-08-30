from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv
from torch_geometric_temporal import DCRNN

from pydgn.model.interface import ModelInterface


class ToyDGN(ModelInterface):
    """
    A toy Deep Graph Network used to test the library
    """

    def __init__(
        self,
        dim_node_features,
        dim_edge_features,
        dim_target,
        readout_class,
        config,
    ):
        super().__init__(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_class,
            config,
        )

        num_layers = config["num_layers"]
        dim_embedding = config["dim_embedding"]
        self.aggregation = config["aggregation"]  # can be mean or max

        if self.aggregation == "max":
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.readout = readout_class(
            dim_node_features=dim_embedding * num_layers,
            dim_edge_features=dim_edge_features,
            dim_target=dim_target,
            config=config,
        )

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_node_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        """
        Implements an Toy DGN with some SAGE graph convolutional layers.

        Args:
            data (torch_geometric.data.Batch): a batch of graphs

        Returns:
            the output depends on the readout passed to the model as argument.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == "max":
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        node_embs = torch.cat(x_all, dim=1)

        return self.readout(node_embs, batch, **dict(edge_index=edge_index))


class ToyDGNTemporal(ModelInterface):
    """
    Simple Temporal Deep Graph Network that can be used to test the library
    """

    def __init__(
        self,
        dim_node_features,
        dim_edge_features,
        dim_target,
        readout_class,
        config,
    ):
        super().__init__(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_class,
            config,
        )

        self.dim_embedding = 32
        filter_size = 1

        self.model = DCRNN(dim_node_features, self.dim_embedding, filter_size)
        self.linear = nn.Linear(self.dim_embedding, self.dim_embedding)

        self.readout = readout_class(
            dim_node_features=self.dim_embedding,
            dim_edge_features=dim_edge_features,
            dim_target=dim_target,
            config=config,
        )

    def forward(self, snapshot: Union[Data, Batch], prev_state=None):
        """
        Implements an Toy Temporal DGN with some DCRNN graph
        convolutional layers.

        Args:
            snapshot (`Union[Data, Batch]`): a graph or batch of graphs
                at timestep t
            prev_state (`torch.Tensor`): hidden state of the model
                (previous time step)

        Returns:
            the output depends on the readout passed to the model as argument.
        """
        # snapshot.x: Tensor of size (num_nodes_t x node_ft_size)
        # snapshot.edge_index: Adj of size (num_nodes_t x num_nodes_t)
        x, edge_index, mask = (
            snapshot.x,
            snapshot.edge_index,
            snapshot.time_prediction_mask,
        )

        h = self.model(x, edge_index, H=prev_state)
        h = torch.relu(h)

        # Node predictors assume the embedding is in field "x"
        out, _ = self.readout(h, snapshot.batch)

        return out, h
