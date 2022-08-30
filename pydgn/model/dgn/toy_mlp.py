from typing import List, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from pydgn.model.interface import ModelInterface


class ToyMLP(ModelInterface):
    """
    A toy MLP model used to test the library. Technically,
    a DGN that ignores the adjacency matrix.
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

        dim_embedding = config["dim_embedding"]
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        """
        Implements an MLP (structure agnostic baseline)

        Args:
            data (torch_geometric.data.Batch): a batch of graphs

        Returns:
            a tuple (output, node_embedddings)
        """
        x, batch = data.x, data.batch

        hg = global_add_pool(x, batch)
        return self.W(hg), x
