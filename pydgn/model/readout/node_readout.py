from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from pydgn.model.interface import ReadoutInterface


class LinearNodeReadout(ReadoutInterface):
    """
    Class that implements a simple readout mapping for node prediction
    """

    def __init__(
        self, dim_node_features, dim_edge_features, dim_target, config
    ):
        super().__init__(
            dim_node_features, dim_edge_features, dim_target, config
        )
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(
        self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        """
        Implements a linear readout of node embeddings

        Args:
            node_embeddings (`torch.Tensor`): the node embeddings of size `Nxd`
            batch (`torch.Tensor`): a tensor specifying to which graphs
                nodes belong to in the batch
            kwargs (dict): additional parameters (unused)

        Returns:
            a tuple (output, node_embeddings)
        """
        return self.W(node_embeddings), node_embeddings
