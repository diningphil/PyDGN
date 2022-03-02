import torch.nn as nn

from pydgn.model.interface import ReadoutInterface


class LinearNodeReadout(ReadoutInterface):
    """
    Class that implements a simple readout mapping for node prediction
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, node_embeddings, batch, **kwargs):
        return self.W(node_embeddings), node_embeddings
