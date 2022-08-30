from typing import Callable, Tuple, Optional, List

import torch
from torch_geometric.data import Batch


class ModelInterface(torch.nn.Module):
    r"""
    Provides the signature for any main model to be trained under PyDGN

    Args:
        dim_node_features (int): dimension of node features
            (according to the :class:`~pydgn.data.dataset.DatasetInterface`
            property)
        dim_edge_features (int): dimension of edge features
            (according to the :class:`~pydgn.data.dataset.DatasetInterface`
            property)
        dim_target (int): dimension of the target
            (according to the :class:`~pydgn.data.dataset.DatasetInterface`
            property)
        readout_class (Callable[...,:class:`torch.nn.Module`]):
            class of the module implementing the readout. This is optional,
            but useful to put different readouts to try in the config file
        config (dict): config dictionary containing all the necessary
            hyper-parameters plus additional information (if needed)
    """

    def __init__(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        readout_class: Callable[..., torch.nn.Module],
        config: dict,
    ):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.readout_class = readout_class
        self.config = config

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        r"""
        Performs a forward pass over a batch of graphs

        Args:
            data (:class:`torch_geometric.data.Batch`): a batch of graphs

        Returns:
            a tuple (model's output, [optional] node embeddings,
            [optional] additional outputs
        """
        raise NotImplementedError("You need to implement this method!")


class ReadoutInterface(torch.nn.Module):
    r"""
    Provides the signature for any readout to be trained under PyDGN

    Args:
        dim_node_features (int): dimension of node features
            (according to the :class:`~pydgn.data.dataset.DatasetInterface`
            property)
        dim_edge_features (int): dimension of edge features
            (according to the :class:`~pydgn.data.dataset.DatasetInterface`
            property)
        dim_target (int): dimension of the target
            (according to the :class:`~pydgn.data.dataset.DatasetInterface`
            property)
        config (dict): config dictionary containing all the necessary
            hyper-parameters plus additional information (if needed)
    """

    def __init__(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        config: dict,
    ):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(
        self, node_embeddings: torch.tensor, batch: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        r"""
        Performs a forward pass over a batch of graphs

        Args:
            node_embeddings (:class:`torch_geometric.data.Batch`):
                the node embeddings
            batch (:class:`torch.Tensor`): the usual ``batch`` object of PyG
            kwargs (dict): additional and optional arguments

        Returns:
            a tuple (model's output, [optional] node embeddings,
            [optional] additional outputs
        """
        raise NotImplementedError("You need to implement this method!")
