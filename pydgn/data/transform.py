import torch
from torch_geometric.utils import degree


class ConstantIfEmpty:
    r"""
    Adds a constant value to each node feature only if x is None.

    Args:
        value (int): The value to add. Default is ``1``
    """

    def __init__(self, value=1):
        self.value = value

    def __call__(self, data):
        if data.x is None:
            c = torch.full((data.num_nodes, 1), self.value, dtype=torch.float)
            data.x = c
        return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)


class ConstantEdgeIfEmpty:
    r"""
    Adds a constant value to each edge feature only if edge_attr is None.

    Args:
        value (int): The value to add. Default is ``1``)
    """

    def __init__(self, value=1):
        self.value = value

    def __call__(self, data):
        if data.edge_attr is None:
            c = torch.full((data.edge_index.shape[1], 1), self.value, dtype=torch.float)
            data.edge_attr = c
        return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)


class Degree:
    r"""
    Adds the node degree to the node features.

    Args:
        in_degree (bool): If set to :obj:`True`, will compute the in-degree of nodes instead of the out-degree.
        Not relevant if the graph is undirected (default: :obj:`False`).
        cat (bool): Concat node degrees to node features instead of replacing them. (default: :obj:`True`)
    """
    def __init__(self, in_degree: bool=False, cat: bool=True):
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.float).view(-1, 1)

        if x is not None and self.cat:
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
