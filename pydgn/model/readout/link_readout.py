import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from pydgn.model.interface import ReadoutInterface


class DotProductLinkReadout(ReadoutInterface):
    """
    Class that implements a simple readout mapping for link prediction via dot product
    """
    def forward(self, node_embeddings, batch, **kwargs):
        edge_index = kwargs['edge_index']

        z, _ = to_dense_batch(node_embeddings, batch)
        adj = to_dense_adj(edge_index, batch)

        batch_size, num_nodes, _ = z.size()
        '''
        >>> As = torch.randn(3,2,5)
        >>> Bs = torch.randn(3,5,4)
        >>> torch.einsum('bij,bjk->bik', As, Bs) # batch matrix multiplication
        s = F.sigmoid(torch.einsum('bij,bji->bii', x, torch.transpose(x)))  # batch matrix multiplication
        '''
        return None, node_embeddings, torch.sigmoid(torch.matmul(z, z.transpose(1, 2))), adj
