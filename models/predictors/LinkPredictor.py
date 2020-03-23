import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.predictors.Predictor import LinkPredictor


class SimpleLinkPredictor(LinkPredictor):

    def __init__(self, dim_features, dim_target, config):
        super().__init__(dim_features, dim_target, config)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        z, _ = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        batch_size, num_nodes, _ = z.size()


        '''
        >>> As = torch.randn(3,2,5)
        >>> Bs = torch.randn(3,5,4)
        >>> torch.einsum('bij,bjk->bik', As, Bs) # batch matrix multiplication
        s = F.sigmoid(torch.einsum('bij,bji->bii', x, torch.transpose(x)))  # batch matrix multiplication
        '''
        return None, x, torch.sigmoid(torch.matmul(z, z.transpose(1, 2))), adj
