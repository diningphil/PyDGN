import math
import torch
from torch_scatter import scatter

pi = torch.FloatTensor([math.pi])


def _compute_unigram(posteriors, use_continuous_states):
    C = posteriors.shape[1]

    if use_continuous_states:
        node_embeddings_batch = posteriors
    else:
        node_embeddings_batch = _make_one_hot(posteriors.argmax(dim=1), C)

    return node_embeddings_batch.double()


def _compute_bigram(posteriors, edge_index, batch, use_continuous_states):
    C = posteriors.shape[1]

    if use_continuous_states:
        # Code provided by Daniele Atzeni to speed up the computation!
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors.float()).repeat(1, C)
        tmp2 = posteriors.view(-1, 1).repeat(1, C).view(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)

    else:
        # Covert into one hot
        posteriors_one_hot = _make_one_hot(posteriors.argmax(dim=1), C).float()

        # Code provided by Daniele Atzeni to speed up the computation!
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors_one_hot).repeat(1, C)
        tmp2 = posteriors_one_hot.view(-1, 1).repeat(1, C).view(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)

    return node_bigram_batch.double()


def _make_one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C)
    one_hot[torch.arange(labels.size(0)), labels] = 1
    return one_hot


def global_min_pool(x, batch, size=None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='min')


def multivariate_diagonal_pdf(data, mean, var):
    '''
    Multivariate case, DIAGONAL cov. matrix. Computes probability distribution for each data point
    :param data: a vector of values
    :param mean: vector of mean values, size F
    :param var:  vector of variance values, size F
    :return:
    '''
    tmp =  torch.log(2 * pi)
    try:
        device = 'cuda:' + str(data.get_device())
        tmp = tmp.to(device)
    except Exception as e:
        pass
        
    diff = (data.float() - mean)
    log_normaliser = -0.5 * (tmp + torch.log(var))
    log_num = - (diff * diff) / (2 * var)
    log_probs = torch.sum(log_num + log_normaliser, dim=1)
    probs = torch.exp(log_probs)
    return probs