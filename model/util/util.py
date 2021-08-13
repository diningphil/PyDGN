import math

import torch
from torch_scatter import scatter

pi = torch.FloatTensor([math.pi])


class StirlingNumbers:

    def __init__(self):
        self.s_table = None

    def stirling(self, n, k):

        if self.s_table is None:
            self.s_table = torch.zeros(n + 1, k + 1, dtype=torch.float64)
            self.s_table[0, 0] = 1

            for i in range(1, n + 1):
                for j in range(1, min(k + 1, i + 1)):
                    self.s_table[i, j] = (i - 1) * self.s_table[i - 1, j] + self.s_table[i - 1, j - 1]
            return self.s_table[n, k]

        elif n < self.s_table.shape[0] and k < self.s_table.shape[1]:
            return self.s_table[n, k]

        elif n >= self.s_table.shape[0] and k >= self.s_table.shape[1]:
            old_n = self.s_table.shape[0]  # it was "n+1" when invoked with previous n
            rows_to_add = n + 1 - old_n
            old_k = self.s_table.shape[1]  # it was "k+1" when invoked with previous k
            columns_to_add = k + 1 - old_k
            new_table = torch.zeros(n + 1, k + 1, dtype=torch.float64)
            new_table[:old_n, :old_k] += self.s_table
            self.s_table = new_table

            for i in range(1, n + 1):
                for j in range(old_k, min(k + 1, i + 1)):
                    self.s_table[i, j] = (i - 1) * self.s_table[i - 1, j] + self.s_table[i - 1, j - 1]

            for i in range(old_n, n + 1):
                for j in range(1, min(k + 1, i + 1)):
                    self.s_table[i, j] = (i - 1) * self.s_table[i - 1, j] + self.s_table[i - 1, j - 1]

            return self.s_table[n, k]

        elif n < self.s_table.shape[0] and k >= self.s_table.shape[1]:
            old_k = self.s_table.shape[1]  # it was "k+1" when invoked with previous k
            columns_to_add = k + 1 - old_k
            self.s_table = torch.cat((self.s_table,
                                      torch.zeros(self.s_table.shape[0], columns_to_add, dtype=torch.float64)), dim=1)

            for i in range(1, n + 1):
                for j in range(old_k, min(k + 1, i + 1)):
                    self.s_table[i, j] = (i - 1) * self.s_table[i - 1, j] + self.s_table[i - 1, j - 1]
            return self.s_table[n, k]

        elif n >= self.s_table.shape[0] and k < self.s_table.shape[1]:
            old_n = self.s_table.shape[0]  # it was "n+1" when invoked with previous n
            rows_to_add = n + 1 - old_n
            self.s_table = torch.cat((self.s_table,
                                      torch.zeros(rows_to_add, self.s_table.shape[1], dtype=torch.float64)), dim=0)

            for i in range(old_n, n + 1):
                for j in range(1, min(k + 1, i + 1)):
                    self.s_table[i, j] = (i - 1) * self.s_table[i - 1, j] + self.s_table[i - 1, j - 1]
            return self.s_table[n, k]


def _compute_unigram(posteriors, use_continuous_states):
    C = posteriors.shape[1]

    if use_continuous_states:
        node_embeddings_batch = posteriors
    else:
        node_embeddings_batch = _make_one_hot(posteriors.argmax(dim=1), C)

    return node_embeddings_batch.double()


def _compute_bigram(posteriors, edge_index, batch, use_continuous_states):
    C = posteriors.shape[1]
    device = posteriors.get_device()
    device = 'cpu' if device == -1 else device

    if use_continuous_states:
        # Code provided by Daniele Atzeni to speed up the computation!
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]).to(device),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors.float()).repeat(1, C)
        tmp2 = posteriors.reshape(-1, 1).repeat(1, C).reshape(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)
    else:
        # Covert into one hot
        posteriors_one_hot = _make_one_hot(posteriors.argmax(dim=1), C).float()

        # Code provided by Daniele Atzeni to speed up the computation!
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]).to(device),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors_one_hot).repeat(1, C)
        tmp2 = posteriors_one_hot.reshape(-1, 1).repeat(1, C).reshape(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)

    return node_bigram_batch.double()


def _make_one_hot(labels, C):
    device = labels.get_device()
    device = 'cpu' if device == -1 else device
    one_hot = torch.zeros(labels.size(0), C).to(device)
    one_hot[torch.arange(labels.size(0)).to(device), labels] = 1
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
    tmp = torch.log(2 * pi)
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
