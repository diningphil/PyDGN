import torch


class ProbabilisticReadout(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.K = dim_node_features
        self.Y = dim_target
        self.E = dim_edge_features
        self.eps = 1e-8

    def init_accumulators(self):
        raise NotImplementedError()

    def e_step(self, p_Q, x_labels, y_labels, batch):
        raise NotImplementedError()

    def infer(self, p_Q, x_labels, batch):
        raise NotImplementedError()

    def complete_log_likelihood(self, posterior, emission_target, batch):
        raise NotImplementedError()

    def _m_step(self, x_labels, y_labels, posterior, batch):
        raise NotImplementedError()

    def m_step(self):
        raise NotImplementedError()
