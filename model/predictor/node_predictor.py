import torch
import torch.nn as nn

from pydgn.experiment.util import s2c
from pydgn.model.predictor.probabilistic_readout import ProbabilisticReadout


class ProbabilisticNodeReadout(ProbabilisticReadout):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.emission_class = s2c(config['emission'])
        self.CN = config['C']  # number of states of a generic node
        self.emission = self.emission_class(self.Y, self.CN)

    def init_accumulators(self):
        self.emission.init_accumulators()

    def e_step(self, p_Q, x_labels, y_labels, batch):
        emission_target = self.emission.e_step(x_labels, y_labels)  # ?n x CN
        readout_posterior = emission_target

        # true log P(y) using the observables
        # Mean of individual node terms
        p_x = (p_Q * readout_posterior).sum(dim=1)
        p_x[p_x == 0.] = 1.
        true_log_likelihood = p_x.log().sum(dim=0)

        return true_log_likelihood, readout_posterior, emission_target

    def infer(self, p_Q, x_labels, batch):
        return self.emission.infer(p_Q, x_labels)

    def complete_log_likelihood(self, eui, emission_target, batch):
        complete_log_likelihood = (eui * (emission_target.log())).sum(1).sum()
        return complete_log_likelihood

    def _m_step(self, x_labels, y_labels, eui, batch):
        self.emission._m_step(x_labels, y_labels, eui)

    def m_step(self):
        self.emission.m_step()
        self.init_accumulators()


class UnsupervisedProbabilisticNodeReadout(ProbabilisticReadout):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.emission_class = s2c(config['emission'])
        self.CN = config['C']  # number of states of a generic node
        self.emission = self.emission_class(self.K, self.CN)

    def init_accumulators(self):
        self.emission.init_accumulators()

    def e_step(self, p_Q, x_labels, y_labels, batch):
        # Pass x_labels as y_labels
        emission_target = self.emission.e_step(x_labels, x_labels)  # ?n x CN
        readout_posterior = emission_target

        # true log P(y) using the observables
        # Mean of individual node terms
        p_x = (p_Q * readout_posterior).sum(dim=1)
        p_x[p_x == 0.] = 1.
        true_log_likelihood = p_x.log().sum(dim=0)

        return true_log_likelihood, readout_posterior, emission_target

    def infer(self, p_Q, x_labels, batch):
        return self.emission.infer(p_Q, x_labels)

    def complete_log_likelihood(self, eui, emission_target, batch):
        complete_log_likelihood = (eui * (emission_target.log())).sum(1).sum()
        return complete_log_likelihood

    def _m_step(self, x_labels, y_labels, eui, batch):
        # Pass x_labels as y_labels
        self.emission._m_step(x_labels, x_labels, eui)

    def m_step(self):
        self.emission.m_step()
        self.init_accumulators()


class NodePredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        raise NotImplementedError('You need to implement this method!')


class IdentityNodePredictor(NodePredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)

    def forward(self, data):
        x = data.x  # Assume these are node embeddings
        return x, x


class LinearNodePredictor(NodePredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, data):
        x = data.x  # Assume these are node embeddings
        return self.W(x), x
