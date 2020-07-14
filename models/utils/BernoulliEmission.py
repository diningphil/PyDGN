import torch


class BernoulliEmission:
    """
    This class models the emission part of a Bernoulli Mixture model where the posterior is computed in
    an arbitrary way. It implements an interface suitable to be easily integrates into CGMM.
    """
    def __init__(self, c):
        self.C = c  # clusters

        self.eps = 1e-8  # Laplace smoothing

        # Initialize accumulators
        self.emission_numerator = None
        self.emission_denominator = None

        self.bernoulli_params = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float32))

        # OLD DEBUG
        #self.bernoulli_params[0] = 1e-4
        #self.bernoulli_params[1] = 1. - 1e-4

        self.init_accumulators()

    def to(self, device):
        self.bernoulli_params = self.bernoulli_params.to(device)
        self.emission_numerator = self.emission_numerator.to(device)
        self.emission_denominator = self.emission_denominator.to(device)

    def init_accumulators(self):
        """
        This method initializes the accumulators for the EM algorithm.
        EM updates the parameters in batch, but needs to accumulate statistics in mini-batch style.
        :return:
        """
        self.emission_numerator = torch.full([self.C], self.eps, dtype=torch.float32)
        self.emission_denominator = torch.full([self.C], self.eps*2, dtype=torch.float32)
    
    def bernoulli_density(self, data, param):
        """
        Univariate case, computes probability distribution for each data point
        """
        return torch.mul(torch.pow(param, data), torch.pow(1-param, 1-data))

    def get_distribution_of_labels(self, labels):
        """
        For each cluster i, returns the probability associated to a specific label.
        :param labels:
        :return: a distribution associated to each label
        """

        emission_of_labels = None
        for i in range(0, self.C):
            if emission_of_labels is None:
                emission_of_labels = torch.reshape(self.bernoulli_density(labels, self.bernoulli_params[i]), (-1, 1))
            else:
                emission_of_labels = torch.cat((emission_of_labels,
                                                torch.reshape(self.bernoulli_density(labels, self.bernoulli_params[i]), (-1, 1))),
                                                dim=1)
        
        return emission_of_labels

    def infer(self, p_Q):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :return:
        """
        # We simply compute P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        inferred_y = torch.mm(p_Q, self.bernoulli_params.unsqueeze(1))  # ?x1
        return inferred_y

    def update_accumulators(self, posterior_estimate, labels):
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        self.emission_numerator += torch.sum(torch.mul(posterior_estimate,
                                                       labels), dim=0)   # --> 1 x C
        self.emission_denominator += torch.sum(posterior_estimate, dim=0)  # --> C

    def update_parameters(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.emission_distr = self.emission_numerator / self.emission_denominator

    def __str__(self):
        return str(self.bernoulli_params)
