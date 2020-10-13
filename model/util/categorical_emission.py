import torch


class CategoricalEmission:
    """
    This class models the emission part of a Categorical Mixture model where the posterior is computed in
    an arbitrary way. It implements an interface suitable to be easily integrates into CGMM.
    """
    def __init__(self, k, c):
        self.K = k  # discrete labels
        self.C = c  # clusters
        self.emission_distr = torch.empty(self.K, self.C, dtype=torch.float32)

        self.eps = 1e-8  # Laplace smoothing

        # Initialize parameters
        self.emission_numerator = None
        self.emission_denominator = None

        for i in range(0, self.C):
            # For debugging w.r.t Numpy version
            # em = torch.from_numpy(np.random.uniform(size=self.K).astype(np.float32))
            em = torch.nn.init.uniform_(torch.empty(self.K, dtype=torch.float32))
            self.emission_distr[:, i] = em / em.sum()

        self.init_accumulators()

    def to(self, device):
        self.emission_distr.to(device)

    def export_parameters(self):
        return {'cat': self.emission_distr}

    def import_parameters(self, params):
        self.emission_distr = torch.from_numpy(params['cat'])

    def init_accumulators(self):
        """
        This method initializes the accumulators for the EM algorithm.
        EM updates the parameters in batch, but needs to accumulate statistics in mini-batch style.
        :return:
        """
        self.emission_numerator = torch.full([self.K, self.C], self.eps, dtype=torch.float32)
        self.emission_denominator = torch.full([1, self.C], self.eps * self.K, dtype=torch.float32)

    def get_distribution_of_labels(self, labels):
        """
        For each cluster i, returns the probability associated to a specific label.
        :param labels:
        :return: a distribution associated to each layer
        """
        labels_squeezed = torch.squeeze(labels).argmax(dim=1)
        emission_for_labels = torch.index_select(self.emission_distr, dim=0, index=labels_squeezed)  # ?xC

        return emission_for_labels

    def infer(self, p_Q):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :return:
        """
        # We simply compute P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        inferred_y = torch.mm(p_Q, self.emission_distr.transpose(0, 1))  # ?xK
        return inferred_y

    def update_accumulators(self, posterior_estimate, labels):

        # removes dimensions of size 1 (current is ?x1)
        labels_squeezed = labels.argmax(dim=1)

        self.emission_numerator.index_add_(dim=0, source=posterior_estimate, index=labels_squeezed)   # --> K x C
        self.emission_denominator += torch.sum(posterior_estimate, dim=0) # --> 1 x C

    def update_parameters(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.emission_distr = (self.emission_numerator / self.emission_denominator).detach()

    def __str__(self):
        return str(self.emission_distr)