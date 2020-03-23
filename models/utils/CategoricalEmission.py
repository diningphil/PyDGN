import torch


class CategoricalEmission:
    """
    This class models the emission part of a Categorical Mixture model where the posterior is computed in
    an arbitrary way. It implements an interface suitable to be easily integrates into CGMM.
    """
    def __init__(self, k, c, device):
        self.K = k  # discrete labels
        self.C = c  # clusters
        self.device = device
        if 'cuda' in device:
            self.emission_distr = torch.empty(self.K, self.C, dtype=torch.float64).cuda()
        else:
            self.emission_distr = torch.empty(self.K, self.C, dtype=torch.float64)

        self.eps = 1e-8  # Laplace smoothing

        # Initialize parameters
        self.emission_numerator = None
        self.emission_denominator = None

        for i in range(0, self.C):
            # For debugging w.r.t Numpy version
            # em = torch.from_numpy(np.random.uniform(size=self.K).astype(np.float32))
            if 'cuda' in device:
                em = torch.nn.init.uniform_(torch.empty(self.K, dtype=torch.float64)).cuda()
            else:
                em = torch.nn.init.uniform_(torch.empty(self.K, dtype=torch.float64))
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
        if 'cuda' in self.device:
            self.emission_numerator = torch.full([self.K, self.C], self.eps, dtype=torch.float64).cuda()
            self.emission_denominator = torch.full([1, self.C], self.eps * self.K, dtype=torch.float64).cuda()
        else:
            self.emission_numerator = torch.full([self.K, self.C], self.eps, dtype=torch.float64)
            self.emission_denominator = torch.full([1, self.C], self.eps * self.K, dtype=torch.float64)

    def get_distribution_of_labels(self, labels):
        """
        For each cluster i, returns the probability associated to a specific label.
        :param labels:
        :return: a distribution associated to each layer
        """
        labels_squeezed = torch.squeeze(labels).argmax(dim=1)
        emission_for_labels = torch.index_select(self.emission_distr, dim=0, index=labels_squeezed)  # ?xC

        return emission_for_labels

    def update_accumulators(self, posterior_estimate, labels):

        # removes dimensions of size 1 (current is ?x1)
        labels_squeezed = torch.squeeze(labels).argmax(dim=1)

        self.emission_numerator.index_add_(dim=0, source=posterior_estimate, index=labels_squeezed)   # --> K x C
        self.emission_denominator += torch.sum(posterior_estimate, dim=0)  # --> 1 x C

    def update_parameters(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.emission_distr = self.emission_numerator / self.emission_denominator