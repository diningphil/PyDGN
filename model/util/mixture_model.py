import torch
from torch_geometric.nn import global_mean_pool, global_add_pool


class MixtureModel(torch.nn.Module):
    def __init__(self, c, emission):
        super().__init__()
        self.device = None

        self.eps = 1e-8  # Laplace smoothing
        self.C = c
        self.emission = emission

        pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float32))
        self.prior = pr / pr.sum()

        # print(self.emission)

        self.init_accumulators()

    def to(self, device):
        super().to(device)
        self.device = device
        self.emission.to(device)

        self.prior = self.prior.to(device)
        self.prior_numerator = self.prior_numerator.to(device)
        self.prior_denominator = self.prior_denominator.to(device)

    def init_accumulators(self):

        # These are variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.emission.init_accumulators()
        self.prior_numerator = torch.full([self.C], self.eps, dtype=torch.float32)
        self.prior_denominator = torch.tensor([self.eps * self.C])

        # Do not delete this!
        if self.device:  # set by to() method
            self.to(self.device)

    def e_step(self, labels, batch, training=False):
        likelihood, posterior_ui = self._e_step(labels, batch)
        if training:
            # Update accumulators (works with mini batches)
            self._m_step(labels, posterior_ui)
        return likelihood, posterior_ui

    def m_step(self):

        self.emission.update_parameters()
        self.prior = self.prior_numerator / self.prior_denominator

        # I need to re-init accumulators, otherwise they will contain statistics of the previous epochs
        self.init_accumulators()

    def _e_step(self, labels, batch):

        emission_of_labels = self.emission.get_distribution_of_labels(labels)

        # Broadcasting the prior
        numerator = torch.mul(emission_of_labels, torch.reshape(self.prior, shape=[1, self.C]))  # --> ?xC

        denominator = torch.sum(numerator, dim=1, keepdim=True)

        posterior_estimate = torch.div(numerator, denominator)  # --> ?xC

        # -------------------------------- Likelihood ------------------------------- #

        likelihood = torch.sum(torch.mul(posterior_estimate, torch.log(numerator)))

        return likelihood, posterior_estimate

    def _m_step(self, labels, posterior_estimate):
        tmp = torch.sum(posterior_estimate, dim=0)
        # These are used at each minibatch
        self.prior_numerator += tmp
        self.prior_denominator += torch.sum(tmp)
        self.emission.update_accumulators(posterior_estimate, labels)