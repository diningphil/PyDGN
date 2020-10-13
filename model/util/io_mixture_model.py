import torch
from model.util.mixture_model import MixtureModel


class IOMixtureModel(MixtureModel):
    def __init__(self, dim_features, c, emission, dirichlet_alpha, hidden_units=32):
        super().__init__(c, emission)

        self.dim_features = dim_features
        self.hidden_units = hidden_units
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.dirichlet_alpha]*self.C, dtype=torch.float32))


        # A simple MLP to compute the unnormalized posterior P(Q|x)
        '''
        self.p_Q_given_x = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_units),
                                               torch.nn.PReLU(),
                                               torch.nn.Linear(hidden_units, c),
                                               torch.nn.PReLU())
        '''
        self.p_Q_given_x = torch.nn.Sequential(torch.nn.Linear(dim_features, c),
                                               torch.nn.PReLU(), torch.nn.Softmax(dim=1))

    def forward(self, labels, inputs, batch, training=False):
        return self.e_step(labels, inputs, batch, training)

    def e_step(self, labels, inputs, batch, training):
        likely_labels, p_Q_given_x, objective, true_log_likelihood, prior_term, posterior_estimate \
            = self._e_step(labels, inputs, batch)

        if training:
            # Update accumulators (works with mini batches)
            self._m_step(labels, posterior_estimate)

        return likely_labels, p_Q_given_x, objective, true_log_likelihood, prior_term

    def m_step(self):
        # I need to re-init the emission accumulators, otherwise they will contain statistics of the previous epochs
        self.emission.update_parameters()
        self.emission.init_accumulators()

        # Do a full-batch backward pass to optimize the neural part.

        if self.device is not None:
            self.to(self.device)

    def _e_step(self, labels, inputs, batch):

        emission_of_labels = self.emission.get_distribution_of_labels(labels)

        p_Q_given_x = self.p_Q_given_x(inputs) + 1e-32  # P(Q|x) + "Laplacian smoothing"
        assert not torch.isnan(p_Q_given_x).any()

        # Use the posterior probability given the input to infer some labels (i.e., do not use the target label y)
        likely_labels = self.emission.infer(p_Q_given_x)

        # Broadcasting the "prior"
        numerator = torch.mul(emission_of_labels, p_Q_given_x)  # --> ?xC
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        posterior_estimate = torch.div(numerator, denominator)  # --> ?xC

        true_log_likelihood = numerator.sum(1).log().mean()
        complete_log_likelihood = torch.mean(torch.sum(torch.mul(posterior_estimate, torch.log(numerator)), dim=1))
        prior_term = torch.mean(self.dirichlet.log_prob(p_Q_given_x), dim=0)
        objective = complete_log_likelihood + prior_term

        return likely_labels, p_Q_given_x, objective, true_log_likelihood, prior_term, posterior_estimate

    def _m_step(self, labels, posterior_estimate):
        self.emission.update_accumulators(posterior_estimate.float(), labels)