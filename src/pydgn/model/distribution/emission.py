import math

import scipy
import scipy.cluster
import scipy.cluster.vq
import torch


# Interface for all emission distributions
from torch.nn import ModuleList


class EmissionDistribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def init_accumulators(self):
        raise NotImplementedError()

    def e_step(self, x_labels, y_labels):
        raise NotImplementedError()

    def infer(self, p_Q, x_labels):
        raise NotImplementedError()

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        raise NotImplementedError()

    def m_step(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


# do not replace replace with torch.distributions yet, it allows GPU computation
class Categorical(EmissionDistribution):

    def __init__(self, dim_target, dim_hidden_states):
        """
        :param dim_target: dimension of output alphabet
        :param dim_hidden_states: hidden states associated with each label
        """
        super().__init__()

        self.eps = 1e-8  # Laplace smoothing
        self.K = dim_target  # discrete output labels
        self.C = dim_hidden_states  # clusters
        self.emission_distr = torch.nn.Parameter(torch.empty(self.K, self.C,
                                                             dtype=torch.float32),
                                                 requires_grad=False)
        self.emission_numerator = torch.nn.Parameter(torch.empty_like(self.emission_distr),
                                                     requires_grad=False)
        for i in range(0, self.C):
            em = torch.nn.init.uniform_(torch.empty(self.K, dtype=torch.float32))
            self.emission_distr[:, i] = em / em.sum()
        self.init_accumulators()

    def _flatten_labels(self, labels):
        labels = torch.squeeze(labels)
        if len(labels.shape) > 1:
            # Compute discrete categories from one_hot_input
            labels_squeezed = labels.argmax(dim=1)
            return labels_squeezed
        return labels.long()

    def init_accumulators(self):
        torch.nn.init.constant_(self.emission_numerator, self.eps)

    def e_step(self, x_labels, y_labels):
        """
        For each cluster i, returns the probability associated with a specific label.
        :param x_labels: unused
        :param y_labels: output observables
        :return: a tensor of size ?xC representing the estimated posterior distribution for the E-step
        """
        y_labels_squeezed = self._flatten_labels(y_labels)

        # Returns the emission probability associated to each observable
        emission_obs = self.emission_distr[y_labels_squeezed]  # ?xC
        return emission_obs

    def infer(self, p_Q, x_labels):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :param x_labels: unused
        :return:
        """
        '''
        # OLD CODE
        # We simply compute P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        inferred_y = torch.mm(p_Q, self.emission_distr.transpose(0, 1))  # ?xK
        return inferred_y
        '''
        p_K_CS = p_Q.unsqueeze(1) * self.emission_distr.unsqueeze(0)  # ?xKxC
        p_KCS = p_K_CS.reshape((-1, self.K * self.C))  # ?xKC
        best_KCS = torch.argmax(p_KCS, dim=1)
        best_K = best_KCS // self.C
        best_CS = torch.remainder(best_KCS, self.C)
        return best_K.unsqueeze(1)

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        """
        Updates the minibatch accumulators
        :param x_labels: unused
        :param y_labels: output observable
        :param posterior_estimate: a ?xC posterior estimate obtained using the output observables
        """
        y_labels_squeezed = self._flatten_labels(y_labels)
        self.emission_numerator.index_add_(dim=0, source=posterior_estimate,
                                           index=y_labels_squeezed)  # KxC

    def m_step(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.emission_distr.data = torch.div(self.emission_numerator,
                                             self.emission_numerator.sum(0))
        curr_device = self.emission_distr.get_device()
        curr_device = curr_device if curr_device != -1 else 'cpu'
        # assert torch.allclose(self.emission_distr.sum(0), torch.tensor([1.]).to(curr_device))
        self.init_accumulators()

    def __str__(self):
        return str(self.emission_distr)


# do not replace replace with torch.distributions yet, it allows GPU computation
class ConditionedCategorical(EmissionDistribution):

    def __init__(self, dim_features, dim_target, dim_hidden_states):
        """
        :param dim_node_features: dimension of input alphabet
        :param dim_target: dimension of output alphabet
        :param dim_hidden_states: hidden states associated with each label
        """
        super().__init__()

        self.eps = 1e-8  # Laplace smoothing
        self.K = dim_features  # discrete input labels
        self.Y = dim_target  # discrete output labels
        self.C = dim_hidden_states  # clusters
        self.emission_distr = torch.nn.Parameter(torch.empty(self.K,
                                                             self.Y,
                                                             self.C,
                                                             dtype=torch.float32),
                                                 requires_grad=False)
        for i in range(self.C):
            for k in range(self.K):
                em = torch.nn.init.uniform_(torch.empty(self.Y,
                                                        dtype=torch.float32))
                self.emission_distr[k, :, i] = em / em.sum()

        self.emission_numerator = torch.nn.Parameter(torch.empty_like(self.emission_distr),
                                                     requires_grad=False)
        self.init_accumulators()

    def init_accumulators(self):
        torch.nn.init.constant_(self.emission_numerator, self.eps)

    def e_step(self, x_labels, y_labels):
        """
        For each cluster i, returns the probability associated with a specific input and output label.
        :param x_labels: input observables
        :param y_labels: output observables
        :return: a tensor of size ?xC representing the estimated posterior distribution for the E-step
        """
        x_labels_squeezed = self._flatten_labels(x_labels)
        y_labels_squeezed = self._flatten_labels(y_labels)
        emission_of_labels = self.emission_distr[x_labels_squeezed, y_labels, :]
        return emission_of_labels  # ?xC

    def infer(self, p_Q, x_labels):
        """
        Compute probability of a label given the probability P(Q|x) as argmax_y \sum_i P(y|Q=i,x)P(Q=i|x)
        :param p_Q: tensor of size ?xC
        :return:
        """
        # We simply compute P(y|x) = \sum_i P(y|Q=i,x)P(Q=i|x) for each node
        x_labels_squeezed = self._flatten_labels(x_labels)
        # First, condition the emission on the input labels
        emission_distr_given_x = self.emission_distr[x_labels_squeezed, :, :]
        # Then, perform inference
        inferred_y = p_Q.unsqueeze(1) * emission_distr_given_x  # ?xYxC
        inferred_y = torch.sum(inferred_y, dim=2)  # ?xY
        return inferred_y

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        """
        Updates the minibatch accumulators
        :param x_labels: unused
        :param y_labels: output observable
        :param posterior_estimate: a ?xC posterior estimate obtained using the output observables
        """

        x_labels_squeezed = self._flatten_labels(x_labels)
        y_labels_squeezed = self._flatten_labels(y_labels)

        for k in range(self.K):
            # filter nodes based on their input value
            mask = x_labels_squeezed == k
            y_labels_masked = y_labels_squeezed[mask]
            posterior_estimate_masked = posterior_estimate[mask, :]

            # posterior_estimate has shape ?xC
            delta_numerator = torch.zeros(self.Y, self.C)
            delta_numerator.index_add_(dim=0, source=posterior_estimate_masked,
                                       index=y_labels_masked)  # --> Y x C
            self.emission_numerator[k, :, :] += delta_numerator

    def m_step(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.emission_distr.data = torch.div(self.emission_numerator,
                                             torch.sum(self.emission_numerator,
                                                       dim=1,
                                                       keepdim=True))
        assert torch.allclose(self.emission_distr.sum(1), torch.tensor([1.]).to(self.emission_distr.get_device()))
        self.init_accumulators()

    def __str__(self):
        return str(self.emission_distr)


# do not replace replace with torch.distributions yet, it allows GPU computation
class BernoulliEmission(EmissionDistribution):

    def __init__(self, dim_target, dim_hidden_states):
        super().__init__()

        self.eps = 1e-8  # Laplace smoothing
        self.C = dim_hidden_states  # clusters

        self.bernoulli_params = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(self.C,
                                                                                      dtype=torch.float32)),
                                                   requires_grad=False)
        self.emission_numerator = torch.nn.Parameter(torch.empty_like(self.bernoulli_params),
                                                     requires_grad=False)
        self.emission_denominator = torch.nn.Parameter(torch.empty_like(self.bernoulli_params),
                                                       requires_grad=False)
        self.init_accumulators()

    def init_accumulators(self):
        torch.nn.init.constant_(self.emission_numerator, self.eps)
        torch.nn.init.constant_(self.emission_denominator, self.eps * 2)

    def bernoulli_density(self, labels, param):
        return torch.mul(torch.pow(param, labels),
                         torch.pow(1 - param, 1 - labels))

    def e_step(self, x_labels, y_labels):
        """
        For each cluster i, returns the probability associated with a specific input and output label.
        :param x_labels: unused
        :param y_labels: output observables
        :return: a tensor of size ?xC representing the estimated posterior distribution for the E-step
        """
        emission_of_labels = None
        for i in range(0, self.C):
            if emission_of_labels is None:
                emission_of_labels = torch.reshape(self.bernoulli_density(y_labels,
                                                                          self.bernoulli_params[i]), (-1, 1))
            else:
                emission_of_labels = torch.cat((emission_of_labels,
                                                torch.reshape(self.bernoulli_density(y_labels,
                                                                                     self.bernoulli_params[i]),
                                                              (-1, 1))),
                                               dim=1)
        return emission_of_labels

    def infer(self, p_Q, x_labels):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :param x_labels: unused
        :return:
        """
        # We simply compute P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        inferred_y = torch.mm(p_Q, self.bernoulli_params.unsqueeze(1))  # ?x1
        return inferred_y

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        """
        Updates the minibatch accumulators
        :param x_labels: unused
        :param y_labels: output observable
        :param posterior_estimate: a ?xC posterior estimate obtained using the output observables
        """
        if len(y_labels.shape) == 1:
            y_labels = y_labels.unsqueeze(1)
        self.emission_numerator += torch.sum(torch.mul(posterior_estimate,
                                                       y_labels), dim=0)  # --> 1 x C
        self.emission_denominator += torch.sum(posterior_estimate, dim=0)  # --> C

    def m_step(self):
        self.emission_distr = self.emission_numerator / self.emission_denominator
        self.init_accumulators()

    def __str__(self):
        return str(self.bernoulli_params)


class IndependentMultivariateBernoulliEmission(EmissionDistribution):

    def init_accumulators(self):
        for b in self.indep_bernoulli:
            b.init_accumulators()

    def __init__(self, dim_target, dim_hidden_states):
        super().__init__()

        self.eps = 1e-8  # Laplace smoothing
        self.indep_bernoulli = ModuleList([BernoulliEmission(dim_target, dim_hidden_states) for _ in range(dim_target)])
        self.init_accumulators()

    def e_step(self, x_labels, y_labels):
        """
        For each cluster i, returns the probability associated with a specific input and output label.
        :param x_labels: unused
        :param y_labels: output observables
        :return: a tensor of size ?xC representing the estimated posterior distribution for the E-step
        """
        emission_of_labels = None
        # Assume independence
        for i, b in enumerate(self.indep_bernoulli):
            est_post_dist = b.e_step(x_labels, y_labels[:,i].unsqueeze(1))
            if emission_of_labels is None:
                emission_of_labels = est_post_dist
            else:
                emission_of_labels *= est_post_dist
        return emission_of_labels

    def infer(self, p_Q, x_labels):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :param x_labels: unused
        :return:
        """
        inferred_y = None
        # Assume independence
        for i, b in enumerate(self.indep_bernoulli):
            inferred_yi = b.infer(p_Q, x_labels)
            if inferred_y is None:
                inferred_y = inferred_yi
            else:
                inferred_y = torch.cat((inferred_y, inferred_yi), dim=1)
        return inferred_y

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        """
        Updates the minibatch accumulators
        :param x_labels: unused
        :param y_labels: output observable
        :param posterior_estimate: a ?xC posterior estimate obtained using the output observables
        """
        # Assume independence
        for i, b in enumerate(self.indep_bernoulli):
            b._m_step(x_labels, y_labels[:,i].unsqueeze(1), posterior_estimate)

    def m_step(self):
        # Assume independence
        for i, b in enumerate(self.indep_bernoulli):
            b.m_step()
        self.init_accumulators()

    def __str__(self):
        return '-'.join([str(b) for b in self.indep_bernoulli])


# do not replace replace with torch.distributions yet, it allows GPU computation
class IsotropicGaussian(EmissionDistribution):

    def __init__(self, dim_features, dim_hidden_states, var_threshold=1e-16):
        super().__init__()

        self.eps = 1e-8  # Laplace smoothing
        self.var_threshold = var_threshold  # do not go below this value

        self.F = dim_features
        self.C = dim_hidden_states  # clusters

        self.mu = torch.nn.Parameter(torch.rand((self.C, self.F),
                                                dtype=torch.float32),
                                     requires_grad=False)
        self.var = torch.nn.Parameter(torch.rand((self.C, self.F),
                                                 dtype=torch.float32),
                                      requires_grad=False)
        self.pi = torch.nn.Parameter(torch.FloatTensor([math.pi]),
                                     requires_grad=False)

        self.mu_numerator = torch.nn.Parameter(torch.empty([self.C, self.F],
                                                           dtype=torch.float32),
                                               requires_grad=False)
        self.mu_denominator = torch.nn.Parameter(torch.empty([self.C, 1],
                                                             dtype=torch.float32),
                                                 requires_grad=False)
        self.var_numerator = torch.nn.Parameter(torch.empty([self.C, self.F],
                                                            dtype=torch.float32),
                                                requires_grad=False)
        self.var_denominator = torch.nn.Parameter(torch.empty([self.C, 1],
                                                              dtype=torch.float32),
                                                  requires_grad=False)

        # To launch k-means the first time
        self.initialized = False

        self.init_accumulators()

    def to(self, device):
        super().to(device)
        self.device = device

    def initialize(self, labels):
        codes, distortion = scipy.cluster.vq.kmeans(labels.cpu().detach().numpy()[:],
                                                    self.C, iter=20,
                                                    thresh=1e-05)
        # Number of prototypes can be < than self.C
        self.mu[:codes.shape[0], :] = torch.from_numpy(codes)
        self.var[:, :] = torch.std(labels, dim=0)

        self.mu = self.mu  # .to(self.device)
        self.var = self.var  # .to(self.device)

    def univariate_pdf(self, labels, mean, var):
        """
        Univariate case, computes probability distribution for each data point
        :param labels:
        :param mean:
        :param var:
        :return:
        """
        return torch.exp(-((labels.float() - mean) ** 2) / (2 * var)) / (torch.sqrt(2 * self.pi * var))

    def multivariate_diagonal_pdf(self, labels, mean, var):
        """
        Multivariate case, DIAGONAL cov. matrix. Computes probability distribution for each data point
        :param labels:
        :param mean:
        :param var:
        :return:
        """
        diff = (labels.float() - mean)

        log_normaliser = -0.5 * (torch.log(2 * self.pi) + torch.log(var))
        log_num = - (diff * diff) / (2 * var)
        log_probs = torch.sum(log_num + log_normaliser, dim=1)
        probs = torch.exp(log_probs)

        # Trick to avoid instability, in case variance collapses to 0
        probs[probs != probs] = self.eps
        probs[probs < self.eps] = self.eps

        return probs

    def init_accumulators(self):
        """
        This method initializes the accumulators for the EM algorithm.
        EM updates the parameters in batch, but needs to accumulate statistics in mini-batch style.
        :return:
        """
        torch.nn.init.constant_(self.mu_numerator, self.eps)
        torch.nn.init.constant_(self.mu_denominator, self.eps * self.C)
        torch.nn.init.constant_(self.var_numerator, self.eps)
        torch.nn.init.constant_(self.var_denominator, self.eps * self.C)

    def e_step(self, x_labels, y_labels):
        """
        For each cluster i, returns the probability associated to a specific label.
        :param x_labels: unused
        :param y_labels: output observables
        :return: a distribution associated to each layer
        """
        if not self.initialized:
            self.initialized = True
            self.initialize(y_labels)

        emission_of_labels = None
        for i in range(0, self.C):
            if emission_of_labels is None:
                emission_of_labels = torch.reshape(self.multivariate_diagonal_pdf(y_labels, self.mu[i], self.var[i]),
                                                   (-1, 1))
            else:
                emission_of_labels = torch.cat((emission_of_labels,
                                                torch.reshape(
                                                    self.multivariate_diagonal_pdf(y_labels, self.mu[i], self.var[i]),
                                                    (-1, 1))), dim=1)
        emission_of_labels += self.eps
        assert not torch.isnan(emission_of_labels).any(), (torch.sum(torch.isnan(emission_of_labels)))
        return emission_of_labels.detach()

    def infer(self, p_Q, x_labels):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :param x_labels: unused
        :return:
        """
        # We simply compute P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        inferred_y = torch.mm(p_Q, self.mu)  # ?xF
        return inferred_y

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        """
        Updates the minibatch accumulators
        :param x_labels: unused
        :param y_labels: output observable
        :param posterior_estimate: a ?xC posterior estimate obtained using the output observables
        """
        y_labels = y_labels.float()

        for i in range(0, self.C):
            reshaped_posterior = torch.reshape(posterior_estimate[:, i], (-1, 1))  # for broadcasting with F > 1

            den = torch.unsqueeze(torch.sum(posterior_estimate[:, i], dim=0), dim=-1)  # size C

            y_weighted = torch.mul(y_labels, reshaped_posterior)  # ?xF x ?x1 --> ?xF

            y_minus_mu_squared_tmp = y_labels - self.mu[i, :]
            # DIAGONAL COV MATRIX
            y_minus_mu_squared = torch.mul(y_minus_mu_squared_tmp, y_minus_mu_squared_tmp)

            self.mu_numerator[i, :] += torch.sum(y_weighted, dim=0)
            self.var_numerator[i] += torch.sum(torch.mul(y_minus_mu_squared, reshaped_posterior), dim=0)
            self.mu_denominator[i, :] += den
            self.var_denominator[i, :] += den

    def m_step(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.mu.data = self.mu_numerator / self.mu_denominator
        # Laplace smoothing
        self.var.data = (self.var_numerator + self.var_threshold) / (self.var_denominator + self.C * self.var_threshold)

        self.init_accumulators()

    def __str__(self):
        return f"{str(self.mu)}, {str(self.mu)}"
