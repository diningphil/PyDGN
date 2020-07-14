import math
import torch
import scipy
import scipy.cluster
import scipy.cluster.vq

import numpy as np


class GaussianEmission:
    """
    This class models the emission part of a Categorical Mixture model where the posterior is computed in
    an arbitrary way. It implements an interface suitable to be easily integrates into CGMM.
    NOTE: THE IMPLEMENTATION USES DIAGONAL COVARIANCE MATRICES TO SCALE LINEARLY WITH THE NUMBER OF FEATURES.
    """

    def __init__(self, f, c):
        self.F = f  # features dimension
        self.C = c  # clusters

        self.mu = torch.rand((self.C, self.F), dtype=torch.float32)
        # Sigma is diagonal! it holds the standard deviation terms, which have to be squared!
        self.var = torch.rand((self.C, self.F), dtype=torch.float32)  # at least var 1
        self.pi = torch.FloatTensor([math.pi])

        self.eps = 1e-8  # Laplace smoothing
        self.var_threshold = 1e-8

        # Initialize parameters
        self.mu_numerator = None
        self.mu_denominator = None
        self.var_numerator = None
        self.var_denominator = None

        self.device = None

        self.init_accumulators()
        self.initialized = False


    def to(self, device):
        self.mu.to(device)
        self.var.to(device)
        self.pi = self.pi.to(device)
        self.device = device

        self.mu_numerator = self.mu_numerator.to(device)
        self.mu_denominator = self.mu_denominator.to(device)

        self.var_numerator = self.var_numerator.to(device)
        self.var_denominator = self.var_denominator.to(device)


    def initialize(self, data):
        """
        :param data: design matrix (examples, features)
        :param K: number of gaussians
        :param var: initial variance
        """

        # choose k points from data to initialize means
        # m = data.size(0)
        # idxs = torch.from_numpy(np.random.choice(m, self.C, replace=False))
        # I do not want the gaussian to collapse on a single data point
        # self.mu = data[idxs]
        # '''

        print("Initializing mu with subset of samples...")
        codes, distortion = scipy.cluster.vq.kmeans(data.cpu().detach().numpy()[:], self.C, iter=20, thresh=1e-05)
        self.mu[:codes.shape[0], :] = torch.from_numpy(codes)
        self.var[:, :] = torch.std(data, dim=0)

        self.mu = self.mu.to(self.device)
        self.var = self.var.to(self.device)
        print("Done.")


    def export_parameters(self):
        return {'mu': self.mu, 'sigma': self.var}

    def import_parameters(self, params):
        self.mu = torch.from_numpy(params['mu'])
        self.var = torch.from_numpy(params['sigma'])

    def univariate_pdf(self, data, mean, var):
        """
        Univariate case, computes probability distribution for each data point
        :param data:
        :param mean:
        :param var:
        :return:
        """
        return torch.exp(-((data.float() - mean) ** 2) / (2 * var)) / (torch.sqrt(2 * self.pi * var))

    def multivariate_diagonal_pdf(self, data, mean, var):
        """
        Multivariate case, DIAGONAL cov. matrix. Computes probability distribution for each data point
        :param data:
        :param mean:
        :param var:
        :return:
        """
        diff = (data.float() - mean)

        '''
        first_log_term = - torch.log(torch.sqrt(2 * self.pi * var))
        second_log_term = - torch.mul(tmp, tmp)/(2*var)
        probs = torch.exp(torch.sum(first_log_term + second_log_term, dim=1))
        '''

        '''
        normaliser = torch.sqrt(2*self.pi*var).unsqueeze(0) # add batch dimension for broadcasting
        num = torch.exp(- (diff * diff)/(0.5*var) )
        probs = torch.prod(num / normaliser, dim=1)
        '''
        log_normaliser = -0.5 * (torch.log(2 * self.pi) + torch.log(var))
        log_num = - (diff * diff) / (2 * var)
        log_probs = torch.sum(log_num + log_normaliser, dim=1)
        probs = torch.exp(log_probs)

        '''
        assert torch.allclose(torch.exp(log_num), num), (num, torch.exp(log_num))
        assert torch.allclose(torch.exp(-log_normaliser), normaliser), (normaliser, torch.exp(-log_normaliser)) # minus because of the fraction
        assert torch.allclose(torch.exp(log_probs), probs), (probs, torch.exp(log_probs))
        '''

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
        self.mu_numerator = torch.full([self.C, self.F], self.eps, dtype=torch.float32)
        self.mu_denominator = torch.full([self.C, 1], self.eps * self.C, dtype=torch.float32)
        self.var_numerator = torch.full([self.C, self.F], self.eps, dtype=torch.float32)
        self.var_denominator = torch.full([self.C, 1], self.eps * self.C, dtype=torch.float32)

        if self.device is not None:
            self.to(self.device)

    def get_distribution_of_labels(self, labels):
        """
        For each cluster i, returns the probability associated to a specific label.
        :param labels:
        :return: a distribution associated to each layer
        """

        if not self.initialized:
            self.initialized = True
            self.initialize(labels)

        emission_of_labels = None
        for i in range(0, self.C):
            if emission_of_labels is None:
                emission_of_labels = torch.reshape(self.multivariate_diagonal_pdf(labels, self.mu[i], self.var[i]),
                                                   (-1, 1))
            else:
                emission_of_labels = torch.cat((emission_of_labels,
                                                torch.reshape(
                                                    self.multivariate_diagonal_pdf(labels, self.mu[i], self.var[i]),
                                                    (-1, 1))), dim=1)

        emission_of_labels += self.eps

        assert not torch.isnan(emission_of_labels).any(), (torch.sum(torch.isnan(emission_of_labels)))

        return emission_of_labels.detach()

    def infer(self, p_Q):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :return:
        """
        # We simply compute P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        inferred_y = torch.mm(p_Q, self.mu)  # ?xF
        return inferred_y

    def update_accumulators(self, posterior_estimate, labels):

        # labels = torch.squeeze(labels)  # removes dimensions of size 1 (current is ?x1)
        labels = labels.float()

        for i in range(0, self.C):
            reshaped_posterior = torch.reshape(posterior_estimate[:, i], (-1, 1))  # for broadcasting with F > 1

            den = torch.unsqueeze(torch.sum(posterior_estimate[:, i], dim=0), dim=-1)  # size C

            y_weighted = torch.mul(labels, reshaped_posterior)  # ?xF x ?x1 --> ?xF

            y_minus_mu_squared_tmp = labels - self.mu[i, :]
            # DIAGONAL COV MATRIX
            y_minus_mu_squared = torch.mul(y_minus_mu_squared_tmp, y_minus_mu_squared_tmp)

            self.mu_numerator[i, :] += torch.sum(y_weighted, dim=0)
            self.var_numerator[i] += torch.sum(torch.mul(y_minus_mu_squared, reshaped_posterior), dim=0)

            self.mu_denominator[i, :] += den
            self.var_denominator[i, :] += den

    def update_parameters(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.mu = self.mu_numerator / self.mu_denominator

        # Laplace smoothing
        self.var = (self.var_numerator + self.var_threshold) / (self.var_denominator + self.C*self.var_threshold)

        # print(f'Mean values are: {self.mu}')
        # print(f'Std values are: {torch.sqrt(self.var)}')
