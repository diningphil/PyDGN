import torch
from .gaussian_emission import GaussianEmission
from .categorical_emission import CategoricalEmission
from .bernoulli_emission import BernoulliEmission


class MixedEmission:

    def __init__(self, k, c):
        self.C = c
        # k list of int, 1 if the attribute is continuous, 2 if Bernoulli, dim output alphabet for categorical
        self.K = k

        # discrete attributes are one-hot encoded!
        self.bernoulli_idxs = []  # list of list
        self.categorical_idxs = [] # list of list
        self.gauss_idxs = [] # list of int
        
        curr_idx = 0
        for dim in self.K: 
            if dim == 1:
                self.gauss_idxs.append(curr_idx)
                curr_idx += 1
            elif dim == 2:
                self.bernoulli_idxs.append([curr_idx])  
                curr_idx += 1
            else:
                self.categorical_idxs.append([range(curr_idx, curr_idx + dim)])
                curr_idx += dim

        self.emissions = []

        if len(self.bernoulli_idxs) > 0:
            self.emissions.extend([BernoulliEmission(self.C) for _ in range(len(self.bernoulli_idxs))])
        if len(self.categorical_idxs) > 0:
            self.emissions.extend([CategoricalEmission(len(feats), self.C) for feats in self.categorical_idxs])
        if len(self.gauss_idxs) > 0:
            self.emissions.append(GaussianEmission(len(self.gauss_idxs), self.C))
        
    def to(self, device):
        for emission in self.emissions:
            emission.to(device)

    def init_accumulators(self):
        for emission in self.emissions:
            emission.init_accumulators()

    def get_distribution_of_labels(self, labels):

        emission_for_labels = 1.

        if len(self.gauss_idxs) > 0:
            # Deal with continuous attributes        
            cont_attr = labels[:, self.gauss_idxs]  # should be a list of int, preserves dimensions
            emission_for_labels = self.emissions[-1].get_distribution_of_labels(cont_attr)

        if len(self.bernoulli_idxs) > 0:
            # Deal with binary attributes        
            for i, bernoulli_emission in enumerate(self.emissions[:len(self.bernoulli_idxs)]):
                attr_idxs = self.bernoulli_idxs[i] # should be a list with a single value, preserves dimensions
                discr_attr = labels[:, attr_idxs]
                emission_for_labels = emission_for_labels * bernoulli_emission.get_distribution_of_labels(discr_attr)

        if len(self.categorical_idxs) > 0:
            # Deal with categorical attributes        
            for i, categorical_emission in enumerate(self.emissions[len(self.bernoulli_idxs):-1]):
                attr_idxs = self.categorical_idxs[i]  # should be a list of values
                discr_attr = labels[:, attr_idxs]
                emission_for_labels = emission_for_labels * categorical_emission.get_distribution_of_labels(discr_attr)

        return emission_for_labels

    def update_accumulators(self, posterior_estimate, labels):

        if len(self.gauss_idxs) > 0:
            # Deal with continuous attributes        
            cont_attr = labels[:, self.gauss_idxs]  # should be a list of int, preserves dimensions
            self.emissions[-1].update_accumulators(posterior_estimate, cont_attr)

        if len(self.bernoulli_idxs) > 0:
            # Deal with binary attributes        
            for i, bernoulli_emission in enumerate(self.emissions[:len(self.bernoulli_idxs)]):
                attr_idxs = self.bernoulli_idxs[i] # should be a list with a single value, preserves dimensions
                discr_attr = labels[:, attr_idxs]
                bernoulli_emission.update_accumulators(posterior_estimate, discr_attr)

        if len(self.categorical_idxs) > 0:
            # Deal with categorical attributes        
            for i, categorical_emission in enumerate(self.emissions[len(self.bernoulli_idxs):len(self.bernoulli_idxs) + len(self.categorical_idxs)]):
                attr_idxs = self.categorical_idxs[i]  # should be a list of values
                discr_attr = labels[:, attr_idxs]
                categorical_emission.update_accumulators(posterior_estimate, discr_attr)

    def update_parameters(self):
        for emission in self.emissions:
            emission.update_parameters()