import torch
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_add, scatter_max

from pydgn.model.util.util import _compute_bigram, _compute_unigram


class CGMM(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target,
                 predictor_class, config):
        super().__init__()
        self.device = None

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target
        self.predictor_class = predictor_class
        self.is_first_layer = config['depth'] == 1
        self.depth = config['depth']
        self.training = False
        self.compute_intermediate_outputs = False

        self.K = dim_node_features
        self.Y = dim_target
        self.L = len(config['prev_outputs_to_consider'])
        self.A = config['A']
        self.C = config['C']
        self.C2 = config['C'] + 1
        self.CS = config.get('CS', None)
        self.is_graph_classification = self.CS is not None
        # self.add_self_arc = config['self_arc'] if 'self_arc' in config else False
        self.use_continuous_states = config['infer_with_posterior']
        self.unibigram = config['unibigram']
        self.aggregation = config['aggregation']

        self.readout = predictor_class(dim_node_features, dim_edge_features,
                                       dim_target, config)

        if self.is_first_layer:
            self.transition = BaseTransition(self.C)
        else:
            self.transition = CGMMTransition(self.C, self.A,
                                             self.C2, self.L)

        self.init_accumulators()

    def init_accumulators(self):
        self.readout.init_accumulators()
        self.transition.init_accumulators()

        # Do not delete this!
        if self.device:  # set by to() method
            self.to(self.device)

    def to(self, device):
        super().to(device)
        self.device = device

    def train(self):
        self.readout.train()
        self.transition.train()
        self.training = True

    def eval(self):
        self.readout.eval()
        self.transition.eval()
        self.training = False

    def forward(self, data):
        extra = None
        if not self.is_first_layer:
            data, extra = data[0], data[1]
        return self.e_step(data, extra)

    def e_step(self, data, extra=None):
        x, y, batch = data.x, data.y, data.batch

        prev_stats = None if self.is_first_layer else extra.vo_outs
        if prev_stats is not None:
            prev_stats.to(self.device)

        # --------------------------- FORWARD PASS --------------------------- #

        # t = time.time()

        # --- TRANSITION contribution
        if self.is_first_layer:
            # p_Q_given_obs --> ?xC
            p_Q_given_obs = self.transition.e_step(x)
            transition_posterior = p_Q_given_obs
            rightmost_term = p_Q_given_obs
        else:
            # p_Q_given_obs --> ?xC / transition_posterior --> ?xLxAxCxC2
            p_Q_given_obs, transition_posterior, rightmost_term = self.transition.e_step(x, prev_stats)

        # assert torch.allclose(p_Q_given_obs.sum(1), torch.tensor([1.]).to(self.device)), p_Q_given_obs.sum(1)

        # print(f"Transition E-step time: {time.time()-t}"); t = time.time()

        # --- READOUT contribution
        # true_log_likelihood --> ?x1 / readout_posterior --> ?xCSxCN or ?xCN
        true_log_likelihood, readout_posterior, emission_target = self.readout.e_step(p_Q_given_obs, x, y, batch)

        # print(f"Readout E-step time: {time.time()-t}"); t = time.time()

        # likely_labels --> ? x Y
        likely_labels = self.readout.infer(p_Q_given_obs, x, batch)

        # print(f"Readout inference time: {time.time()-t}"); t = time.time()

        # -------------------------------------------------------------------- #

        if not self.is_graph_classification:
            complete_log_likelihood, eui = self._e_step_node(x, y, p_Q_given_obs,
                                                             transition_posterior, rightmost_term,
                                                             readout_posterior, emission_target,
                                                             batch)
        else:
            complete_log_likelihood, eui = self._e_step_graph(x, y, p_Q_given_obs,
                                                              transition_posterior, rightmost_term,
                                                              readout_posterior, emission_target,
                                                              batch)

        # print(f"Posterior E-step time: {time.time()-t}"); t = time.time()

        num_nodes = x.shape[0]

        # CGMM uses the true posterior (over node attributes) as it is unsupervised!
        # Different from IO version
        if self.compute_intermediate_outputs:
            # print("Computing intermediate outputs")

            assert not self.training
            statistics_batch = self._compute_statistics(eui, data, self.device)

            node_unigram = _compute_unigram(eui, self.use_continuous_states)
            graph_unigram = self._get_aggregation_fun()(node_unigram, batch)

            if self.unibigram:
                node_bigram = _compute_bigram(eui.float(), data.edge_index, batch,
                                              self.use_continuous_states)
                graph_bigram = self._get_aggregation_fun()(node_bigram, batch)

                node_embeddings_batch = torch.cat((node_unigram, node_bigram), dim=1)
                graph_embeddings_batch = torch.cat((graph_unigram, graph_bigram), dim=1)
            else:
                node_embeddings_batch = node_unigram
                graph_embeddings_batch = graph_unigram

            # to save time during debug
            embeddings = (None, None, graph_embeddings_batch, statistics_batch, None, None)
        else:
            embeddings = None

        return likely_labels, embeddings, complete_log_likelihood, \
               true_log_likelihood, num_nodes

    def _e_step_graph(self, x, y, p_Q_given_obs, transition_posterior,
                      rightmost_term, readout_posterior, emission_target, batch):

        # batch (i.e., replicate) graph readout posterior for all nodes
        b_readout_posterior = readout_posterior[batch]  # ?nxCSxCN

        if self.is_first_layer:
            # ----------------------------- Posterior ---------------------------- #

            # expand
            exp_readout_posterior = b_readout_posterior.reshape((-1, self.CS,
                                                                 self.C))
            # expand
            exp_transition_posterior = transition_posterior.unsqueeze(1)

            # batch graph sizes + expand
            b_graph_sizes = scatter_add(torch.ones_like(batch).to(self.device),
                                        batch)[batch].reshape([-1, 1, 1])

            unnorm_posterior_estimate = torch.div(torch.mul(exp_readout_posterior,
                                                            exp_transition_posterior),
                                                  b_graph_sizes)

            Z = global_add_pool(unnorm_posterior_estimate.sum((1, 2), keepdim=True), batch)
            Z[Z == 0.] = 1.

            esui = unnorm_posterior_estimate / (Z[batch])  # --> ?n x CS x CN
            eui = esui.sum(1)  # ?n x CN

            if self.training:
                # Update the accumulators (also useful for minibatch training)
                self.readout._m_step(x, y, esui, batch)
                self.transition._m_step(x, y, eui)

            # -------------------------------------------------------------------- #

            # ---------------------- Complete Log Likelihood --------------------- #

            complete_log_likelihood_readout = self.readout.complete_log_likelihood(esui, emission_target, batch)
            complete_log_likelihood_transition = self.transition.complete_log_likelihood(eui, p_Q_given_obs)
            complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition

            # -------------------------------------------------------------------- #
        else:
            # ----------------------------- Posterior ---------------------------- #

            # expand
            exp_readout_posterior = b_readout_posterior.reshape((-1, self.CS,
                                                                 1, 1,
                                                                 self.C, 1))
            # expand
            exp_transition_posterior = transition_posterior.unsqueeze(1)
            # batch graph sizes + expand
            b_graph_sizes = scatter_add(torch.ones_like(batch).to(self.device),
                                        batch)[batch].reshape([-1, 1, 1, 1, 1, 1])

            unnorm_posterior_estimate = torch.div(torch.mul(exp_readout_posterior,
                                                            exp_transition_posterior),
                                                  b_graph_sizes)
            Z = global_add_pool(unnorm_posterior_estimate.sum((1, 2, 3, 4, 5), keepdim=True), batch)
            Z[Z == 0.] = 1.

            esuilaj = unnorm_posterior_estimate / (Z[batch])  # --> ?n x CS x L x A x C x C2
            euilaj = esuilaj.sum(1)  # Marginalize over CS --> transition M-step
            euila = euilaj.sum(4)  # ?n x L x A x C
            euil = euila.sum(2)  # ?n x L x C
            esui = esuilaj.sum((2, 3, 5))  # Marginalize over L,A,C2 --> readout M-step
            eui = euil.sum(1)  # ?n x C

            if self.training:
                # Update the accumulators (also useful for minibatch training)
                self.readout._m_step(x, y, esui, batch)
                self.transition._m_step(x, y, euilaj, euila, euil)

            # -------------------------------------------------------------------- #

            # ---------------------- Complete Log Likelihood --------------------- #

            complete_log_likelihood_readout = self.readout.complete_log_likelihood(esui, emission_target, batch)
            complete_log_likelihood_transition = self.transition.complete_log_likelihood(euilaj, euila, euil,
                                                                                         rightmost_term)
            complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition

            # -------------------------------------------------------------------- #

        return complete_log_likelihood, eui

    def _e_step_node(self, x, y, p_Q_given_obs, transition_posterior,
                     rightmost_term, readout_posterior, emission_target, batch):

        if self.is_first_layer:
            # ----------------------------- Posterior ---------------------------- #

            unnorm_posterior_estimate = readout_posterior * transition_posterior
            Z = unnorm_posterior_estimate.sum(1, keepdim=True)
            Z[Z == 0.] = 1.

            eui = unnorm_posterior_estimate / Z  # --> ? x CN

            if self.training:
                # Update the accumulators (also useful for minibatch training)
                self.readout._m_step(x, y, eui, batch)
                self.transition._m_step(x, y, eui)

            # -------------------------------------------------------------------- #

            # ---------------------- Complete Log Likelihood --------------------- #

            complete_log_likelihood_readout = self.readout.complete_log_likelihood(eui, emission_target, batch)
            complete_log_likelihood_transition = self.transition.complete_log_likelihood(eui, p_Q_given_obs)
            complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition

            # -------------------------------------------------------------------- #
        else:
            # ----------------------------- Posterior ---------------------------- #

            # expand
            exp_readout_posterior = readout_posterior.reshape((-1, 1, 1, self.C, 1))

            unnorm_posterior_estimate = torch.mul(exp_readout_posterior,
                                                  transition_posterior)
            Z = unnorm_posterior_estimate.sum((1, 2, 3, 4), keepdim=True)
            Z[Z == 0.] = 1.
            euilaj = unnorm_posterior_estimate / Z  # --> ?n x L x A x C x C2
            euila = euilaj.sum(4)  # ?n x L x A x C
            euil = euila.sum(2)  # ?n x L x C
            eui = euil.sum(1)  # ?n x C

            if self.training:
                # Update the accumulators (also useful for minibatch training)
                self.readout._m_step(x, y, eui, batch)
                self.transition._m_step(x, y, euilaj, euila, euil)

            # -------------------------------------------------------------------- #

            # ---------------------- Complete Log Likelihood --------------------- #

            complete_log_likelihood_readout = self.readout.complete_log_likelihood(eui, emission_target, batch)
            complete_log_likelihood_transition = self.transition.complete_log_likelihood(euilaj, euila, euil,
                                                                                         rightmost_term)
            complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition
            # -------------------------------------------------------------------- #

        # assert torch.allclose(eui.sum(1), torch.tensor([1.]).to(self.device)), eui.sum(1)[eui.sum(1) != 1.]

        return complete_log_likelihood, eui

    def m_step(self):
        self.readout.m_step()
        self.transition.m_step()
        self.init_accumulators()

    def stopping_criterion(self, depth, max_layers, train_loss, train_score, val_loss, val_score,
                           dict_per_layer, layer_config, logger=None):
        return depth == max_layers

    def _compute_statistics(self, posteriors, data, device):

        statistics = torch.full((posteriors.shape[0], self.A + 1, posteriors.shape[1] + 1), 0., dtype=torch.float32).to(
            device)
        srcs, dsts = data.edge_index

        if self.A == 1:
            sparse_adj_matr = torch.sparse_coo_tensor(data.edge_index, \
                                                      torch.ones(data.edge_index.shape[1], dtype=posteriors.dtype).to(
                                                          device), \
                                                      torch.Size([posteriors.shape[0],
                                                                  posteriors.shape[0]])).to(device).transpose(0, 1)
            statistics[:, 0, :-1] = torch.sparse.mm(sparse_adj_matr, posteriors)
        else:
            for arc_label in range(self.A):
                sparse_label_adj_matr = torch.sparse_coo_tensor(data.edge_index, \
                                                                (data.edge_attr == arc_label).to(device).float(), \
                                                                torch.Size([posteriors.shape[0],
                                                                            posteriors.shape[0]])).to(device).transpose(
                    0, 1)
                statistics[:, arc_label, :-1] = torch.sparse.mm(sparse_label_adj_matr, posteriors)

        # Deal with nodes with degree 0: add a single fake neighbor with uniform posterior
        degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()
        statistics[degrees == 0., :, :] = 1. / self.C2

        '''
        if self.add_self_arc:
            statistics[:, self.A, :-1] += posteriors
        '''
        # use self.A+1 as special edge for bottom states (all in self.C2-1)
        max_arieties, _ = self._compute_max_ariety(degrees.int().to(self.device), data.batch)
        max_arieties[max_arieties == 0] = 1
        statistics[:, self.A, self.C] += degrees / max_arieties[data.batch].float()

        return statistics

    def _compute_sizes(self, batch, device):
        return scatter_add(torch.ones(len(batch), dtype=torch.int).to(device), batch)

    def _compute_max_ariety(self, degrees, batch):
        return scatter_max(degrees, batch)

    def _get_aggregation_fun(self):
        if self.aggregation == 'mean':
            aggregate = global_mean_pool
        elif self.aggregation == 'sum':
            aggregate = global_add_pool
        return aggregate


class CGMMTransition(torch.nn.Module):
    def __init__(self, c, a, c2, l):
        super().__init__()

        self.device = None

        self.eps = 1e-8  # Laplace smoothing
        self.C = c
        self.orig_A = a
        self.A = a + 1  # bottom state connected with a special arc
        self.C2 = c2
        self.L = l

        self.layerS = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float32)),
                                         requires_grad=False)
        self.arcS = torch.nn.Parameter(torch.zeros((self.L, self.A), dtype=torch.float32), requires_grad=False)
        self.transition = torch.nn.Parameter(torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float32),
                                             requires_grad=False)

        self.layerS /= self.layerS.sum()  # inplace

        for layer in range(self.L):
            self.arcS[layer, :] = torch.nn.init.uniform_(self.arcS[layer, :])
            self.arcS[layer, :] /= self.arcS[layer, :].sum()
            for arc in range(self.A):
                for j in range(self.C2):
                    tr = torch.nn.init.uniform_(torch.empty(self.C))
                    self.transition[layer, arc, :, j] = tr / tr.sum()

        # These are variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.layerS_numerator = torch.nn.Parameter(torch.empty_like(self.layerS),
                                                   requires_grad=False)
        self.arcS_numerator = torch.nn.Parameter(torch.empty_like(self.arcS),
                                                 requires_grad=False)
        self.transition_numerator = torch.nn.Parameter(torch.empty_like(self.transition),
                                                       requires_grad=False)
        self.init_accumulators()

    def to(self, device):
        super().to(device)
        self.device = device

    def init_accumulators(self):
        torch.nn.init.constant_(self.layerS_numerator, self.eps)
        torch.nn.init.constant_(self.arcS_numerator, self.eps)
        torch.nn.init.constant_(self.transition_numerator, self.eps)

    def e_step(self, x_labels, stats=None, batch=None):

        # ---------------------------- Forward Pass -------------------------- #

        stats = stats.float()

        # Compute the neighbourhood dimension for each vertex
        neighbDim = stats.sum(dim=3, keepdim=True).unsqueeze(4)  # --> ?n x L x A x 1
        # Replace zeros with ones to avoid divisions by zero
        # This does not alter learning: the numerator can still be zero
        neighbDim[neighbDim == 0] = 1.

        transition = torch.unsqueeze(self.transition, 0)  # --> 1 x L x A x C x C2

        stats = stats.unsqueeze(3)  # --> ?n x L x A x 1 x C2
        rightmost_term = (transition * stats) / neighbDim  # --> ?n x L x A x C x C2
        layerS = torch.reshape(self.layerS, [1, self.L, 1])  # --> L x 1 x 1
        arcS = torch.reshape(self.arcS, [1, self.L, self.A, 1])  # --> 1 x L x A x 1

        tmp = (arcS * rightmost_term.sum(4)).sum(dim=2)  # --> ?n x L x C
        p_Q_given_obs = (layerS * tmp).sum(dim=1)  # --> ?n x C

        # -------------------------------------------------------------------- #

        # ----------------------------- Posterior ---------------------------- #

        layerS_expanded = torch.reshape(self.layerS, [1, self.L, 1, 1, 1])
        arcS_expanded = torch.reshape(self.arcS, [1, self.L, self.A, 1, 1])
        transition_posterior = layerS_expanded * arcS_expanded * rightmost_term

        # -------------------------------------------------------------------- #

        return p_Q_given_obs, transition_posterior, rightmost_term

    def complete_log_likelihood(self, euilaj, euila, euil, rightmost_term):
        layerS = torch.reshape(self.layerS, [1, self.L, 1])
        term_1 = (euil * (layerS.log())).sum((1, 2)).sum()

        arcS = torch.reshape(self.arcS, [1, self.L, self.A, 1])
        term_2 = (euila * (arcS.log())).sum((1, 2, 3)).sum()

        rightmost_term[rightmost_term == 0.] = 1
        term_3 = (euilaj * (rightmost_term.log())).sum((1, 2, 3, 4)).sum()

        return term_1 + term_2 + term_3

    def _m_step(self, x_labels, y_labels, euilaj, euila, euil):

        self.layerS_numerator += euil.sum(dim=(0, 2))
        self.arcS_numerator += euila.sum(dim=(0, 3))
        self.transition_numerator += euilaj.sum(0)  # --> L x A x C x C2

    def m_step(self):
        self.layerS.data = self.layerS_numerator / self.layerS_numerator.sum(dim=0, keepdim=True)
        self.arcS.data = self.arcS_numerator / self.arcS_numerator.sum(dim=1, keepdim=True)
        self.transition.data = self.transition_numerator / self.transition_numerator.sum(dim=2, keepdim=True)
        self.init_accumulators()


class BaseTransition(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.device = None

        self.eps = 1e-8  # Laplace smoothing
        self.C = c

        self.transition = torch.nn.Parameter(torch.empty([self.C], dtype=torch.float32), requires_grad=False)

        tr = torch.nn.init.uniform_(torch.empty(self.C))
        self.transition.data = tr / tr.sum()

        self.transition_numerator = torch.nn.Parameter(torch.empty_like(self.transition), requires_grad=False)
        self.init_accumulators()

    def to(self, device):
        super().to(device)
        self.device = device

    def init_accumulators(self):
        torch.nn.init.constant_(self.transition_numerator, self.eps)

    def e_step(self, x_labels, stats=None, batch=None):
        # ---------------------------- Forward Pass -------------------------- #

        p_Q_given_obs = self.transition.unsqueeze(0)  # --> 1 x C
        return p_Q_given_obs

    def infer(self, x_labels, stats=None, batch=None):
        p_Q_given_obs, _ = self.e_step(x_labels, stats, batch)
        return p_Q_given_obs

    def complete_log_likelihood(self, eui, p_Q_given_obs):
        complete_log_likelihood = (eui * (p_Q_given_obs.log())).sum(1).sum()
        return complete_log_likelihood

    def _m_step(self, x_labels, y_labels, eui):
        self.transition_numerator += eui.sum(0)

    def m_step(self):
        self.transition.data = self.transition_numerator / self.transition_numerator.sum()
        self.init_accumulators()
