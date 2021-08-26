import torch
import torch_geometric
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import global_mean_pool, global_add_pool

from pydgn.experiment.util import s2c
from pydgn.model.util.util import _compute_bigram, _compute_unigram, _make_one_hot
from pydgn.model.dgn.cgmm import CGMMTransition, BaseTransition
from copy import deepcopy


class ECGMM(torch.nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target,
                 predictor_class, config):
        super().__init__()
        self.device = None

        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.is_first_layer = config['depth'] == 1
        self.depth = config['depth']
        self.training = False
        self.compute_intermediate_outputs = False

        self.K = dim_node_features
        self.Y = dim_target
        self.L = len(config['prev_outputs_to_consider'])
        self.C = config['C']
        self.CA = config['CA']
        self.C2 = config['C'] + 1
        self.CS = config.get('CS', None)
        self.is_graph_classification = self.CS is not None
        #self.add_self_arc = config['self_arc'] if 'self_arc' in config else False
        self.use_continuous_states = config['infer_with_posterior']
        self.unibigram = config['unibigram']
        self.aggregation = config['aggregation']

        self.node_predictor_class = s2c(config['node_predictor'])
        self.edge_predictor_class = s2c(config['edge_predictor'])
        self.node_emission_class = config['node_emission']
        self.edge_emission_class = config['edge_emission']

        cfg_cpy = deepcopy(config)

        cfg_cpy['emission'] = self.node_emission_class
        self.node_readout = self.node_predictor_class(dim_node_features, 0,
                                       dim_target, cfg_cpy)

        cfg_cpy['emission'] = self.edge_emission_class
        cfg_cpy['C'] = self.CA
        self.edge_readout = self.edge_predictor_class(dim_edge_features, 0,
                                       dim_target, cfg_cpy)

        if self.is_first_layer:
            self.node_transition = BaseTransition(self.C)
            self.edge_transition = BaseTransition(self.CA)
        else:
            self.node_transition = CGMMTransition(self.C, self.CA,
                                               self.C2, self.L)

            # TODO refactor: passing 1 but acts as 2 (no real bottom state)
            self.edge_transition = CGMMTransition(self.CA, 1,
                                               self.C, self.L)

    def init_accumulators(self):
        self.node_readout.init_accumulators()
        self.edge_readout.init_accumulators()
        self.node_transition.init_accumulators()
        self.edge_transition.init_accumulators()

        # Do not delete this!
        if self.device:  # set by to() method
            self.to(self.device)

    def to(self, device):
        super().to(device)
        self.device = device

    def train(self):
        self.node_readout.train()
        self.node_transition.train()
        self.edge_readout.train()
        self.edge_transition.train()
        self.training = True

    def eval(self):
        self.node_readout.eval()
        self.node_transition.eval()
        self.edge_readout.eval()
        self.edge_transition.eval()
        self.training = False

    def forward(self, data):
        extra = None
        if not self.is_first_layer:
            data, extra = data[0], data[1]
        return self.e_step(data, extra)

    def e_step(self, data, extra=None):
        x, y, edge_attr, batch = data.x, data.y, data.edge_attr, data.batch
        edge_index, edge_batch = data.edge_index, batch[data.edge_index[0]]

        prev_node_stats = None if self.is_first_layer else extra.vo_outs
        prev_edge_stats = None if self.is_first_layer else extra.eo_outs

        if prev_node_stats is not None:
            prev_node_stats.to(self.device)
        if prev_edge_stats is not None:
            prev_edge_stats.to(self.device)

        # --------------------------- FORWARD PASS --------------------------- #

        #t = time.time()

        # --- TRANSITION contribution
        if self.is_first_layer:
            # p_Q_given_obs --> ?xC
            node_p_Q_given_obs  = self.node_transition.e_step(x)
            node_transition_posterior = node_p_Q_given_obs
            node_rightmost_term = node_p_Q_given_obs

            edge_p_Q_given_obs  = self.edge_transition.e_step(edge_attr)
            edge_transition_posterior = edge_p_Q_given_obs
            edge_rightmost_term = edge_p_Q_given_obs

        else:
            # p_Q_given_obs --> ?xC / transition_posterior --> ?xLxAxCxC2
            node_p_Q_given_obs, node_transition_posterior, node_rightmost_term = self.node_transition.e_step(x, prev_node_stats)
            edge_p_Q_given_obs, edge_transition_posterior, edge_rightmost_term = self.edge_transition.e_step(edge_attr, prev_edge_stats)

        assert torch.allclose(node_p_Q_given_obs.sum(1), torch.tensor([1.]).to(self.device)), node_p_Q_given_obs
        assert torch.allclose(edge_p_Q_given_obs.sum(1), torch.tensor([1.]).to(self.device)), edge_p_Q_given_obs

        #print(f"Transition E-step time: {time.time()-t}"); t = time.time()

        # --- READOUT contribution
        # true_log_likelihood --> ?x1 / readout_posterior --> ?xCSxCN or ?xCN
        node_true_log_likelihood, node_readout_posterior, node_emission_target = self.node_readout.e_step(node_p_Q_given_obs, x, y, batch)
        edge_true_log_likelihood, edge_readout_posterior, edge_emission_target = self.edge_readout.e_step(edge_p_Q_given_obs, edge_attr, y, edge_batch)
        true_log_likelihood = node_true_log_likelihood + edge_true_log_likelihood

        #print(f"Readout E-step time: {time.time()-t}"); t = time.time()

        # likely_labels --> ? x Y
        node_likely_labels = self.node_readout.infer(node_p_Q_given_obs, x, batch)
        edge_likely_labels = self.edge_readout.infer(edge_p_Q_given_obs, edge_attr, edge_batch)

        #print(f"Readout inference time: {time.time()-t}"); t = time.time()

        # -------------------------------------------------------------------- #

        if not self.is_graph_classification:
            node_complete_log_likelihood, node_eui = self._e_step_node(x, y, node_p_Q_given_obs,
                                        node_transition_posterior, node_rightmost_term,
                                        node_readout_posterior, node_emission_target,
                                        batch, process_nodes=True)

            edge_complete_log_likelihood, edge_eui = self._e_step_node(edge_attr, y, edge_p_Q_given_obs,
                                        edge_transition_posterior, edge_rightmost_term,
                                        edge_readout_posterior, edge_emission_target,
                                        edge_batch, process_nodes=False)

            complete_log_likelihood = node_complete_log_likelihood + edge_complete_log_likelihood
        else:
            assert False # TODO

        #print(f"Posterior E-step time: {time.time()-t}"); t = time.time()

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        # ECGMM uses the true posterior (over node attributes) as it is unsupervised!
        # Different from IO version
        if self.compute_intermediate_outputs:
            # print("Computing intermediate outputs")

            assert not self.training
            node_statistics_batch = self._compute_node_statistics(node_eui, edge_eui, data, self.device)
            edge_statistics_batch = self._compute_edge_statistics(node_eui, data, self.device)

            node_unigram = _compute_unigram(node_eui, self.use_continuous_states)
            edge_unigram = _compute_unigram(edge_eui, self.use_continuous_states)

            graph_node_unigram = self._get_aggregation_fun()(node_unigram, batch)
            graph_edge_unigram = self._get_aggregation_fun()(edge_unigram, edge_batch)
            graph_unigram = torch.cat((graph_node_unigram, graph_edge_unigram), dim=1)

            # NOTE: we do not compute edge bigrams for efficiency reasons
            edge_embeddings_batch = edge_unigram

            if self.unibigram:
                node_bigram = _compute_bigram(node_eui.float(), data.edge_index, batch,
                                              self.use_continuous_states)
                graph_bigram = self._get_aggregation_fun()(node_bigram, batch)
                node_embeddings_batch = torch.cat((node_unigram, node_bigram), dim=1)

                # graph unigram "contains" edge_unigrams, graph_bigram does "contain" only node_bigrams
                graph_embeddings_batch = torch.cat((graph_unigram, graph_bigram), dim=1)
            else:
                node_embeddings_batch = node_unigram
                graph_embeddings_batch = graph_unigram

            # to save time during debug
            embeddings = (None, None, graph_embeddings_batch, node_statistics_batch, edge_statistics_batch, None)
        else:
            embeddings = None

        return node_likely_labels, embeddings, complete_log_likelihood, \
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

            Z = global_add_pool(unnorm_posterior_estimate.sum((1,2), keepdim=True), batch)
            Z[Z==0.] = 1.

            esui = unnorm_posterior_estimate/(Z[batch])  # --> ?n x CS x CN
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
            exp_readout_posterior = b_readout_posterior.reshape((-1,self.CS,
                                                                 1,1,
                                                                 self.C, 1))
            # expand
            exp_transition_posterior = transition_posterior.unsqueeze(1)
            # batch graph sizes + expand
            b_graph_sizes = scatter_add(torch.ones_like(batch).to(self.device),
                                        batch)[batch].reshape([-1, 1, 1, 1, 1, 1])

            unnorm_posterior_estimate = torch.div(torch.mul(exp_readout_posterior,
                                                            exp_transition_posterior),
                                                  b_graph_sizes)
            Z = global_add_pool(unnorm_posterior_estimate.sum((1,2,3,4,5), keepdim=True), batch)
            Z[Z==0.] = 1.

            esuilaj = unnorm_posterior_estimate/(Z[batch])  # --> ?n x CS x L x A x C x C2
            euilaj = esuilaj.sum(1)  # Marginalize over CS --> transition M-step
            euila = euilaj.sum(4)  # ?n x L x A x C
            euil = euila.sum(2)  # ?n x L x C
            esui = esuilaj.sum((2,3,5)) # Marginalize over L,A,C2 --> readout M-step
            eui = euil.sum(1)  # ?n x C

            if self.training:
                # Update the accumulators (also useful for minibatch training)
                self.readout._m_step(x, y, esui, batch)
                self.transition._m_step(x, y, euilaj, euila, euil)

            # -------------------------------------------------------------------- #

            # ---------------------- Complete Log Likelihood --------------------- #

            complete_log_likelihood_readout = self.readout.complete_log_likelihood(esui, emission_target, batch)
            complete_log_likelihood_transition = self.transition.complete_log_likelihood(euilaj, euila, euil, rightmost_term)
            complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition

            # -------------------------------------------------------------------- #

        return complete_log_likelihood, eui

    def _e_step_node(self, x, y, p_Q_given_obs, transition_posterior,
                rightmost_term, readout_posterior, emission_target, batch, process_nodes):

        if self.is_first_layer:
            # ----------------------------- Posterior ---------------------------- #

            unnorm_posterior_estimate = readout_posterior*transition_posterior
            Z = unnorm_posterior_estimate.sum(1, keepdim=True)
            Z[Z==0.] = 1.

            eui = unnorm_posterior_estimate/Z  # --> ? x CN

            if self.training:
                if process_nodes:
                    # Update the accumulators (also useful for minibatch training)
                    self.node_readout._m_step(x, y, eui, batch)
                    self.node_transition._m_step(x, y, eui)
                else:
                    # Update the accumulators (also useful for minibatch training)
                    self.edge_readout._m_step(x, y, eui, batch)
                    self.edge_transition._m_step(x, y, eui)
            # -------------------------------------------------------------------- #

            # ---------------------- Complete Log Likelihood --------------------- #

            if process_nodes:
                complete_log_likelihood_readout = self.node_readout.complete_log_likelihood(eui, emission_target, batch)
                complete_log_likelihood_transition = self.node_transition.complete_log_likelihood(eui, p_Q_given_obs)
                complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition
            else:
                complete_log_likelihood_readout = self.edge_readout.complete_log_likelihood(eui, emission_target, batch)
                complete_log_likelihood_transition = self.edge_transition.complete_log_likelihood(eui, p_Q_given_obs)
                complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition
            # -------------------------------------------------------------------- #
        else:
            # ----------------------------- Posterior ---------------------------- #

            # expand
            if process_nodes:
                exp_readout_posterior = readout_posterior.reshape((-1,1,1,self.C,1))
            else:
                exp_readout_posterior = readout_posterior.reshape((-1,1,1,self.CA,1))

            unnorm_posterior_estimate = torch.mul(exp_readout_posterior,
                                                  transition_posterior)
            Z = unnorm_posterior_estimate.sum((1,2,3,4), keepdim=True)
            Z[Z==0.] = 1.
            euilaj = unnorm_posterior_estimate/Z  # --> ?n x L x A x C x C2
            euila = euilaj.sum(4)  # ?n x L x A x C
            euil = euila.sum(2)  # ?n x L x C
            eui = euil.sum(1)  # ?n x C

            if self.training:
                if process_nodes:
                    # Update the accumulators (also useful for minibatch training)
                    self.node_readout._m_step(x, y, eui, batch)
                    self.node_transition._m_step(x, y, euilaj, euila, euil)
                else:
                    # Update the accumulators (also useful for minibatch training)
                    self.edge_readout._m_step(x, y, eui, batch)
                    self.edge_transition._m_step(x, y, euilaj, euila, euil)

            # -------------------------------------------------------------------- #

            # ---------------------- Complete Log Likelihood --------------------- #
            if process_nodes:
                complete_log_likelihood_readout = self.node_readout.complete_log_likelihood(eui, emission_target, batch)
                complete_log_likelihood_transition = self.node_transition.complete_log_likelihood(euilaj, euila, euil, rightmost_term)
                complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition
            else:
                complete_log_likelihood_readout = self.edge_readout.complete_log_likelihood(eui, emission_target, batch)
                complete_log_likelihood_transition = self.edge_transition.complete_log_likelihood(euilaj, euila, euil, rightmost_term)
                complete_log_likelihood = complete_log_likelihood_readout + complete_log_likelihood_transition

            # -------------------------------------------------------------------- #

        assert torch.allclose(eui.sum(1), torch.tensor([1.]).to(self.device)), eui
        return complete_log_likelihood, eui

    def m_step(self):
        self.node_readout.m_step()
        self.node_transition.m_step()
        self.edge_readout.m_step()
        self.edge_transition.m_step()
        self.init_accumulators()

    def stopping_criterion(self, depth, max_layers, train_loss, train_score, val_loss, val_score,
                           dict_per_layer, layer_config, logger=None):
        return depth == max_layers

    def _compute_node_statistics(self, node_posteriors, edge_posteriors, data, device):

        if not self.use_continuous_states:
            node_posteriors = _make_one_hot(node_posteriors.argmax(dim=1), self.C).float()
            # TODO
            #edge_posteriors = _make_one_hot(edge_posteriors.argmax(dim=1), self.CA).float()

        statistics = torch.full((node_posteriors.shape[0], self.CA + 1, node_posteriors.shape[1]+1), 0., dtype=torch.float32).to(device)
        srcs, dsts = data.edge_index

        for arc_label in range(self.CA):
            sparse_label_adj_matr = torch.sparse_coo_tensor(data.edge_index, \
                                                            (edge_posteriors[:, arc_label]).to(device), \
                                                            torch.Size([node_posteriors.shape[0],
                                                                        node_posteriors.shape[0]])).to(device).transpose(0, 1)
            statistics[:, arc_label, :-1] = torch.sparse.mm(sparse_label_adj_matr, node_posteriors)

        # Deal with nodes with degree 0: add a single fake neighbor with uniform posterior
        degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()
        statistics[degrees==0., :, :] = 1./self.C2

        ## use self.A+1 as special edge for bottom states (all in self.C2-1)
        max_arieties, _ = self._compute_max_ariety(degrees.int().to(self.device), data.batch)
        max_arieties[max_arieties==0] = 1
        statistics[:, -1, -1] += degrees / max_arieties[data.batch].float()

        return statistics

    def _compute_edge_statistics(self, node_posteriors, data, device):

        if not self.use_continuous_states:
            node_posteriors = _make_one_hot(node_posteriors.argmax(dim=1), self.C).float()

        statistics = torch.full((data.edge_index.shape[1], 2, self.C), 0., dtype=torch.float32).to(device)
        srcs, dsts = data.edge_index

        srcs, dsts = data.edge_index
        statistics[:, 0, :] = node_posteriors[srcs]
        statistics[:, 1, :] = node_posteriors[dsts]
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
