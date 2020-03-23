import torch

from config.utils import s2c
from models.utils.CategoricalEmission import CategoricalEmission
from models.utils.GaussianEmission import GaussianEmission
from torch_scatter import scatter_max
from torch_geometric.nn import global_mean_pool, global_add_pool


from training.core.engine import TrainingEngine

# Contextual Graph Markov Model, ICML 2018
class CGMM:

    def __init__(self, dim_features, dim_target, predictor_class, config):
        """
        CGMM
        :param k: dimension of a vertex output's alphabet, which goes from 0 to K-1 (when discrete)
        :param a: dimension of an edge output's alphabet, which goes from 0 to A-1
        :param cn: vertexes latent space dimension
        :param ca: edges latent space dimension
        :param l: number of previous layers to consider. You must pass the appropriate number of statistics at training
        """
        self.dim_features = dim_features
        self.dim_target = dim_target
        self.predictor_class = predictor_class
        self.layer_config = config
        self.depth = config['depth']
        self.is_first_layer = self.depth == 1
        self.training = False
        self.compute_intermediate_outputs = False

        print(f'Initializing layer {self.depth}')
        self.K = dim_features

        self.node_type = config['node_type']
        self.L = len(config['prev_outputs_to_consider'])
        self.A = config['A']
        self.C = config['C']
        self.C2 = config['C'] + 1
        self.add_self_arc = config['self_arc'] if 'self_arc' in config else False

        self.max_epochs = config['max_epochs'] if 'max_epochs' in config else 10
        self.node_type = config['node_type']
        self.threshold = config['threshold'] if 'threshold' in config else 0.
        self.use_continuous_states = config['infer_with_posterior']
        self.unibigram = config['unibigram']
        self.aggregation = config['aggregation']
        self.device = None

        self.layer = CGMMLayer(self.K, self.C, self.A, self.C2, self.L, self.is_first_layer, self.node_type)

    def to(self, device):
        self.device = device
        self.layer.emission.to(device)
        if not self.layer.is_layer_0:
            self.layer.layerS.to(device)
            self.layer.arcS.to(device)
            self.layer.transition.to(device)

        return self

    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, data):
        return self.E_step(data, self.training)

    def E_step(self, data, training=False):
        graph_embeddings_batch, statistics_batch = None, None

        if isinstance(data, list):
            data, extra = data[0], data[1]
        else:
            extra = None

        prev_stats = extra.vo_outs if extra is not None else None
        if prev_stats is not None:
            prev_stats.to(self.device)

        # E-step also accumulates statistics for the M-step!
        likelihood, posterior_batch = self.layer.E_step(data.x, prev_stats, training=training)
        likelihood, posterior_batch = likelihood.detach(), posterior_batch.detach()

        if self.compute_intermediate_outputs:
            # print("Computing intermediate outputs")
            statistics_batch = self._compute_statistics(posterior_batch, data, self.device)

            node_unigram, graph_unigram = self._compute_unigram(posterior_batch, data.batch, self.device)

            if self.unibigram:
                node_bigram, graph_bigram = self._compute_bigram(posterior_batch.double(), data.edge_index, data.batch,
                                                                 graph_unigram.shape[0], self.device)

                node_embeddings_batch = torch.cat((node_unigram, node_bigram), dim=1)
                graph_embeddings_batch = torch.cat((graph_unigram, graph_bigram), dim=1)
            else:
                node_embeddings_batch = node_unigram
                graph_embeddings_batch = graph_unigram


            return likelihood, (node_embeddings_batch, None, graph_embeddings_batch, statistics_batch, None, None)

        return likelihood, None

    def M_step(self):
        self.layer.M_step()

    def arbitrary_logic(self, experiment, train_loader, layer_config, is_last_layer, validation_loader=None, test_loader=None,
                        logger=None, device='cpu'):
        if is_last_layer:
            dim_features = self.C*self.depth if not self.unibigram else self.C*self.C2*self.depth

            config = layer_config['arbitrary_function_config']

            predictor_class = s2c(config['predictor'])
            model = predictor_class(dim_features, self.dim_target, config)

            loss_class, loss_args = experiment._return_class_and_args(config, 'loss')
            loss = loss_class(**loss_args) if loss_class is not None else None

            scorer_class, scorer_args = experiment._return_class_and_args(config, 'scorer')
            scorer = scorer_class(**scorer_args) if scorer_class is not None else None

            optim_class, optim_args = experiment._return_class_and_args(config, 'optimizer')
            optimizer = optim_class(model=model, **optim_args) if optim_class is not None else None

            sched_class, sched_args = experiment._return_class_and_args(config, 'scheduler')
            scheduler = sched_class(**sched_args) if sched_class is not None else None

            grad_clip_class, grad_clip_args = experiment._return_class_and_args(config, 'gradient_clipping')
            grad_clipper = grad_clip_class(**grad_clip_args) if grad_clip_class is not None else None

            early_stop_class, early_stop_args = experiment._return_class_and_args(config, 'early_stopper')
            early_stopper = early_stop_class(**early_stop_args) if early_stop_class is not None else None

            wrapper = s2c(config['wrapper'])(model=model, loss=loss,
                                             optimizer=optimizer, scorer=scorer, scheduler=scheduler,
                                             early_stopper=early_stopper, gradient_clipping=grad_clipper,
                                             device=device)

            train_loss, train_score, _, \
            val_loss, val_score, _, \
            test_loss, test_score, _ = wrapper.train(train_loader=train_loader,
                                                        validation_loader=validation_loader,
                                                        test_loader=test_loader,
                                                        max_epochs=config['epochs'],
                                                        logger=logger)

            return {'train_score': train_score, 'validation_score': val_score, 'test_score': test_score}
        else:
            return {}

    def stopping_criterion(self, depth, max_layers, train_loss, train_score, val_loss, val_score,
                           dict_per_layer, layer_config, logger=None):
        return depth == max_layers

    def _compute_statistics(self, posteriors, data, device):

        # Compute statistics
        if 'cuda' in device:
            statistics = torch.full((posteriors.shape[0], self.A + 2, self.C2), 1e-8, dtype=torch.float64).cuda()
        else:
            statistics = torch.full((posteriors.shape[0], self.A + 2, self.C2), 1e-8, dtype=torch.float64)

        srcs, dsts = data.edge_index

        if self.A == 1:
            for source, dest, in zip(srcs, dsts):
                statistics[dest, 0, :-1] += posteriors[source]
        else:
            arc_labels = data.edge_attr

            for source, dest, arc_label in zip(srcs, dsts, arc_labels):
                statistics[dest, arc_label, :-1] += posteriors[source]

        if self.add_self_arc:
            statistics[:, self.A, :-1] += posteriors

        # use self.A+1 as special edge for bottom states (all in self.C2-1)
        degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()

        max_arieties, _ = self._compute_max_ariety(degrees.int(), data.batch)
        statistics[:, self.A + 1, self.C2 - 1] += 1 - (degrees / max_arieties[data.batch].double())
        return statistics

    def _compute_unigram(self, posteriors, batch, device):

        aggregate = self._get_aggregation_fun()

        if self.use_continuous_states:
            node_embeddings_batch = posteriors
            graph_embeddings_batch = aggregate(posteriors, batch)
        else:
            if 'cuda' in device:
                node_embeddings_batch = self._make_one_hot(posteriors.argmax(dim=1)).cuda()
            else:
                node_embeddings_batch = self._make_one_hot(posteriors.argmax(dim=1))
            graph_embeddings_batch = aggregate(node_embeddings_batch, batch)

        return node_embeddings_batch.double(), graph_embeddings_batch.double()

    def _compute_bigram(self, posteriors, edge_index, batch, no_graphs, device):

        if self.use_continuous_states:
            # Code provided by Daniele Atzeni to speed up the computation!
            nodes_in_batch = len(batch)
            sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                         torch.ones(edge_index.shape[1]),
                                                         torch.Size([nodes_in_batch, nodes_in_batch]))
            tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors.float()).repeat(1, self.C)
            tmp2 = posteriors.view(-1, 1).repeat(1, self.C).view(-1, self.C*self.C)
            node_bigram_batch = torch.mul(tmp1, tmp2)

            graph_bigram_batch = global_add_pool(node_bigram_batch, batch)

        else:
            # Covert into one hot
            posteriors_one_hot = self._make_one_hot(posteriors.argmax(dim=1)).float()

            # Code provided by Daniele Atzeni to speed up the computation!
            nodes_in_batch = len(batch)
            sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                         torch.ones(edge_index.shape[1]),
                                                         torch.Size([nodes_in_batch, nodes_in_batch]))
            tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors_one_hot).repeat(1, self.C)
            tmp2 = posteriors_one_hot.view(-1, 1).repeat(1, self.C).view(-1, self.C*self.C)
            node_bigram_batch = torch.mul(tmp1, tmp2)
            graph_bigram_batch = global_add_pool(node_bigram_batch, batch)

        return node_bigram_batch.double(), graph_bigram_batch.double()

    def _compute_max_ariety(self, degrees, batch):
        return scatter_max(degrees, batch)

    def _get_aggregation_fun(self):
        if self.aggregation == 'mean':
            aggregate = global_mean_pool
        elif self.aggregation == 'sum':
            aggregate = global_add_pool
        return aggregate

    def _make_one_hot(self, labels):
        one_hot = torch.zeros(labels.size(0), self.C)
        one_hot[torch.arange(labels.size(0)), labels] = 1
        return one_hot


class CGMMLayer:
    def __init__(self, k, c, a, c2, l, is_first_layer, node_type='discrete', device='cpu'):
        """
        utils Layer
        :param k: dimension of output's alphabet, which goes from 0 to K-1 (when discrete)
        :param c: the number of hidden states
        :param c2: the number of states of the neighbours
        :param l: number of previous layers to consider. You must pass the appropriate number of statistics at training
        :param a: dimension of edges' alphabet, which goes from 0 to A-1
        """
        super().__init__()
        self.device = device
        # For comparison w.r.t Numpy implementation
        # np.random.seed(seed=10)
        self.node_type = node_type
        self.is_layer_0 = is_first_layer

        self.eps = 1e-8  # Laplace smoothing
        self.C = c
        self.K = k
        self.orig_A = a
        self.A = a + 2  # may consider a special case of the recurrent arc and the special case of bottom state

        if not self.is_layer_0:
            self.C2 = c2
            self.L = l

        # Initialisation of the model's parameters.
        # torch.manual_seed(0)

        if self.is_layer_0:
            # For debugging w.r.t Numpy version
            # pr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))
            if 'cuda' in device:
                pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float64)).cuda()
            else:
                pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float64))
            self.prior = pr / pr.sum()

            # print(self.prior)

        if self.node_type == 'discrete':
            self.emission = CategoricalEmission(self.K, self.C, device)
        elif self.node_type == 'continuous':
            self.emission = GaussianEmission(self.K, self.C, device)

        # print(self.emission)

        if not self.is_layer_0:
            # For debugging w.r.t Numpy version
            # self.layerS = torch.from_numpy(np.random.uniform(size=self.L).astype(np.float32))  #
            if 'cuda' in device:
                self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float64)).cuda()
                self.arcS = torch.zeros((self.L, self.A), dtype=torch.float64).cuda()
                self.transition = torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float64).cuda()
            else:
                self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float64))
                self.arcS = torch.zeros((self.L, self.A), dtype=torch.float64)
                self.transition = torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float64)

            self.layerS /= self.layerS.sum()

            for layer in range(0, self.L):
                # For debugging w.r.t Numpy version
                # elf.arcS[layer, :] = torch.from_numpy(np.random.uniform(size=self.A).astype(np.float32))

                self.arcS[layer, :] = torch.nn.init.uniform_(self.arcS[layer, :])
                self.arcS[layer, :] /= self.arcS[layer, :].sum()
                for arc in range(0, self.A):
                    for j in range(0, self.C2):
                        # For debugging w.r.t Numpy version
                        # tr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))

                        tr = torch.nn.init.uniform_(torch.empty(self.C))
                        self.transition[layer, arc, :, j] = tr / tr.sum()

            # print(self.arcS)
            # print(self.transition)

        self.init_accumulators()

    def init_accumulators(self):

        # These are variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.emission.init_accumulators()

        if self.is_layer_0:
            if 'cuda' in self.device:
                self.prior_numerator = torch.full([self.C], self.eps, dtype=torch.float64).cuda()
            else:
                self.prior_numerator = torch.full([self.C], self.eps, dtype=torch.float64)
            self.prior_denominator = self.eps * self.C

        else:
            if 'cuda' in self.device:

                self.layerS_numerator = torch.full([self.L], self.eps, dtype=torch.float64).cuda()
                self.arcS_numerator = torch.full([self.L, self.A], self.eps, dtype=torch.float64).cuda()
                self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps, dtype=torch.float64).cuda()
                self.arcS_denominator = torch.full([self.L, 1], self.eps * self.A, dtype=torch.float64).cuda()
                self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps * self.C,
                                                     dtype=torch.float64).cuda()
            else:
                self.layerS_numerator = torch.full([self.L], self.eps, dtype=torch.float64)
                self.arcS_numerator = torch.full([self.L, self.A], self.eps, dtype=torch.float64)
                self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps, dtype=torch.float64)
                self.arcS_denominator = torch.full([self.L, 1], self.eps * self.A, dtype=torch.float64)
                self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps * self.C,
                                                         dtype=torch.float64)

            self.layerS_denominator = self.eps * self.L

    def _compute_posterior_estimate(self, emission_for_labels, stats):

        # print(stats.shape)

        batch_size = emission_for_labels.size()[0]

        # Compute the neighbourhood dimension for each vertex
        neighbDim = torch.sum(stats[:, :, :, :], dim=3).float()  # --> ? x L x A

        # Replace zeros with ones to avoid divisions by zero
        # This does not alter learning: the numerator can still be zero

        neighbDim = torch.where(neighbDim == 0., torch.tensor([1.]).to(self.device), neighbDim)
        neighbDim[:, :, -1] = 1

        broadcastable_transition = torch.unsqueeze(self.transition, 0)  # --> 1 x L x A x C x C2
        broadcastable_stats = torch.unsqueeze(stats, 3).double()  # --> ? x L x A x 1 x C2

        tmp = torch.sum(torch.mul(broadcastable_transition, broadcastable_stats), dim=4)  # --> ? x L x A x C2

        broadcastable_layerS = torch.unsqueeze(self.layerS, 1)  # --> L x 1

        tmp2 = torch.reshape(torch.mul(broadcastable_layerS, self.arcS), [1, self.L, self.A, 1])  # --> 1 x L x A x 1

        div_neighb = torch.reshape(neighbDim, [batch_size, self.L, self.A, 1]).double()  # --> ? x L x A x 1

        tmp_unnorm_posterior_estimate = torch.div(torch.mul(tmp, tmp2), div_neighb)  # --> ? x L x A x C2

        tmp_emission = torch.reshape(emission_for_labels,
                                     [batch_size, 1, 1, self.C])  # --> ? x 1 x 1 x C2

        unnorm_posterior_estimate = torch.mul(tmp_unnorm_posterior_estimate, tmp_emission)  # --> ? x L x A x C2

        # Normalize
        norm_constant = torch.reshape(torch.sum(unnorm_posterior_estimate, dim=[1, 2, 3]), [batch_size, 1, 1, 1])
        norm_constant = torch.where(norm_constant == 0., torch.Tensor([1.]).double().to(self.device), norm_constant)

        posterior_estimate = torch.div(unnorm_posterior_estimate, norm_constant)  # --> ? x L x A x C2

        return posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb

    def _E_step(self, labels, stats=None):
        batch_size = labels.size()[0]

        emission_of_labels = self.emission.get_distribution_of_labels(labels)

        if self.is_layer_0:
            # Broadcasting the prior
            numerator = torch.mul(emission_of_labels, torch.reshape(self.prior, shape=[1, self.C]))  # --> ?xC

            denominator = torch.sum(numerator, dim=1, keepdim=True)

            posterior_estimate = torch.div(numerator, denominator)  # --> ?xC

            # -------------------------------- Likelihood ------------------------------- #

            likelihood = torch.sum(torch.mul(posterior_estimate, torch.log(numerator)))

            return likelihood, posterior_estimate

        else:

            posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb \
                = self._compute_posterior_estimate(emission_of_labels, stats)

            posterior_uli = torch.sum(posterior_estimate, dim=2)  # --> ? x L x C
            posterior_ui = torch.sum(posterior_uli, dim=1)  # --> ? x C

            # -------------------------------- Likelihood -------------------------------- #

            # NOTE: these terms can become expensive in terms of memory consumption, mini-batch computation is required.

            log_trans = torch.log(self.transition)

            num = torch.div(
                torch.mul(self.transition,
                          torch.mul(torch.reshape(self.layerS, [self.L, 1, 1, 1]),
                                    torch.reshape(self.arcS, [self.L, self.A, 1, 1]))),
                torch.unsqueeze(div_neighb, 4))

            num = torch.mul(num, torch.reshape(emission_of_labels, [batch_size, 1, 1, self.C, 1]))
            num = torch.mul(num, broadcastable_stats)

            den = torch.sum(num, dim=[1, 2, 3, 4], keepdim=True)  # --> ? x 1 x 1 x 1 x 1
            den = torch.where(torch.eq(den, 0.), torch.tensor([1.]).double().to(self.device), den)

            eulaij = torch.div(num, den)  # --> ? x L x A x C x C2

            # Compute the expected complete log likelihood
            likelihood1 = torch.sum(torch.mul(posterior_ui, torch.log(emission_of_labels)))
            likelihood2 = torch.sum(torch.mul(posterior_uli, torch.log(broadcastable_layerS)))
            likelihood3 = torch.sum(torch.mul(posterior_estimate,
                                              torch.reshape(torch.log(self.arcS), [1, self.L, self.A, 1])))

            likelihood4 = torch.sum(torch.mul(torch.mul(eulaij, broadcastable_stats), log_trans))

            likelihood = likelihood1 + likelihood2 + likelihood3 + likelihood4

            return likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats

    def E_step(self, labels, stats=None, training=False):

        with torch.no_grad():
            if self.is_layer_0:
                likelihood, posterior_ui = self._E_step(labels, stats)
                if training:
                    self._M_step(labels, posterior_ui, None, None, None, None)

                return likelihood, posterior_ui

            else:
                likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats \
                    = self._E_step(labels, stats)
                if training:
                    self._M_step(labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats)
                    return likelihood, eulaij
                else:
                    return likelihood, posterior_ui

    def _M_step(self, labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats):

        if self.is_layer_0:

            tmp = torch.sum(posterior_estimate, dim=0)
            # These are used at each minibatch
            self.prior_numerator += tmp
            self.prior_denominator += torch.sum(tmp)

            self.emission.update_accumulators(posterior_estimate, labels)

        else:

            # These are equivalent to the categorical mixture model, it just changes how the posterior is computed
            self.emission.update_accumulators(posterior_ui, labels)

            tmp_arc_num = torch.sum(posterior_estimate, dim=[0, 3])  # --> L x A
            self.arcS_numerator += tmp_arc_num
            self.arcS_denominator += torch.unsqueeze(torch.sum(tmp_arc_num, dim=1), 1)  # --> L x 1

            new_layer_num = torch.sum(posterior_uli, dim=[0, 2])  # --> [L]
            self.layerS_numerator += new_layer_num
            self.layerS_denominator += torch.sum(new_layer_num)  # --> [1]

            new_trans_num = torch.sum(torch.mul(eulaij, broadcastable_stats), dim=0)
            self.transition_numerator += new_trans_num
            self.transition_denominator += torch.unsqueeze(torch.sum(new_trans_num, dim=2), 2)  # --> L x A x 1 x C2

    def M_step(self):

        self.emission.update_parameters()
        if self.is_layer_0:
            self.prior = self.prior_numerator / self.prior_denominator

        else:

            self.layerS = self.layerS_numerator / self.layerS_denominator
            self.arcS = self.arcS_numerator / self.arcS_denominator

            self.transition = self.transition_numerator / self.transition_denominator

        # I need to re-init accumulators, otherwise they will contain statistics of the previous epochs
        self.init_accumulators()
