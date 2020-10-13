import torch
from time import time
from config.utils import s2c
from model.util.categorical_emission import CategoricalEmission
from model.util.gaussian_emission import GaussianEmission
from model.util.mixed_emission import MixedEmission
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import global_mean_pool, global_add_pool

from model.util.mixture_model import MixtureModel
from model.util.utils import _compute_bigram, _compute_unigram
from training.core.engine import TrainingEngine


class CGMM:

    def __init__(self, dim_node_features, dim_edge_features, dim_target, predictor_class, config):

        self.dim_node_features = dim_node_features
        # self.dim_edge_features = dim_edge_features not used
        self.dim_target = dim_target
        self.predictor_class = predictor_class
        self.layer_config = config
        self.depth = config['depth']
        self.is_first_layer = self.depth == 1
        self.training = False
        self.compute_intermediate_outputs = False

        print(f'Initializing layer {self.depth}')
        self.K = dim_node_features

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

        if self.node_type == 'discrete':
            emission = CategoricalEmission(self.K, self.C)
        elif self.node_type == 'continuous':
            emission = GaussianEmission(self.K, self.C)
        elif self.node_type == 'mixed':
            emission = MixedEmission(self.K, self.C)

        if self.is_first_layer:
            self.layer = MixtureModel(self.C, emission)
        else:
            self.layer = CGMMLayer(self.C, self.A, self.C2, self.L, self.is_first_layer, emission)

    def to(self, device):
        self.device = device
        self.layer.to(device)

        return self

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, data):
        return self.e_step(data, self.training)

    def e_step(self, data, training=False):

        if self.is_first_layer:
            # If training, E-step also accumulates statistics for the M-step!
            likelihood, posterior_batch = self.layer.e_step(data.x, data.batch, training=training)
        else:
            data, extra = data[0], data[1]

            prev_stats = extra.vo_outs if extra is not None else None
            if prev_stats is not None:
                prev_stats.to(self.device)

            # E-step also accumulates statistics for the M-step!
            likelihood, posterior_batch = self.layer.e_step(data.x, prev_stats, training=training)
            likelihood, posterior_batch = likelihood.detach(), posterior_batch.detach()

        if self.compute_intermediate_outputs:
            # print("Computing intermediate outputs")
            statistics_batch = self._compute_statistics(posterior_batch, data, self.device)

            node_unigram = _compute_unigram(posterior_batch, self.use_continuous_states)
            graph_unigram = self._get_aggregation_fun()(node_unigram, data.batch)

            if self.unibigram:
                node_bigram = _compute_bigram(posterior_batch.float(), data.edge_index, data.batch,
                                              self.use_continuous_states)
                graph_bigram = self._get_aggregation_fun()(node_unigram, data.batch)

                node_embeddings_batch = torch.cat((node_unigram, node_bigram), dim=1)
                graph_embeddings_batch = torch.cat((graph_unigram, graph_bigram), dim=1)
            else:
                node_embeddings_batch = node_unigram
                graph_embeddings_batch = graph_unigram

            # to save time during debug
            embeddings = (node_embeddings_batch, None, graph_embeddings_batch, statistics_batch, None, None)
        else:
            embeddings = None

        return likelihood, embeddings

    def m_step(self):
        self.layer.m_step()

    def arbitrary_logic(self, experiment, train_loader, layer_config, is_last_layer, validation_loader=None,
                        test_loader=None,
                        logger=None, device=None):
        if is_last_layer:
            dim_features = (
            self.dim_node_features, self.C * self.depth if not self.unibigram else (self.C + self.C * self.C2) * self.depth)

            config = layer_config['arbitrary_function_config']
            device = config['device']

            predictor_class = s2c(config['predictor'])
            model = predictor_class(dim_features, 0, self.dim_target, config)

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

            plot_class, plot_args = experiment._return_class_and_args(config, 'plotter')
            plotter = plot_class(exp_path=experiment.exp_path, **plot_args) if plot_class is not None else None

            wrapper = s2c(config['wrapper'])(model=model, loss=loss,
                                             optimizer=optimizer, scorer=scorer, scheduler=scheduler,
                                             early_stopper=early_stopper, gradient_clipping=grad_clipper,
                                             device=device, plotter=plotter)

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
            statistics = torch.full((posteriors.shape[0], self.A + 2, self.C2), 1e-8, dtype=torch.float32).cuda()
        else:
            statistics = torch.full((posteriors.shape[0], self.A + 2, self.C2), 1e-8, dtype=torch.float32)

        srcs, dsts = data.edge_index

        if self.A == 1:

            # for source, dest, in zip(srcs, dsts):
            #    statistics[dest, 0, :-1] += posteriors[source]

            sparse_adj_matr = torch.sparse_coo_tensor(data.edge_index, \
                                                      torch.ones(data.edge_index.shape[1], dtype=posteriors.dtype), \
                                                      torch.Size([posteriors.shape[0],
                                                                  posteriors.shape[0]])).transpose(0, 1)
            statistics[:, 0, :-1] = torch.sparse.mm(sparse_adj_matr, posteriors)

            # assert torch.allclose(statistics, new_statistics)

        else:

            #arc_labels = data.edge_attr.long()
            #for source, dest, arc_label in zip(srcs, dsts, arc_labels):
            #    statistics[dest, arc_label, :-1] += posteriors[source]

            for arc_label in range(self.A):
                sparse_label_adj_matr = torch.sparse_coo_tensor(data.edge_index, \
                                                                (data.edge_attr == arc_label).long, \
                                                                torch.Size([posteriors.shape[0],
                                                                            posteriors.shape[0]])).transpose(0, 1)

                statistics[:, arc_label, :-1] = torch.sparse.mm(sparse_label_adj_matr, posteriors)


        if self.add_self_arc:
            statistics[:, self.A, :-1] += posteriors

        # use self.A+1 as special edge for bottom states (all in self.C2-1)
        degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()

        max_arieties, _ = self._compute_max_ariety(degrees.int(), data.batch)
        statistics[:, self.A + 1, self.C2 - 1] += 1 - (degrees / max_arieties[data.batch].float())

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


class CGMMLayer:
    def __init__(self, c, a, c2, l, is_first_layer, emission):
        super().__init__()
        self.device = None
        self.is_layer_0 = is_first_layer

        self.eps = 1e-8  # Laplace smoothing
        self.C = c
        self.orig_A = a
        self.A = a + 2  # may consider a special case of the recurrent arc and the special case of bottom state

        self.emission = emission
        self.C2 = c2
        self.L = l

        if not self.is_layer_0:
            self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float32))
            self.arcS = torch.zeros((self.L, self.A), dtype=torch.float32)
            self.transition = torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float32)

            self.layerS /= self.layerS.sum()

            for layer in range(0, self.L):
                self.arcS[layer, :] = torch.nn.init.uniform_(self.arcS[layer, :])
                self.arcS[layer, :] /= self.arcS[layer, :].sum()
                for arc in range(0, self.A):
                    for j in range(0, self.C2):
                        tr = torch.nn.init.uniform_(torch.empty(self.C))
                        self.transition[layer, arc, :, j] = tr / tr.sum()

            # print(self.arcS)
            # print(self.transition)

        self.init_accumulators()

    def to(self, device):
        self.device = device
        self.emission.to(device)
        self.layerS.to(device)
        self.arcS.to(device)
        self.transition.to(device)

    def init_accumulators(self):

        # These are variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.emission.init_accumulators()
        self.layerS_numerator = torch.full([self.L], self.eps, dtype=torch.float32)
        self.arcS_numerator = torch.full([self.L, self.A], self.eps, dtype=torch.float32)
        self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps, dtype=torch.float32)
        self.arcS_denominator = torch.full([self.L, 1], self.eps * self.A, dtype=torch.float32)
        self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps * self.C,
                                                     dtype=torch.float32)
        self.layerS_denominator = self.eps * self.L

        # Do not delete this!
        if self.device:  # set by to() method
            self.to(self.device)

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
        broadcastable_stats = torch.unsqueeze(stats, 3).float()  # --> ? x L x A x 1 x C2

        tmp = torch.sum(torch.mul(broadcastable_transition, broadcastable_stats), dim=4)  # --> ? x L x A x C2

        broadcastable_layerS = torch.unsqueeze(self.layerS, 1)  # --> L x 1

        tmp2 = torch.reshape(torch.mul(broadcastable_layerS, self.arcS), [1, self.L, self.A, 1])  # --> 1 x L x A x 1

        div_neighb = torch.reshape(neighbDim, [batch_size, self.L, self.A, 1]).float()  # --> ? x L x A x 1

        tmp_unnorm_posterior_estimate = torch.div(torch.mul(tmp, tmp2), div_neighb)  # --> ? x L x A x C2

        tmp_emission = torch.reshape(emission_for_labels,
                                     [batch_size, 1, 1, self.C])  # --> ? x 1 x 1 x C2

        unnorm_posterior_estimate = torch.mul(tmp_unnorm_posterior_estimate, tmp_emission)  # --> ? x L x A x C2

        # Normalize
        norm_constant = torch.reshape(torch.sum(unnorm_posterior_estimate, dim=[1, 2, 3]), [batch_size, 1, 1, 1])
        norm_constant = torch.where(norm_constant == 0., torch.Tensor([1.]).float().to(self.device), norm_constant)

        posterior_estimate = torch.div(unnorm_posterior_estimate, norm_constant)  # --> ? x L x A x C2

        return posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb

    def _e_step(self, labels, stats=None):
        batch_size = labels.size()[0]

        emission_of_labels = self.emission.get_distribution_of_labels(labels)

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
        den = torch.where(torch.eq(den, 0.), torch.tensor([1.]).float().to(self.device), den)

        eulaij = torch.div(num, den)  # --> ? x L x A x C x C2

        # Compute the expected complete log likelihood

        # TODO this should be rewritten as done in IOMixtureModel

        likelihood1 = torch.mean(torch.mul(posterior_ui, torch.log(emission_of_labels)).sum(1))
        likelihood2 = torch.mean(torch.mul(posterior_uli, torch.log(broadcastable_layerS)).sum((1,2)))
        likelihood3 = torch.mean(torch.mul(posterior_estimate,
                                          torch.reshape(torch.log(self.arcS), [1, self.L, self.A, 1])).sum((1,2,3)))

        likelihood4 = torch.mean(torch.mul(torch.mul(eulaij, broadcastable_stats), log_trans).sum((1,2,3,4)))

        likelihood = likelihood1 + likelihood2 + likelihood3 + likelihood4

        return likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats

    def e_step(self, labels, stats=None, training=False):

        likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats \
            = self._e_step(labels, stats)
        if training:
            self._m_step(labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats)
            return likelihood, eulaij
        else:
            return likelihood, posterior_ui

    def _m_step(self, labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats):

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

    def m_step(self):

        self.emission.update_parameters()
        self.layerS = self.layerS_numerator / self.layerS_denominator
        self.arcS = self.arcS_numerator / self.arcS_denominator

        self.transition = self.transition_numerator / self.transition_denominator

        # I need to re-init accumulators, otherwise they will contain statistics of the previous epochs
        self.init_accumulators()