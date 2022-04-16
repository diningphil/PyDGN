from typing import List, Union

import torch
from torch.nn import Module, CrossEntropyLoss, MSELoss

from pydgn.experiment.util import s2c
from pydgn.static import BATCH_LOSS_EXTRA, ARGS, CLASS_NAME
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State


class Metric(Module, EventHandler):
    r"""
    Metric is the main event handler for all metrics. Other metrics can easily subclass by implementing the :func:`forward`
    method, though sometimes more complex implementations are required.

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
    """
    def __init__(self, use_as_loss: bool=False, reduction: str='mean', use_nodes_batch_size: bool=False):
        super().__init__()
        self.batch_metrics = None
        self.num_samples = None
        self.use_as_loss = use_as_loss
        self.reduction = reduction
        self.use_nodes_batch_size = use_nodes_batch_size

    @property
    def name(self) -> str:
        raise NotImplementedError('You should subclass Metric and implement this method!')

    def get_main_metric_name(self) -> str:
        """
        Return the metric's main name. Useful when a metric is the combination of many.

        Returns:
            the metric's main name
        """
        return self.name

    def _expand_reduction(self, state: State):
        #
        # Returns the number of samples to use, conditioned on the type of reduction (e.g., sum, mean) to obtain
        # the sum of the individual sample scores.
        # Used to average sample scores across the entire epoch, rather than taking an average of minibatch scores
        #
        if self.reduction == 'mean':
            # Used to recover the "sum" version of the metric
            return state.batch_num_targets if not self.use_nodes_batch_size else state.batch_num_nodes

        elif self.reduction == 'sum':
            return 1
        else:
            raise NotImplementedError('The only reductions allowed are sum and mean')

    def _update_num_samples(self, state: State):
        #
        # Returns the number of samples to use when updating the number of total samples per epoch
        #
        return state.batch_num_targets if not self.use_nodes_batch_size else state.batch_num_nodes

    def on_training_epoch_start(self, state: State):
        self.batch_metrics = []
        self.num_samples = 0

    def on_training_batch_end(self, state: State):
        if self.use_as_loss:
            batch_metric = state.batch_loss
        else:
            batch_metric = state.batch_score

        self.batch_metrics.append(batch_metric[self.name].item() * self._expand_reduction(state))
        self.num_samples += self._update_num_samples(state)

    def on_training_epoch_end(self, state: State):
        if self.use_as_loss:
            state.update(epoch_loss={self.name: torch.tensor(self.batch_metrics).sum() / self.num_samples})
        else:
            state.update(epoch_score={self.name: torch.tensor(self.batch_metrics).sum() / self.num_samples})

        self.batch_metrics = None
        self.num_samples = None

    def on_eval_epoch_start(self, state: State):
        self.batch_metrics = []
        self.num_samples = 0

    def on_eval_epoch_end(self, state: State):
        if self.use_as_loss:
            state.update(epoch_loss={self.name: torch.tensor(self.batch_metrics).sum() / self.num_samples})
        else:
            state.update(epoch_score={self.name: torch.tensor(self.batch_metrics).sum() / self.num_samples})

        self.batch_metrics = None
        self.num_samples = None

    def on_eval_batch_end(self, state: State):
        if self.use_as_loss:
            batch_metric = state.batch_loss
        else:
            batch_metric = state.batch_score

        self.batch_metrics.append(batch_metric[self.name].item() * self._expand_reduction(state))
        self.num_samples += self._update_num_samples(state)

    def on_compute_metrics(self, state: State):
        outputs, targets = state.batch_outputs, state.batch_targets

        if self.use_as_loss:
            loss_output = self.forward(targets, *outputs)

            if isinstance(loss_output, tuple):
                # Allow loss to produce intermediate results that speed up
                # Score computation. This loss callback MUST occur before the score one.
                loss, extra = loss_output
                state.update(batch_loss_extra={self.name: extra})
            else:
                loss = loss_output

            state.update(batch_loss={self.name: loss})

        else:
            score = self.forward(targets,
                                 *outputs,
                                 batch_loss_extra=getattr(state, BATCH_LOSS_EXTRA, None))

            # Handle base case: forward returns a score
            # This is used to work with implementations that return multiple scores, such as MultiScore
            if type(score) is not dict:
                score = {self.name: score}

            # Score is a dictionary with key-value pairs
            # we need to detach each score from the computational graph
            score = {k: v.detach().cpu() for k, v in score.items()}
            state.update(batch_score=score)

    def on_backward(self, state: State):
        if self.use_as_loss:
            try:
                state.batch_loss[self.name].backward()
            except Exception as e:
                # Here we catch potential multiprocessing related issues
                # see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
                print(e)

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> Union[dict, float]:
        r"""
        Computes the metric value. Optionally, and only for scores used as losses, some extra information can be also returned.

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model
            batch_loss_extra (dict): dictionary of information computed by metrics used as losses

        Returns:
            A dictionary containing associations metric_name - value or simply a value
        """
        raise NotImplementedError('To be implemented by a subclass')


class MultiScore(Metric):
    r"""
    This class is used to keep track of multiple additional metrics used as scores, rather than losses.

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
    """
    def __init__(self, use_as_loss, reduction='mean', use_nodes_batch_size=False, main_scorer=None, **extra_scorers):
        assert not use_as_loss, "MultiScore cannot be used as loss"
        assert not main_scorer is None, "You have to provide a main scorer"
        super().__init__(use_as_loss, reduction, use_nodes_batch_size)
        self.scorers = [self._istantiate_scorer(main_scorer)] + [self._istantiate_scorer(score) for score in
                                                                 extra_scorers.values()]

    @property
    def name(self) -> str:
        return "Multi Score"

    def _istantiate_scorer(self, scorer):
        if isinstance(scorer, dict):
            args = scorer[ARGS]
            return s2c(scorer[CLASS_NAME])(use_as_loss=False, **args)
        else:
            return s2c(scorer)(use_as_loss=False)

    def get_main_metric_name(self):
        return self.scorers[0].get_main_metric_name()

    def on_training_epoch_start(self, state: State):
        self.batch_metrics = {s.name: [] for s in self.scorers}
        for scorer in self.scorers:
            scorer.on_training_epoch_start(state)

    def on_training_batch_end(self, state: State):
        for scorer in self.scorers:
            scorer.on_training_batch_end(state)

    def on_training_epoch_end(self, state: State):
        epoch_score = {}
        for scorer in self.scorers:
            # This will update the epoch_score variable in State
            scorer.on_training_epoch_end(state)
            epoch_score.update(state.epoch_score)
        state.update(epoch_score=epoch_score)

    def on_eval_epoch_start(self, state: State):
        for scorer in self.scorers:
            scorer.on_eval_epoch_start(state)

    def on_eval_batch_end(self, state: State):
        for scorer in self.scorers:
            scorer.on_eval_batch_end(state)

    def on_eval_epoch_end(self, state: State):
        epoch_score = {}
        for scorer in self.scorers:
            # This will update the epoch_score variable in State
            scorer.on_training_epoch_end(state)
            epoch_score.update(state.epoch_score)
        state.update(epoch_score=epoch_score)

    def on_compute_metrics(self, state: State):
        super().on_compute_metrics(state)

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> dict:
        res = {}
        for scorer in self.scorers:
            # each scorer __call__ method returns a dict
            res.update({scorer.name: scorer.forward(targets, *outputs, batch_loss_extra=batch_loss_extra)})

        return res


class AdditiveLoss(Metric):
    r"""
    AdditiveLoss sums an arbitrary number of losses together.

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
        losses (dict): dictionary of metrics to add together
    """
    def __init__(self, use_as_loss, reduction='mean', use_nodes_batch_size=False, **losses: dict):
        assert use_as_loss, "Additive loss can only be used as a loss"
        super().__init__(use_as_loss, reduction, use_nodes_batch_size)

        self.losses = [self._instantiate_loss(loss) for loss in losses.values()]
        self.use_nodes_batch_size = use_nodes_batch_size

    @property
    def name(self) -> str:
        return "Additive Loss"

    def _instantiate_loss(self, loss):
        if isinstance(loss, dict):
            args = loss[ARGS]
            return s2c(loss[CLASS_NAME])(use_as_loss=True, **args)
        else:
            return s2c(loss)(use_as_loss=True)

    def on_training_epoch_start(self, state: State):
        self.batch_metrics = {l.name: [] for l in [self] + self.losses}
        self.num_samples = 0

    def on_training_batch_end(self, state: State):
        for k, v in state.batch_loss.items():
            self.batch_metrics[k].append(v.item() * self._expand_reduction(state))
        self.num_samples += self._update_num_samples(state)

    def on_training_epoch_end(self, state: State):
        state.update(epoch_loss={l.name: torch.tensor(self.batch_metrics[l.name]).sum() / self.num_samples
                                 for l in [self] + self.losses})
        self.batch_metrics = None
        self.num_samples = None

    def on_eval_epoch_start(self, state: State):
        self.batch_metrics = {l.name: [] for l in [self] + self.losses}
        self.num_samples = 0

    def on_eval_epoch_end(self, state: State):
        state.update(epoch_loss={l.name: torch.tensor(self.batch_metrics[l.name]).sum() / self.num_samples
                                 for l in [self] + self.losses})
        self.batch_metrics = None
        self.num_samples = None

    def on_eval_batch_end(self, state: State):
        for k, v in state.batch_loss.items():
            self.batch_metrics[k].append(v.item() * self._expand_reduction(state))
        self.num_samples += self._update_num_samples(state)

    def on_compute_metrics(self, state: State):
        outputs, targets = state.batch_outputs, state.batch_targets
        loss = {}
        extra = {}
        loss_sum = 0.
        for l in self.losses:
            single_loss = l.forward(targets, *outputs)
            if isinstance(single_loss, tuple):
                # Allow loss to produce intermediate results that speed up
                # Score computation. Loss callback MUST occur before the score one.
                loss_output, loss_extra = single_loss
                extra[l.name] = loss_extra
                state.update(batch_loss_extra=extra)
            else:
                loss_output = single_loss
            loss[l.name] = loss_output
            loss_sum += loss_output

        loss[self.name] = loss_sum
        state.update(batch_loss=loss)

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> dict:
        pass
    

class Classification(Metric):
    r"""
    Generic metric for classification tasks. Used to maximize code reuse for classical metrics.

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
    """
    def __init__(self, use_as_loss=False, reduction='mean', use_nodes_batch_size=False):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)
        self.metric = None

    @property
    def name(self) -> str:
        return 'Classification Metric'

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> dict:
        outputs = outputs[0]

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        metric = self.metric(outputs.squeeze(dim=1), targets)
        return metric


class Regression(Metric):
    r"""
    Generic metric for regression tasks. Used to maximize code reuse for classical metrics.

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
    """
    def __init__(self, use_as_loss=False, reduction='mean', use_nodes_batch_size=False):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)
        self.metric = None

    @property
    def name(self) -> str:
        return 'Regression Metric'

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> dict:
        outputs = outputs[0]

        metric = self.metric(outputs.squeeze(dim=1), targets.squeeze(dim=1))
        return metric


class MulticlassClassification(Classification):
    r"""
    Wrapper around :class:`torch.nn.CrossEntropyLoss`

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**.
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
    """
    def __init__(self, use_as_loss=False, reduction='mean', use_nodes_batch_size=False):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)
        self.metric = CrossEntropyLoss(reduction=reduction)

    @property
    def name(self) -> str:
        return 'Multiclass Classification'

class MeanSquareError(Regression):
    r"""
    Wrapper around :class:`torch.nn.MSELoss`

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**.
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
    """
    def __init__(self, use_as_loss=False, reduction='mean', use_nodes_batch_size=False):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)
        self.metric = MSELoss(reduction=reduction)

    @property
    def name(self) -> str:
        return 'Mean Square Error'


class DotProductLink(Metric):
    """
    Implements a dot product link prediction metric, as defined in https://arxiv.org/abs/1611.07308.
    """
    @property
    def name(self) -> str:
        return 'Dot Product Link Prediction'

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> dict:
        node_embs = outputs[1]
        _, pos_edges, neg_edges = targets

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                 torch.zeros(neg_edges.shape[1])))

        # Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, loss_target.to(link_logits.device))
        return loss


class MulticlassAccuracy(Metric):
    """
    Implements multiclass classification accuracy.
    """

    @property
    def name(self) -> str:
        return 'Multiclass Accuracy'

    @staticmethod
    def _get_correct(output):
        return torch.argmax(output, dim=1)

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> dict:
        pred = outputs[0]
        correct = self._get_correct(pred)

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        return 100. * (correct == targets).sum().float() / targets.size(0)


class ToyMetric(Metric):
    """
    Implements a toy metric.

    Args:
        use_as_loss (bool): whether this metric should act as a loss (i.e., it should act when :func:`on_backward` is called). **Used by PyDGN, no need to care about this.**.
        reduction (str): the type of reduction to apply across samples of the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        use_nodes_batch_size (bool): whether or not to use the # of nodes in the batch, rather than the number of graphs, to compute
        the metric's aggregated value for the entire epoch.
    """

    def __init__(self, use_as_loss=False, reduction='mean', use_nodes_batch_size=False):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)


    @property
    def name(self) -> str:
        return 'Toy Metric'

    @staticmethod
    def _get_correct(output):
        return torch.argmax(output, dim=1)

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict=None) -> dict:
        return 100. * torch.ones(1)


class ToyUnsupervisedMetric(Metric):

    @property
    def name(self) -> str:
        return 'Toy Unsupervised Metric'

    def forward(self,
                targets: torch.Tensor,
                *outputs: List[torch.Tensor],
                batch_loss_extra: dict = None) -> dict:
        return (outputs[0]*0.).mean()