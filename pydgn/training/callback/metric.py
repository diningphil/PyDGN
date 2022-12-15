from typing import List, Union, Tuple

import torch
from torch.nn import Module, CrossEntropyLoss, MSELoss, L1Loss

from pydgn.experiment.util import s2c
from pydgn.static import ARGS, CLASS_NAME
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State


class Metric(Module, EventHandler):
    r"""
    Metric is the main event handler for all metrics. Other metrics can easily
    subclass by implementing the :func:`forward` method, though sometimes
    more complex implementations are required.

    Args:
        use_as_loss (bool): whether this metric should act as a loss
            (i.e., it should act when :func:`on_backward` is called).
            **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of
            the mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        accumulate_over_epoch (bool): Whether or not to display the epoch-wise
            metric rather than an average of per-batch metrics.
            If true, it keep a list of predictions and target values across
            the entire epoch. Use it especially with batch-sensitive metrics,
            such as micro AP/F1 scores. Default is ``True``.
        force_cpu (bool): Whether or not to move all predictions to cpu
            before computing the epoch-wise loss/score. Default is ``True``.
        device (bool): The device used. Default is 'cpu'.
    """

    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.batch_metrics = None
        self.use_as_loss = use_as_loss
        self.reduction = reduction
        self.accumulate_over_epoch = accumulate_over_epoch
        self.force_cpu = force_cpu
        self.device = device
        self.y_pred, self.y_true = None, None

        # Keeps track of how many times on_compute_metrics has been called
        # in the same batch.
        # Useful for temporal graph learning, for static graph learning it will
        # be always 1
        self.timesteps_in_batch = None

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        raise NotImplementedError(
            "You should subclass Metric and implement this method!"
        )

    def get_main_metric_name(self) -> str:
        """
        Return the metric's main name.
        Useful when a metric is the combination of many.

        Returns:
            the metric's main name
        """
        return self.name

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns predictions and target tensors to be
        accumulated for a given metric

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        raise NotImplementedError(
            "You should subclass Metric and implement this method!"
        )

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        r"""
        Computes the metric for a given set of targets and predictions

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        raise NotImplementedError(
            "You should subclass Metric and implement this method!"
        )

    def on_training_epoch_start(self, state: State):
        """
        Initialize list of batch metrics as well as the list of batch
        predictions and targets for the metric

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        self.batch_metrics = []
        self.y_pred, self.y_true = {self.name: []}, {self.name: []}

    def on_training_batch_start(self, state: State):
        """
        Initializes the number of potential time steps in a batch
        (for temporal learning)

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        self.timesteps_in_batch = 0.0

    def on_training_batch_end(self, state: State):
        """
        If we do not computed aggregated metric values over the entire epoch,
        populate the batch metrics list with the new loss/score.
        Divide by the number of timesteps in the batch
        (default is 1 for static datasets)

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        if self.use_as_loss:
            batch_metric = state.batch_loss
        else:
            batch_metric = state.batch_score

        if not self.accumulate_over_epoch:
            self.batch_metrics.append(
                batch_metric[self.name].item() / self.timesteps_in_batch
            )

        self.timesteps_in_batch = None

    def on_training_epoch_end(self, state: State):
        """
        Computes the mean of batch metrics or an aggregated score over all
        epoch depending on the `accumulate_over_epoch` parameter. Updates
        `epoch_loss` and `epoch_score` fields in the state
        and resets the basic fields used.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        if not self.accumulate_over_epoch:
            epoch_res = {
                self.name: torch.tensor(self.batch_metrics).sum()
                / len(self.batch_metrics)
            }
        else:
            epoch_res = {
                self.name: self.compute_metric(
                    torch.cat(self.y_true[self.name], dim=0),
                    torch.cat(self.y_pred[self.name], dim=0),
                )
            }

        if self.use_as_loss:
            state.update(epoch_loss=epoch_res)
        else:
            state.update(epoch_score=epoch_res)

        self.batch_metrics = None
        self.y_pred, self.y_true = None, None

    def on_eval_epoch_start(self, state: State):
        """
        Initialize list of batch metrics as well as the list of batch
        predictions and targets for the metric

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        self.batch_metrics = []
        self.y_pred, self.y_true = {self.name: []}, {self.name: []}

    def on_eval_batch_start(self, state: State):
        """
        Initializes the number of potential time steps in a batch
        (for temporal learning)

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        self.timesteps_in_batch = 0.0

    def on_eval_batch_end(self, state: State):
        """
        If we do not computed aggregated metric values over the entire epoch,
        populate the batch metrics list with the new loss/score.
        Divide by the number of timesteps in the batch
        (default is 1 for static datasets)

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        if self.use_as_loss:
            batch_metric = state.batch_loss
        else:
            batch_metric = state.batch_score

        if not self.accumulate_over_epoch:
            self.batch_metrics.append(
                batch_metric[self.name].item() / self.timesteps_in_batch
            )

        self.timesteps_in_batch = None

    def on_eval_epoch_end(self, state: State):
        """
        Computes the mean of batch metrics or an aggregated score over all
        epoch depending on the `accumulate_over_epoch` parameter.
        Updates `epoch_loss` and `epoch_score` fields in the state
        and resets the basic fields used.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        if not self.accumulate_over_epoch:
            epoch_res = {
                self.name: torch.tensor(self.batch_metrics).sum()
                / len(self.batch_metrics)
            }
        else:
            epoch_res = {
                self.name: self.compute_metric(
                    torch.cat(self.y_true[self.name], dim=0),
                    torch.cat(self.y_pred[self.name], dim=0),
                )
            }

        if self.use_as_loss:
            state.update(epoch_loss=epoch_res)
        else:
            state.update(epoch_score=epoch_res)

        self.batch_metrics = None
        self.y_pred, self.y_true = None, None

    def accumulate_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> None:
        """
        Used to specify how to accumulate predictions and targets.
        This can be customized by subclasses like
        AdditiveLoss and MultiScore to accumulate predictions and
        targets for different losses/scores.

        Args:
            targets: target tensor
            *outputs: outputs of the model
        """
        y_pred_batch, y_true_batch = self.get_predictions_and_targets(
            targets, *outputs
        )
        metric_name = self.name

        self.y_pred[metric_name].append(
            y_pred_batch.detach().cpu()
            if self.force_cpu
            else y_pred_batch.detach()
        )
        self.y_true[metric_name].append(
            y_true_batch.detach().cpu()
            if self.force_cpu
            else y_true_batch.detach()
        )

    def on_compute_metrics(self, state: State):
        """
        Computes the loss/score depending on the metric, updating the
        `batch_loss` or `batch_score` field in the state.
        In temporal graph learning, this method is computed more than once
        before the batch ends, so we accumulate the loss or scores across
        timesteps of a single batch.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        self.timesteps_in_batch += 1.0

        outputs, targets = state.batch_outputs, state.batch_targets

        # Loss case
        if self.use_as_loss:
            loss = self.forward(targets, *outputs)

            # [temporal graph learning] accumulate, rather than substitute,
            # the losses across multiple snapshots
            # these are reset accordingly by the Training Engine!
            if state.batch_loss is not None:
                old_loss = state.batch_loss
                loss = {k: old_loss[k] + loss[k] for k in loss.keys()}

            # this has to be updated per-batch no matter what
            state.update(batch_loss=loss)

        # Score case
        else:
            if not self.accumulate_over_epoch:
                score = self.forward(targets, *outputs)
                score = {
                    k: score[k].detach().cpu()
                    if self.force_cpu
                    else score[k].detach()
                    for k in score.keys()
                }

                # [temporal graph learning] accumulate, rather than substitute,
                # the scores across multiple snapshots
                # these are reset accordingly by the Training Engine!
                if state.batch_score is not None:
                    old_score = state.batch_score
                    score = {k: old_score[k] + score[k] for k in score.keys()}

                # mean of batches
                state.update(batch_score=score)

        if self.accumulate_over_epoch:
            self.accumulate_predictions_and_targets(targets, *outputs)

    def on_backward(self, state: State):
        """
        Calls backward on the loss if the metric is a loss.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        if self.use_as_loss:
            try:
                state.batch_loss[self.name].backward()
            except Exception as e:
                # Here we catch potential multiprocessing related issues
                # see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
                raise (e)

    def forward(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> dict:
        r"""
        Computes the metric value. Optionally, and only for scores used
        as losses, some extra information can be also returned.

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model
            batch_loss_extra (dict): dictionary of information
                computed by metrics used as losses

        Returns:
            A dictionary containing associations metric_name - value
        """
        y_pred_batch, y_true_batch = self.get_predictions_and_targets(
            targets, *outputs
        )
        return {self.name: self.compute_metric(y_true_batch, y_pred_batch)}


class MultiScore(Metric):
    r"""
    This class is used to keep track of multiple additional metrics
    used as scores, rather than losses.

    Args:
        use_as_loss (bool): whether this metric should act as a loss
            (i.e., it should act when :func:`on_backward` is called).
            **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of the
            mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        accumulate_over_epoch (bool): Whether or not to display the epoch-wise
            metric rather than an average of per-batch metrics.  If true, it
            keeps a list of predictions and target values across the entire
            epoch. Use it especially with batch-sensitive metrics,
            such as micro AP/F1 scores. Default is ``True``.
        force_cpu (bool): Whether or not to move all predictions to cpu
            before computing the epoch-wise loss/score. Default is ``True``.
        device (bool): The device used. Default is 'cpu'.
        main_scorer (:class:`~pydgn.training.callback.metric.Metric`): the
            score on which final results are computed.
        extra_scorers (dict): dictionary of other metrics to consider.
    """

    def __init__(
        self,
        use_as_loss,
        reduction="mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        main_scorer=None,
        **extra_scorers,
    ):

        assert not use_as_loss, "MultiScore cannot be used as loss"
        assert main_scorer is not None, "You have to provide a main scorer"
        super().__init__(
            use_as_loss, reduction, accumulate_over_epoch, force_cpu, device
        )
        self.scores = [self._istantiate_scorer(main_scorer)] + [
            self._istantiate_scorer(score) for score in extra_scorers.values()
        ]

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Multi Score"

    def _istantiate_scorer(self, scorer):
        """
        Istantiate a scorer with its own arguments (if any are given)
        """
        if isinstance(scorer, dict):
            args = scorer[ARGS]
            return s2c(scorer[CLASS_NAME])(use_as_loss=False, **args)
        else:
            return s2c(scorer)(use_as_loss=False)

    def get_main_metric_name(self):
        """
        Returns the name of the first scorer that is passed to this class via
        the `__init__` method.
        """
        return self.scores[0].get_main_metric_name()

    def on_training_epoch_start(self, state: State):
        """
        Compared to superclass version, initializes a dictionary for each score
        to track rather than single lists

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.batch_metrics = {s.name: [] for s in self.scores}
        self.y_pred = {s.name: [] for s in self.scores}
        self.y_true = {s.name: [] for s in self.scores}

    def on_training_batch_end(self, state: State):
        """
        For each scorer, computes the average metric in the batch wrt the
        number of timesteps (default is 1 for static datasets) unless
        statistics are accumulated over the entire epoch

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information

        """
        if not self.accumulate_over_epoch:
            for k, v in state.batch_score.items():
                self.batch_metrics[k].append(
                    v.item() / self.timesteps_in_batch
                )

        self.timesteps_in_batch = None

    def on_training_epoch_end(self, state: State):
        """
        For each score, computes the epoch scores using the same
        ogic as the superclass

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_over_epoch:
            epoch_res = {
                s.name: torch.tensor(self.batch_metrics[s.name]).sum()
                / len(self.batch_metrics[s.name])
                for s in self.scores
            }
        else:
            epoch_res = {
                s.name: s.compute_metric(
                    torch.cat(self.y_true[s.name], dim=0),
                    torch.cat(self.y_pred[s.name], dim=0),
                )
                for s in self.scores
            }

        state.update(epoch_score=epoch_res)

        self.batch_metrics = None
        self.y_pred, self.y_true = None, None

    def on_eval_epoch_start(self, state: State):
        """
        Compared to superclass version, initializes a dictionary for each
        score to track rather than single lists

         Args:
             state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.batch_metrics = {s.name: [] for s in self.scores}
        self.y_pred = {s.name: [] for s in self.scores}
        self.y_true = {s.name: [] for s in self.scores}

    def on_eval_batch_end(self, state: State):
        """
        For each scorer, computes the average metric in the batch wrt the
        number of timesteps (default is 1 for static datasets) unless
        statistics are accumulated over the entire epoch

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_over_epoch:
            for k, v in state.batch_score.items():
                self.batch_metrics[k].append(
                    v.item() / self.timesteps_in_batch
                )

        self.timesteps_in_batch = None

    def on_eval_epoch_end(self, state: State):
        """
        For each score, computes the epoch scores using the
        same logic as the superclass

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_over_epoch:
            epoch_res = {
                s.name: torch.tensor(self.batch_metrics[s.name]).sum()
                / len(self.batch_metrics[s.name])
                for s in self.scores
            }
        else:
            epoch_res = {
                s.name: s.compute_metric(
                    torch.cat(self.y_true[s.name], dim=0),
                    torch.cat(self.y_pred[s.name], dim=0),
                )
                for s in self.scores
            }

        state.update(epoch_score=epoch_res)

        self.batch_metrics = None
        self.y_pred, self.y_true = None, None

    def accumulate_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> None:
        """
        Accumulates predictions and targets of the batch into a list for
        each scorer, so as to compute aggregated statistics at the
        end of an epoch.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        for s in self.scores:
            y_pred_batch, y_true_batch = s.get_predictions_and_targets(
                targets, *outputs
            )
            metric_name = s.name

            self.y_pred[metric_name].append(
                y_pred_batch.detach().cpu()
                if self.force_cpu
                else y_pred_batch.detach()
            )
            self.y_true[metric_name].append(
                y_true_batch.detach().cpu()
                if self.force_cpu
                else y_true_batch.detach()
            )

    def forward(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Union[dict, float]:
        """
        For each scorer, it computes a score and returns them into a dictionary

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        res = {}
        for s in self.scores:
            y_pred_batch, y_true_batch = s.get_predictions_and_targets(
                targets, *outputs
            )
            res.update({s.name: s.compute_metric(y_true_batch, y_pred_batch)})
        return res


class AdditiveLoss(Metric):
    r"""
    AdditiveLoss sums an arbitrary number of losses together.

    Args:
        use_as_loss (bool): whether this metric should act as a loss
            (i.e., it should act when :func:`on_backward` is called).
            **Used by PyDGN, no need to care about this.**
        reduction (str): the type of reduction to apply across samples of the
            mini-batch. Supports ``mean`` and ``sum``. Default is ``mean``.
        accumulate_over_epoch (bool): Whether or not to display the epoch-wise
            metric rather than an average of per-batch metrics. If true,
            it keeps a list of predictions and target values across the
            entire epoch. Use it especially with batch-sensitive metrics,
            such as micro AP/F1 scores. Default is ``True``.
        force_cpu (bool): Whether or not to move all predictions to cpu
            before computing the epoch-wise loss/score. Default is ``True``.
        device (bool): The device used. Default is 'cpu'.
        losses_weights: (dict): dictionary of (loss_name, loss_weight) that
            specifies the weight to apply to each loss to be summed.
        losses (dict): dictionary of metrics to add together
    """

    def __init__(
        self,
        use_as_loss,
        reduction="mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        losses_weights: dict = None,
        **losses: dict,
    ):
        assert use_as_loss, "Additive loss can only be used as a loss"
        super().__init__(
            use_as_loss, reduction, accumulate_over_epoch, force_cpu, device
        )

        self.losses_weights = losses_weights
        self.losses = [
            self._instantiate_loss(loss) for loss in losses.values()
        ]

        if self.losses_weights is not None:
            for loss in self.losses:
                # check that a weight exists for each loss
                assert loss.name in self.losses_weights, (
                    "You have to specify a weight for each loss! "
                    f"We could not find the weight for {loss.name} "
                    f"in the dict."
                )
        else:
            # all losses are simply added together
            self.losses_weights = {loss.name: 1.0 for loss in self.losses}

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Additive Loss"

    def _instantiate_loss(self, loss):
        """
        Istantiate a loss with its own arguments (if any are given)
        """
        if isinstance(loss, dict):
            args = loss[ARGS]
            return s2c(loss[CLASS_NAME])(use_as_loss=True, **args)
        else:
            return s2c(loss)(use_as_loss=True)

    def on_training_epoch_start(self, state: State):
        """
        Instantiates a dictionary with one list per loss (including itself,
        representing the sum of all losses)

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.batch_metrics = {loss.name: [] for loss in [self] + self.losses}
        self.y_pred = {loss.name: [] for loss in [self] + self.losses}
        self.y_true = {loss.name: [] for loss in [self] + self.losses}

    def on_training_batch_end(self, state: State):
        """
        For each loss, computes the average metric in the batch wrt the number
        of timesteps (default is 1 for static datasets)
        unless statistics are accumulated over the entire epoch

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_over_epoch:
            for k, v in state.batch_loss.items():
                self.batch_metrics[k].append(
                    v.item() / self.timesteps_in_batch
                )

        self.timesteps_in_batch = None

    def on_training_epoch_end(self, state: State):
        """
        Computes an averaged or aggregated loss across the entire epoch,
        including itself as the main loss. Updates the field `epoch_loss`
        in state.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_over_epoch:
            epoch_res = {
                loss.name: torch.tensor(self.batch_metrics[loss.name]).sum()
                / len(self.batch_metrics[loss.name])
                for loss in [self] + self.losses
            }
        else:
            epoch_res = {
                loss.name: loss.compute_metric(
                    torch.cat(self.y_true[loss.name], dim=0),
                    torch.cat(self.y_pred[loss.name], dim=0),
                )
                * self.losses_weights[loss.name]
                for loss in self.losses
            }
            additive_epoch_loss = 0.0
            for loss in self.losses:
                additive_epoch_loss += epoch_res[loss.name]
            epoch_res[self.name] = additive_epoch_loss

        state.update(epoch_loss=epoch_res)

        self.batch_metrics = None
        self.y_pred, self.y_true = None, None

    def on_eval_epoch_start(self, state: State):
        """
        Instantiates a dictionary with one list per loss (including itself,
        representing the sum of all losses)

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.batch_metrics = {loss.name: [] for loss in [self] + self.losses}
        self.y_pred = {loss.name: [] for loss in [self] + self.losses}
        self.y_true = {loss.name: [] for loss in [self] + self.losses}

    def on_eval_epoch_end(self, state: State):
        """
        Computes an averaged or aggregated loss across the entire epoch,
        including itself as the main loss. Updates the field `epoch_loss`
        in state.

        Args:
            state (:class:`~training.event.state.State`):
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_over_epoch:
            epoch_res = {
                loss.name: torch.tensor(self.batch_metrics[loss.name]).sum()
                / len(self.batch_metrics[loss.name])
                for loss in [self] + self.losses
            }
        else:
            epoch_res = {
                loss.name: loss.compute_metric(
                    torch.cat(self.y_true[loss.name], dim=0),
                    torch.cat(self.y_pred[loss.name], dim=0),
                )
                * self.losses_weights[loss.name]
                for loss in self.losses
            }
            additive_epoch_loss = 0.0
            for loss in self.losses:
                additive_epoch_loss += epoch_res[loss.name]
            epoch_res[self.name] = additive_epoch_loss

        state.update(epoch_loss=epoch_res)

        self.batch_metrics = None
        self.y_pred, self.y_true = None, None

    def on_eval_batch_end(self, state: State):
        """
        For each loss, computes the average metric in the batch wrt the number
        of timesteps (default is 1 for static datasets)
        unless statistics are accumulated over the entire epoch

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if not self.accumulate_over_epoch:
            for k, v in state.batch_loss.items():
                self.batch_metrics[k].append(
                    v.item() / self.timesteps_in_batch
                )

        self.timesteps_in_batch = None

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Sums the value of all different losses into one

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        loss_sum = 0.0
        for loss in self.losses:
            single_loss = (
                loss.compute_metric(targets, predictions)
                * self.losses_weights[loss.name]
            )
            loss_sum += single_loss
        return loss_sum

    def accumulate_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> None:
        """
        Accumulates predictions and targets of the batch into a list
        for each loss, so as to compute aggregated statistics at the end
        of an epoch.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        for loss in self.losses:
            y_pred_batch, y_true_batch = loss.get_predictions_and_targets(
                targets, *outputs
            )
            metric_name = loss.name

            self.y_pred[metric_name].append(
                y_pred_batch.detach().cpu()
                if self.force_cpu
                else y_pred_batch.detach()
            )
            self.y_true[metric_name].append(
                y_true_batch.detach().cpu()
                if self.force_cpu
                else y_true_batch.detach()
            )

    def forward(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> dict:
        """
        For each scorer, it computes a loss and returns them into a dictionary,
        alongside the sum of all losses.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        res = {}
        loss_sum = 0.0

        for loss in self.losses:
            y_pred_batch, y_true_batch = loss.get_predictions_and_targets(
                targets, *outputs
            )
            single_loss = (
                loss.compute_metric(y_true_batch, y_pred_batch)
                * self.losses_weights[loss.name]
            )

            res[loss.name] = single_loss
            loss_sum += single_loss

        res[self.name] = loss_sum
        return res


class Classification(Metric):
    r"""
    Generic metric for classification tasks. Used to maximize code reuse
    for classical metrics.
    """

    def __init__(
        self,
        use_as_loss=False,
        reduction="mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
        )
        self.metric = None

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Classification Metric"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns output[0] as predictions and dataset targets. Squeezes the
        first dimension of output and targets to get single vectors.

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        outputs = outputs[0].squeeze(dim=1)

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        return outputs, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Applies a classification metric
        (to be subclassed as it is None in this class)

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        metric = self.metric(predictions, targets)
        return metric


class Regression(Metric):
    r"""
    Generic metric for regression tasks. Used to maximize code reuse
    for classical metrics.
    """

    def __init__(
        self,
        use_as_loss=False,
        reduction="mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
        )
        self.metric = None

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Regression Metric"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns output[0] as predictions and dataset targets.
        Squeezes the first dimension of output and targets to get
        single vectors.

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        outputs = outputs[0].squeeze(dim=1)

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        return outputs, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Applies a regression metric
        (to be subclassed as it is None in this class)

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        metric = self.metric(predictions, targets)
        return metric


class MulticlassClassification(Classification):
    r"""
    Wrapper around :class:`torch.nn.CrossEntropyLoss`
    """

    def __init__(
        self,
        use_as_loss=False,
        reduction="mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
        )
        self.metric = CrossEntropyLoss(reduction=reduction)

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Multiclass Classification"


class MeanSquareError(Regression):
    r"""
    Wrapper around :class:`torch.nn.MSELoss`
    """

    def __init__(
        self,
        use_as_loss=False,
        reduction="mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
        )
        self.metric = MSELoss(reduction=reduction)

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Mean Square Error"


class MeanAverageError(Regression):
    r"""
    Wrapper around :class:`torch.nn.MSELoss`
    """

    def __init__(
        self,
        use_as_loss=False,
        reduction="mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
        )
        self.metric = L1Loss(reduction=reduction)

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Mean Average Error"


class DotProductLink(Metric):
    """
    Implements a dot product link prediction metric,
    as defined in https://arxiv.org/abs/1611.07308.
    """

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Dot Product Link Prediction"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uses node embeddings (outputs[1]) aand positive/negative edges
        (contained in targets by means of
        e.g.,
        a :obj:`~pydgn.data.provider.LinkPredictionSingleGraphDataProvider`)
        to return logits and target labels of an edge classification task.

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        node_embs = outputs[1]
        _, pos_edges, neg_edges = targets

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat(
            (torch.ones(pos_edges.shape[1]), torch.zeros(neg_edges.shape[1]))
        )

        # Taken from
        # rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)

        return link_logits, loss_target.to(link_logits.device)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Applies BCEWithLogits to link logits and targets.

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        metric = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets
        )
        return metric


class MulticlassAccuracy(Metric):
    """
    Implements multiclass classification accuracy.
    """

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Multiclass Accuracy"

    @staticmethod
    def _get_correct(output):
        """
        Returns the argmax of the output alongside dimension 1.
        """
        return torch.argmax(output, dim=1)

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes output[0] as predictions and computes a discrete class
        using argmax. Returns standard dataset targets as well. Squeezes
        the first dimension of output and targets to get single vectors.

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        pred = outputs[0]
        correct = self._get_correct(pred)

        if len(targets.shape) == 2:
            targets = targets.squeeze(dim=1)

        return correct, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Takes output[0] as predictions and computes a discrete class using
        argmax. Returns standard dataset targets as well. Squeezes the first
        dimension of output and targets to get single vectors.

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        metric = (
            100.0 * (predictions == targets).sum().float() / targets.size(0)
        )
        return metric


class ToyMetric(Metric):
    r"""
    Implements a toy metric.
    """

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Toy Metric"

    @staticmethod
    def _get_correct(output):
        """
        Returns the argmax of the output alongside dimension 1.
        """
        return torch.argmax(output, dim=1)

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns output[0] and dataset targets

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        return outputs[0], targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Computes a dummy score

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        metric = 100.0 * torch.ones(1)
        return metric


class ToyUnsupervisedMetric(Metric):
    r"""
    Implements a toy metric.
    """

    @property
    def name(self) -> str:
        """
        The name of the loss to be used in configuration files and displayed
        on Tensorboard
        """
        return "Toy Unsupervised Metric"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns output[0] and dataset targets

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        return outputs[0], targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        """
        Computes a dummy score

        Args:
            targets (:class:`torch.Tensor`): tensor of ground truth values
            predictions (:class:`torch.Tensor`):
                tensor of predictions of the model

        Returns:
            A tensor with the metric value
        """
        metric = (predictions * 0.0).mean()
        return metric
