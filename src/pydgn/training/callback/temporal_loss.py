import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydgn.experiment.util import s2c
from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.callback.loss import Loss, AdditiveLoss
from torch.nn.modules.loss import MSELoss, L1Loss


class TemporalLoss(Loss):
    """
    Temporal Loss is the main event handler for temporal loss metrics.
    Other losses can easily subclass by implementing the forward
    method, though sometimes more complex implementations are required.
    In this case, we assume that on_compute_metrics is called at each snapshot
    in the mini-batch, so we must accumulate the loss and the number of targets
    seen across different snapshots.
    """
    __name__ = "Temporal Loss"
    op = operator.lt  # less than to determine improvement

    def _handle_reduction(self, state):
        if self.reduction == 'mean':
            # Used to recover the "sum" version of the loss
            return self.cumulative_batch_num_targets
        elif self.reduction == 'sum':
            return 1
        else:
            raise NotImplementedError('The only reductions allowed are sum and mean')

    def on_training_batch_start(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.cumulative_batch_num_targets = 0
        # Reset batch_loss value ow the computational graph gets retained
        state.update(batch_loss={self.__name__: 0.})


    def on_eval_batch_start(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.cumulative_batch_num_targets = 0
        # Reset batch_loss value ow the computational graph gets retained
        state.update(batch_loss={self.__name__: 0.})

    def on_training_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * self._handle_reduction(state))
        self.num_samples += self.cumulative_batch_num_targets

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * self._handle_reduction(state))
        self.num_samples += self.cumulative_batch_num_targets


    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        outputs, targets = state.batch_outputs, state.batch_targets
        if outputs[0] is None:
            return

        loss = self.forward(targets, *outputs)

        # In the temporal scenario, on_compute_metrics is called at each
        # snapshot. so we need to accumulate the loss across multiple snapshots
        cumulative_batch_loss = state.batch_loss.get(self.__name__) + loss
        state.update(batch_loss={self.__name__: cumulative_batch_loss})
        self.cumulative_batch_num_targets += state.batch_num_targets


class TemporalAdditiveLoss(AdditiveLoss):
    """
    TemporalAdditiveLoss combines an arbitrary number of TemporalLoss objects
    to perform backprop without having to istantiate a new class.
    The final loss is formally defined as the sum of the individual losses.
    """
    __name__ = "Temporal Additive Loss"
    op = operator.lt  # less than to determine improvement

    def on_training_batch_start(self, state):
        self.cumulative_batch_num_targets = 0
        # Reset batch_loss value ow the computational graph gets retained
        state.update(batch_loss={self.__name__: 0.})

    def on_eval_batch_start(self, state):
        self.cumulative_batch_num_targets = 0
        # Reset batch_loss value ow the computational graph gets retained
        state.update(batch_loss={self.__name__: 0.})

    def on_training_batch_end(self, state):
        for k, v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * self._handle_reduction(state))
        self.num_samples += self.cumulative_batch_num_targets

    def on_eval_batch_end(self, state):
        for k, v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * self._handle_reduction(state))
        self.num_samples += self.cumulative_batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        outputs, targets = state.batch_outputs, state.batch_targets
        if outputs[0] is None:
            return

        loss = {}
        loss_sum = state.batch_loss.get(self.__name__, 0.)
        for l in self.losses:
            single_loss = l.forward(targets, *outputs)
            loss_output = single_loss

            # In the temporal scenario, on_compute_metrics is called at each
            # snapshot. so we need to accumulate the loss across multiple snapshots
            cumulative_batch_loss = state.batch_loss.get(l.__name__, 0.).item() + loss_output.item()
            loss[l.__name__] = cumulative_batch_loss
            loss_sum += loss_output

        loss[self.__name__] = loss_sum
        state.update(batch_loss=loss)


class ClassificationLoss(TemporalLoss):
    __name__ = 'Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]
        loss = self.loss(outputs, targets)
        return loss


class RegressionLoss(TemporalLoss):
    __name__ = 'Regression Loss'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]
        loss = self.loss(outputs.squeeze(), targets.squeeze())
        return loss


class BinaryClassificationLoss(ClassificationLoss):
    __name__ = 'Binary Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)


class MulticlassClassificationLoss(ClassificationLoss):
    __name__ = 'Multiclass Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = nn.CrossEntropyLoss(reduction=reduction)


class MeanSquareErrorLoss(RegressionLoss):
    __name__ = 'MSE'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = MSELoss(reduction=reduction)


class MeanAverageErrorLoss(RegressionLoss):
    __name__ = 'MAE'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = L1Loss(reduction=reduction)
