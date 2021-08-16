import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss, L1Loss

from pydgn.experiment.util import s2c
from pydgn.static import *
from pydgn.training.event.handler import EventHandler


class Loss(nn.Module, EventHandler):
    """
    Loss is the main event handler for loss metrics. Other losses can easily subclass by implementing the forward
    method, though sometimes more complex implementations are required.
    """
    __name__ = "loss"
    op = operator.lt  # less than to determine improvement

    def __init__(self):
        super().__init__()
        self.batch_losses = None
        self.num_samples = None

    def get_main_loss_name(self):
        return self.__name__

    def on_training_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_samples = 0

    def on_training_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * state.batch_num_targets)
        self.num_samples += state.batch_num_targets

    def on_training_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss={self.__name__: torch.tensor(self.batch_losses).sum() / self.num_samples})
        self.batch_losses = None
        self.num_samples = None

    def on_eval_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_samples = 0

    def on_eval_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss={self.__name__: torch.tensor(self.batch_losses).sum() / self.num_samples})
        self.batch_losses = None

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss[self.__name__].item() * state.batch_num_targets)
        self.num_samples += state.batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
        outputs, targets = state.batch_outputs, state.batch_targets
        loss_output = self.forward(targets, *outputs)

        if isinstance(loss_output, tuple):
            # Allow loss to produce intermediate results that speed up
            # Score computation. Loss callback MUST occur before the score one.
            loss, extra = loss_output
            state.update(batch_loss_extra={self.__name__: extra})
        else:
            loss = loss_output
        state.update(batch_loss={self.__name__: loss})

    def on_backward(self, state):
        """
        Computes the gradient of the computation graph
        :param state: the shared State object
        """
        try:
            state.batch_loss[self.__name__].backward()
        except Exception as e:
            # Here we catch potential multiprocessing related issues
            # see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
            print(e)

    def forward(self, targets, *outputs):
        """
        Computes the loss for a batch of output/target valies
        :param targets:
        :param outputs: a tuple of outputs returned by a model
        :return: loss and accuracy values
        """
        raise NotImplementedError('To be implemented by a subclass')


class AdditiveLoss(Loss):
    """
    MultiLoss combines an arbitrary number of Loss objects to perform backprop without having to istantiate a new class.
    The final loss is formally defined as the sum of the individual losses.
    """
    __name__ = "Additive Loss"
    op = operator.lt  # less than to determine improvement

    def _istantiate_loss(self, loss):
        if isinstance(loss, dict):
            args = loss[ARGS]
            return s2c(loss[CLASS_NAME])(**args)
        else:
            return s2c(loss)()

    def __init__(self, **losses):
        super().__init__()
        self.losses = [self._istantiate_loss(loss) for loss in losses.values()]

    def on_training_epoch_start(self, state):
        self.batch_losses = {l.__name__: [] for l in [self] + self.losses}
        self.num_targets = 0

    def on_training_batch_end(self, state):
        for k, v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_training_epoch_end(self, state):
        state.update(epoch_loss={l.__name__: torch.tensor(self.batch_losses[l.__name__]).sum() / self.num_targets
                                 for l in [self] + self.losses})
        self.batch_losses = None
        self.num_targets = None

    def on_eval_epoch_start(self, state):
        self.batch_losses = {l.__name__: [] for l in [self] + self.losses}
        self.num_targets = 0

    def on_eval_epoch_end(self, state):
        state.update(epoch_loss={l.__name__: torch.tensor(self.batch_losses[l.__name__]).sum() / self.num_targets
                                 for l in [self] + self.losses})
        self.batch_losses = None
        self.num_targets = None

    def on_eval_batch_end(self, state):
        for k, v in state.batch_loss.items():
            self.batch_losses[k].append(v.item() * state.batch_num_targets)
        self.num_targets += state.batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the loss
        :param state: the shared State object
        """
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
                extra[l.__name__] = loss_extra
                state.update(batch_loss_extra=extra)
            else:
                loss_output = single_loss
            loss[l.__name__] = loss_output
            loss_sum += loss_output

        loss[self.__name__] = loss_sum
        state.update(batch_loss=loss)


class ClassificationLoss(Loss):
    __name__ = 'Classification Loss'

    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]

        # print(outputs.shape, targets.shape)

        loss = self.loss(outputs, targets)
        return loss


class RegressionLoss(Loss):
    __name__ = 'Regression Loss'

    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]
        loss = self.loss(outputs.squeeze(), targets.squeeze())
        return loss


class BinaryClassificationLoss(ClassificationLoss):
    __name__ = 'Binary Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)


class MulticlassClassificationLoss(ClassificationLoss):
    __name__ = 'Multiclass Classification Loss'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)


class MeanSquareErrorLoss(RegressionLoss):
    __name__ = 'MSE'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = MSELoss(reduction=reduction)


class MeanAverageErrorLoss(RegressionLoss):
    __name__ = 'MAE'

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = L1Loss(reduction=reduction)


class CGMMLoss(Loss):
    __name__ = 'CGMM Loss'

    def __init__(self):
        super().__init__()
        self.old_likelihood = -float('inf')
        self.new_likelihood = None

    def on_training_batch_end(self, state):
        self.batch_losses.append(state.batch_loss[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def on_training_epoch_end(self, state):
        super().on_training_epoch_end(state)

        if (state.epoch_loss[self.__name__].item() - self.old_likelihood) < 0:
            pass
            # tate.stop_training = True
        self.old_likelihood = state.epoch_loss[self.__name__].item()

    def on_eval_batch_end(self, state):
        self.batch_losses.append(state.batch_loss[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    # Simply ignore targets
    def forward(self, targets, *outputs):
        likelihood = outputs[2]
        return likelihood

    def on_backward(self, state):
        pass


class LinkPredictionLoss(Loss):
    __name__ = 'Link Prediction Loss'

    def __init__(self):
        super().__init__()

    def forward(self, targets, *outputs):
        node_embs = outputs[1]
        _, pos_edges, neg_edges = targets[0]

        loss_edge_index = torch.cat((pos_edges, neg_edges), dim=1)
        loss_target = torch.cat((torch.ones(pos_edges.shape[1]),
                                 torch.zeros(neg_edges.shape[1])))

        # Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/link_pred.py
        x_j = torch.index_select(node_embs, 0, loss_edge_index[0])
        x_i = torch.index_select(node_embs, 0, loss_edge_index[1])
        link_logits = torch.einsum("ef,ef->e", x_i, x_j)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, loss_target)
        return loss
