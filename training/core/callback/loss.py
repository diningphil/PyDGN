import operator
import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss, L1Loss
from training.core.event.handler import EventHandler
from training.core.event.state import State


class Loss(nn.Module, EventHandler):
    """
    Loss is the main event handler for loss metrics. Other losses can easily subclass by implementing the forward
    method, though sometimes more complex implementations are required.
    """

    name = "loss"
    op = operator.lt  # less than to determine improvement

    def __init__(self):
        super().__init__()
        self.batch_losses = None
        self.num_graphs = None

    def on_training_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_graphs = 0

    def on_training_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss.item() * state.batch_num_graphs)
        self.num_graphs += state.batch_num_graphs

    def on_training_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss=torch.tensor(self.batch_losses).sum()/self.num_graphs)
        self.batch_losses = None
        self.num_graphs = None

    def on_eval_epoch_start(self, state):
        """
        Initializes the array with batches of loss values
        :param state: the shared State object
        """
        self.batch_losses = []
        self.num_graphs = 0

    def on_eval_epoch_end(self, state):
        """
        Computes a loss value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_loss=torch.tensor(self.batch_losses).sum()/self.num_graphs)
        self.batch_losses = None

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.batch_losses.append(state.batch_loss.item() * state.batch_num_graphs)
        self.num_graphs += state.batch_num_graphs

    def on_backward(self, state):
        """
        Computes the gradient of the computation graph
        :param state: the shared State object
        """
        assert state.mode == State.TRAINING
        state.batch_loss.backward()

    def forward(self, targets, *outputs):
        """
        Computes the loss for a batch of output/target valies
        :param targets:
        :param outputs: a tuple of outputs returned by a model
        :return: loss and accuracy values
        """
        raise NotImplementedError('To be implemented by a subclass')


class ClassificationLoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]
        loss = self.loss(outputs, targets)
        return loss


class RegressionLoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, targets, *outputs):
        outputs = outputs[0]
        loss = self.loss(outputs, targets)
        return loss


class BinaryClassificationLoss(ClassificationLoss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)


class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)


class LinkPredictionLoss(ClassificationLoss):

    # Simply ignore targets
    def forward(self, targets, *outputs):
        _, z, out, adj = outputs
        link_loss = adj - out
        link_loss = torch.norm(link_loss, p=2)
        # link_loss = link_loss / adj.numel()

        return link_loss


class MeanSquareErrorLoss(RegressionLoss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = MSELoss(reduction=reduction)


class MeanAverageErrorLoss(RegressionLoss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = L1Loss(reduction=reduction)


class CGMMLoss(Loss):

    def __init__(self):
        super().__init__()
        self.old_likelihood = -float('inf')
        self.new_likelihood = None
        self.training = None

    def on_training_epoch_start(self, state):
        super().on_training_epoch_start(state)
        self.training = True

    def on_training_epoch_end(self, state):
        super().on_training_epoch_end(state)
        self.training = False

    # Simply ignore targets
    def forward(self, targets, *outputs):  # IMPORTANT: This method assumes the batch size is the size of the dataset
        likelihood = outputs[0].detach()

        if self.training:
            self.new_likelihood = likelihood

        return likelihood

    def on_backward(self, state):
        assert state.mode == State.TRAINING
        pass
    
    def on_training_epoch_end(self, state):
        super().on_training_epoch_end(state)

        if (self.new_likelihood - self.old_likelihood) <= 0:
            state.stop_training = True
        self.old_likelihood = self.new_likelihood