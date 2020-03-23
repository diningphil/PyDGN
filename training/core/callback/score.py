import operator
import torch
from sklearn.metrics import r2_score

from training.core.event.handler import EventHandler


class Score(EventHandler):
    """
    Score is the main event handler for score metrics. Other scores can easily subclass by implementing the __call__
    method, though sometimes more complex implementations are required.
    """
    def __init__(self):
        assert hasattr(self, 'name')
        self.batch_scores = None
        self.num_graphs = None

    def on_training_epoch_start(self, state):
        """
        Initializes the array with batches of score values
        :param state: the shared State object
        """
        self.batch_scores = []
        self.num_graphs = 0

    def on_training_batch_end(self, state):
        """
        Updates the array of batch score
        :param state: the shared State object
        """
        self.batch_scores.append(state.batch_score.item() * state.batch_num_graphs)
        self.num_graphs += state.batch_num_graphs

    def on_training_epoch_end(self, state):
        """
        Computes a score value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_score=torch.tensor(self.batch_scores).sum()/self.num_graphs)
        self.batch_scores = None
        self.num_graphs = None

    def on_eval_epoch_start(self, state):
        """
        Initializes the array with batches of score values
        :param state: the shared State object
        """
        self.batch_scores = []
        self.num_graphs = 0

    def on_eval_epoch_end(self, state):
        """
        Computes a score value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_score=torch.tensor(self.batch_scores).sum()/self.num_graphs)
        self.batch_scores = None
        self.num_graphs = None

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch score
        :param state: the shared State object
        """
        self.batch_scores.append(state.batch_score.item() * state.batch_num_graphs)
        self.num_graphs += state.batch_num_graphs

    def __call__(self, targets, *outputs):
        """
        :param targets:
        :param outputs: a tuple of outputs returned by a model
        :return: score value
        """
        raise NotImplementedError('To be implemented by a subclass')


class PerformanceMetric(Score):
    name = 'performance metric'

    def __init__(self):
        assert hasattr(self, 'op')
        super().__init__()

    @property
    def greater_is_better(self):
        return self.op == operator.gt


class RSquareScore(PerformanceMetric):
    name = 'Determination Coefficient'
    op = operator.gt

    def __init__(self):
        super().__init__()

    def on_training_epoch_start(self, state):
        super().on_training_epoch_start(state)
        self.y = None
        self.pred = None

    def on_training_epoch_end(self, state):
        state.update(epoch_score=r2_score(self.y.detach().cpu().numpy(), self.pred.detach().cpu().numpy()))
        self.batch_scores = None
        self.num_graphs = None
        self.y = None
        self.pred = None

    def on_eval_epoch_start(self, state):
        super().on_eval_epoch_start(state)
        self.y = None
        self.pred = None

    def on_eval_epoch_end(self, state):
        state.update(epoch_score=r2_score(self.y.detach().cpu().numpy(), self.pred.detach().cpu().numpy()))
        self.batch_scores = None
        self.num_graphs = None
        self.y = None
        self.pred = None

    def __call__(self, targets, *outputs):
        self.y = targets if self.y is None else torch.cat((self.y, targets), dim=0)
        self.pred = outputs[0] if self.pred is None else torch.cat((self.pred, outputs[0]), dim=0)

        # Return dummy score
        return torch.tensor([0.])


class BinaryAccuracyScore(PerformanceMetric):
    name = 'binary accuracy'
    op = operator.gt

    def __init__(self):
        super().__init__()

    def _get_correct(self, output):
        return output > 0.5

    def __call__(self, targets, *outputs):
        pred = outputs[0]
        correct = self._get_correct(pred)
        return 100. * (correct == targets).sum().float() / targets.size(0)


class MulticlassAccuracyScore(BinaryAccuracyScore):
    name = 'multiclass accuracy'
    op = operator.gt

    def __init__(self):
        super().__init__()

    def _get_correct(self, output):
        return torch.argmax(output, dim=1)

    def __call__(self, targets, *outputs):
        pred = outputs[0]
        correct = self._get_correct(pred)
        return 100. * (correct == targets).sum().float() / targets.size(0)