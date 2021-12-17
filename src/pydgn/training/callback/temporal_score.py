import torch
from pydgn.experiment.util import s2c
from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.callback.score import Score
from sklearn.metrics import r2_score


class TemporalScore(Score):
    """
    Score is the main event handler for temporal score metrics.
    Other scores can easily subclass by implementing the __call__
    method, though sometimes more complex implementations are required.
    In this case, we assume that on_compute_metrics is called at each snapshot
    in the mini-batch, so we must accumulate the score and the number of targets
    seen across different snapshots.
    """
    __name__ = 'Temporal Score'

    def _handle_reduction(self, state):
        if self.reduction == 'mean':
            # Used to recover the "sum" version of the loss
            return state.batch_num_targets
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
        # Reset batch_score value ow the computational graph gets retained
        state.update(batch_score={self.__name__: 0.})

    def on_eval_batch_start(self, state):
        """
        Updates the array of batch losses
        :param state: the shared State object
        """
        self.cumulative_batch_num_targets = 0
        # Reset batch_score value ow the computational graph gets retained
        state.update(batch_score={self.__name__: 0.})

    def on_training_batch_end(self, state):
        """
        Updates the array of batch score
        :param state: the shared State object
        """
        self.batch_scores.append(state.batch_score[self.__name__].item() * self._handle_reduction(state))
        self.num_samples += self.cumulative_batch_num_targets

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch score
        :param state: the shared State object
        """
        self.batch_scores.append(state.batch_score[self.__name__].item() * self._handle_reduction(state))
        self.num_samples += self.cumulative_batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the score
        :param state: the shared State object
        """
        self.current_set = state.set
        outputs, targets = state.batch_outputs, state.batch_targets
        if outputs[0] is None:
            return

        score = self(targets, *outputs)

        # Score is a dictionary with key-value pairs
        # we need to detach each score from the graph
        score = {k: v.detach().cpu() + state.batch_score[k] for k, v in score.items()}
        state.update(batch_score=score)
        self.cumulative_batch_num_targets += state.batch_num_targets

    def __call__(self, targets, *outputs):
        """
        :param targets:
        :param outputs: a tuple of outputs returned by a model
        :return: dictionary with {score_name: score value}
        """
        score = self._score_fun(targets, *outputs)
        return {self.__name__: score}


# class MultiScore(TemporalScore):
#     __name__ = 'Temporal Multi Score'
#
#     def _istantiate_scorer(self, scorer):
#         if isinstance(scorer, dict):
#             args = scorer[ARGS]
#             return s2c(scorer[CLASS_NAME])(**args)
#         else:
#             return s2c(scorer)()
#
#     def __init__(self, main_scorer, **extra_scorers):
#         super().__init__()
#         self.scorers = [self._istantiate_scorer(main_scorer)] + [self._istantiate_scorer(score) for score in
#                                                                  extra_scorers.values()]
#
#     def get_main_score_name(self):
#         """
#         Used by the training engine to retrieve the main score (assuming multiple scores are used)
#         :return: the main score name
#         """
#         return self.scorers[0].get_main_score_name()
#
#     def on_training_epoch_start(self, state):
#         self.batch_scores = {s.__name__: [] for s in self.scorers}
#         for scorer in self.scorers:
#             scorer.on_training_epoch_start(state)
#
#     def on_training_batch_end(self, state):
#         for scorer in self.scorers:
#             scorer.on_training_batch_end(state)
#
#     def on_training_epoch_end(self, state):
#         epoch_score = {}
#         for scorer in self.scorers:
#             # This will update the epoch_score variable in State
#             scorer.on_training_epoch_end(state)
#             epoch_score.update(state.epoch_score)
#         state.update(epoch_score=epoch_score)
#
#     def on_eval_epoch_start(self, state):
#         for scorer in self.scorers:
#             scorer.on_eval_epoch_start(state)
#
#     def on_eval_batch_end(self, state):
#         for scorer in self.scorers:
#             scorer.on_eval_batch_end(state)
#
#     def on_eval_epoch_end(self, state):
#         epoch_score = {}
#         for scorer in self.scorers:
#             # This will update the epoch_score variable in State
#             scorer.on_training_epoch_end(state)
#             epoch_score.update(state.epoch_score)
#         state.update(epoch_score=epoch_score)
#
#     def on_compute_metrics(self, state):
#         super().on_compute_metrics(state)
#         for scorer in self.scorers:
#             scorer.current_set = self.current_set
#
#     def __call__(self, targets, *outputs):
#         res = {}
#         for scorer in self.scorers:
#             # each scorer __call__ method returns a dict
#             res.update(scorer(targets, *outputs))
#         return res
#

class RSquareScore(TemporalScore):
    __name__ = 'R2 Determination Coefficient'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.y = None
        self.pred = None

    def on_training_epoch_start(self, state):
        super().on_training_epoch_start(state)
        self.y = None
        self.pred = None

    def on_training_epoch_end(self, state):
        state.update(
            epoch_score={self.__name__: r2_score(self.y.detach().cpu().numpy(), self.pred.detach().cpu().numpy())})
        self.batch_scores = None
        self.num_samples = None
        self.y = None
        self.pred = None

    def on_eval_epoch_start(self, state):
        super().on_eval_epoch_start(state)
        self.y = None
        self.pred = None

    def on_eval_epoch_end(self, state):
        state.update(
            epoch_score={self.__name__: r2_score(self.y.detach().cpu().numpy(), self.pred.detach().cpu().numpy())})
        self.batch_scores = None
        self.num_samples = None
        self.y = None
        self.pred = None

    def _score_fun(self, targets, *outputs):
        self.y = targets if self.y is None else torch.cat((self.y, targets), dim=0)
        self.pred = outputs[0] if self.pred is None else torch.cat((self.pred, outputs[0]), dim=0)

        # Minibatch R2 score (needed when used with MultiScore, but behaves differently)
        return torch.tensor([r2_score(targets.detach().cpu().numpy(), outputs[0].detach().cpu().numpy())])


class BinaryAccuracyScore(TemporalScore):
    __name__ = 'Binary Accuracy'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def _get_correct(self, output):
        return output > 0.5

    def _score_fun(self, targets, *outputs):
        pred = outputs[0]

        if len(pred.shape) > 1:
            assert len(pred.shape) == 2 and pred.shape[1] == 1
            pred = pred.squeeze()

        correct = self._get_correct(pred)
        return 100. * (correct == targets).sum().float() / targets.size(0)


class MeanAverageErrorScore(TemporalScore):
    __name__ = 'MAE Score'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = torch.nn.L1Loss(reduction=reduction)

    def _score_fun(self, targets, *outputs):
        pred = outputs[0]
        return self.loss(pred.squeeze(), targets.squeeze())


class MeanSquareErrorScore(TemporalScore):
    __name__ = 'MSE Score'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = torch.nn.MSELoss(reduction=reduction)

    def _score_fun(self, targets, *outputs):
        outputs = outputs[0]
        return self.loss(outputs.squeeze(), targets.squeeze())


class Toy1Score(BinaryAccuracyScore):
    __name__ = 'toy 1 accuracy'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def _get_correct(self, output):
        return torch.argmax(output, dim=1)

    def _score_fun(self, targets, *outputs):
        return 100. * torch.ones(1)


class MulticlassAccuracyScore(BinaryAccuracyScore):
    __name__ = 'Multiclass Accuracy'

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def _get_correct(self, output):
        return torch.argmax(output, dim=1)

    def _score_fun(self, targets, *outputs):
        pred = outputs[0]
        correct = self._get_correct(pred)
        return 100. * (correct == targets).sum().float() / targets.size(0)
