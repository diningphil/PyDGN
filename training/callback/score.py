import torch
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score

from pydgn.experiment.util import s2c
from pydgn.static import *
from pydgn.training.event.handler import EventHandler


class Score(EventHandler):
    """
    Score is the main event handler for score metrics. Other scores can easily subclass by implementing the __call__
    method, though sometimes more complex implementations are required.
    """
    __name__ = 'score'

    def __init__(self):
        super().__init__()
        assert hasattr(self, '__name__')
        self.batch_scores = None
        self.num_samples = None
        self.current_set = None

    def get_main_score_name(self):
        """
        Used by the training engine to retrieve the main score (assuming multiple scores are used)
        :return: the main score name
        """
        return self.__name__

    def _score_fun(self, targets, *outputs):
        raise NotImplementedError('To be implemented by a subclass')

    def on_training_epoch_start(self, state):
        """
        Initializes the array with batches of score values
        :param state: the shared State object
        """
        self.batch_scores = []
        self.num_samples = 0

    def on_training_batch_end(self, state):
        """
        Updates the array of batch score
        :param state: the shared State object
        """
        self.batch_scores.append(state.batch_score[self.__name__].item() * state.batch_num_targets)
        self.num_samples += state.batch_num_targets

    def on_training_epoch_end(self, state):
        """
        Computes a score value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_score={self.__name__: torch.tensor(self.batch_scores).sum() / self.num_samples})
        self.batch_scores = None
        self.num_samples = None

    def on_eval_epoch_start(self, state):
        """
        Initializes the array with batches of score values
        :param state: the shared State object
        """
        self.batch_scores = []
        self.num_samples = 0

    def on_eval_epoch_end(self, state):
        """
        Computes a score value for the entire epoch
        :param state: the shared State object
        """
        state.update(epoch_score={self.__name__: torch.tensor(self.batch_scores).sum() / self.num_samples})
        self.batch_scores = None
        self.num_samples = None

    def on_eval_batch_end(self, state):
        """
        Updates the array of batch score
        :param state: the shared State object
        """
        self.batch_scores.append(state.batch_score[self.__name__].item() * state.batch_num_targets)
        self.num_samples += state.batch_num_targets

    def on_compute_metrics(self, state):
        """
        Computes the score
        :param state: the shared State object
        """
        self.current_set = state.set
        outputs, targets = state.batch_outputs, state.batch_targets
        score = self(targets, *outputs, batch_loss_extra=getattr(state, BATCH_LOSS_EXTRA, None))
        # Score is a dictionary with key-value pairs
        # we need to detach each score from the graph
        score = {k: v.detach().cpu() for k, v in score.items()}
        state.update(batch_score=score)

    def __call__(self, targets, *outputs, batch_loss_extra):
        """
        :param targets:
        :param outputs: a tuple of outputs returned by a model
        :param batch_loss_extra: optional loss extra vars to save compute time
        :return: dictionary with {score_name: score value}
        """
        score = self._score_fun(targets, *outputs, batch_loss_extra=batch_loss_extra)
        return {self.__name__: score}


class MultiScore(Score):
    __name__ = 'score'

    def _istantiate_scorer(self, scorer):
        if isinstance(scorer, dict):
            args = scorer[ARGS]
            return s2c(scorer[CLASS_NAME])(**args)
        else:
            return s2c(scorer)()

    def __init__(self, main_scorer, **extra_scorers):
        super().__init__()
        self.scorers = [self._istantiate_scorer(main_scorer)] + [self._istantiate_scorer(score) for score in
                                                                 extra_scorers.values()]

    def get_main_score_name(self):
        """
        Used by the training engine to retrieve the main score (assuming multiple scores are used)
        :return: the main score name
        """
        return self.scorers[0].get_main_score_name()

    def on_training_epoch_start(self, state):
        self.batch_scores = {s.__name__: [] for s in self.scorers}
        for scorer in self.scorers:
            scorer.on_training_epoch_start(state)

    def on_training_batch_end(self, state):
        for scorer in self.scorers:
            scorer.on_training_batch_end(state)

    def on_training_epoch_end(self, state):
        epoch_score = {}
        for scorer in self.scorers:
            # This will update the epoch_score variable in State
            scorer.on_training_epoch_end(state)
            epoch_score.update(state.epoch_score)
        state.update(epoch_score=epoch_score)

    def on_eval_epoch_start(self, state):
        for scorer in self.scorers:
            scorer.on_eval_epoch_start(state)

    def on_eval_batch_end(self, state):
        for scorer in self.scorers:
            scorer.on_eval_batch_end(state)

    def on_eval_epoch_end(self, state):
        epoch_score = {}
        for scorer in self.scorers:
            # This will update the epoch_score variable in State
            scorer.on_training_epoch_end(state)
            epoch_score.update(state.epoch_score)
        state.update(epoch_score=epoch_score)

    def on_compute_metrics(self, state):
        super().on_compute_metrics(state)
        for scorer in self.scorers:
            scorer.current_set = self.current_set

    def __call__(self, targets, *outputs, batch_loss_extra):
        res = {}
        for scorer in self.scorers:
            # each scorer __call__ method returns a dict
            res.update(scorer(targets, *outputs, batch_loss_extra=batch_loss_extra))
        return res


class RSquareScore(Score):
    __name__ = 'R2 Determination Coefficient'

    def __init__(self):
        super().__init__()
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

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        self.y = targets if self.y is None else torch.cat((self.y, targets), dim=0)
        self.pred = outputs[0] if self.pred is None else torch.cat((self.pred, outputs[0]), dim=0)

        # Minibatch R2 score (needed when used with MultiScore, but behaves differently)
        return torch.tensor([r2_score(targets.detach().cpu().numpy(), outputs[0].detach().cpu().numpy())])


class BinaryAccuracyScore(Score):
    __name__ = 'Binary Accuracy'

    def __init__(self):
        super().__init__()

    def _get_correct(self, output):
        return output > 0.5

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        pred = outputs[0]

        if len(pred.shape) > 1:
            assert len(pred.shape) == 2 and pred.shape[1] == 1
            pred = pred.squeeze()

        correct = self._get_correct(pred)
        return 100. * (correct == targets).sum().float() / targets.size(0)


class MeanAverageErrorScore(Score):
    __name__ = 'MAE Score'

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        pred = outputs[0]
        return self.loss(pred, targets)


class Toy1Score(BinaryAccuracyScore):
    __name__ = 'toy 1 accuracy'

    def __init__(self):
        super().__init__()

    def _get_correct(self, output):
        return torch.argmax(output, dim=1)

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return 100. * torch.ones(1)


class MulticlassAccuracyScore(BinaryAccuracyScore):
    __name__ = 'Multiclass Accuracy'

    def __init__(self):
        super().__init__()

    def _get_correct(self, output):
        return torch.argmax(output, dim=1)

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        pred = outputs[0]
        correct = self._get_correct(pred)
        return 100. * (correct == targets).sum().float() / targets.size(0)


class LikelihoodScore(Score):
    __name__ = 'True Log Likelihood'

    def __init__(self):
        super().__init__()

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return outputs[3]


class CGMMCompleteLikelihoodScore(Score):
    __name__ = 'Complete Log Likelihood'

    def __init__(self):
        super().__init__()

    def on_training_batch_end(self, state):
        self.batch_scores.append(state.batch_score[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def on_eval_epoch_end(self, state):
        state.update(epoch_score={self.__name__: torch.tensor(self.batch_scores).sum() / self.num_samples})
        self.batch_scores = None
        self.num_samples = None

    def on_eval_batch_end(self, state):
        self.batch_scores.append(state.batch_score[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return outputs[2]


class CGMMTrueLikelihoodScore(Score):
    __name__ = 'True Log Likelihood'

    def __init__(self):
        super().__init__()

    def on_training_batch_end(self, state):
        self.batch_scores.append(state.batch_score[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def on_eval_batch_end(self, state):
        self.batch_scores.append(state.batch_score[self.__name__].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return outputs[3]
