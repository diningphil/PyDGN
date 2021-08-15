import copy
import operator
from pathlib import Path

from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.util import atomic_save


class EarlyStopper(EventHandler):
    """
    EarlyStopper is the main event handler for optimizers. Just create a subclass that implements an early stopping
    method.
    """
    __name__ = 'early stopper'

    def __init__(self, monitor, mode, checkpoint=False):
        super().__init__()
        self.monitor = monitor
        self.best_metric = None
        self.checkpoint = checkpoint

        if MIN in mode:
            self.operator = operator.le
        elif MAX in mode:
            self.operator = operator.ge
        else:
            raise NotImplementedError('Mode not understood by early stopper.')

        assert TEST not in monitor, "Do not apply early stopping to the test set!"

    def on_epoch_end(self, state):
        """
        At the end of an epoch, check that the validation score improves over the current best validation score.
        If so, store the necessary info in a dictionary and save it into the "best_epoch_results" property of the state.
        If it is time to stop, updates the stop_training field of the state.
        :param state: the State object that is shared by the TrainingEngine during training
        """

        assert self.monitor in state.epoch_results[SCORES] or self.monitor in state.epoch_results[
            LOSSES], f'{self.monitor} not found in epoch_results'

        if self.monitor in state.epoch_results[SCORES]:
            score_or_loss = SCORES
        else:
            score_or_loss = LOSSES
        metric_to_compare = state.epoch_results[score_or_loss][self.monitor]

        if not hasattr(state, BEST_EPOCH_RESULTS):
            state.update(best_epoch_results=state.epoch_results)
            state.best_epoch_results[BEST_EPOCH] = state.epoch
            state.best_epoch_results[score_or_loss][self.monitor] = state.epoch_results[score_or_loss][self.monitor]
            state.best_epoch_results[MODEL_STATE] = copy.deepcopy(state.model.state_dict())
            state.best_epoch_results[OPTIMIZER_STATE] = state[OPTIMIZER_STATE]  # computed by optimizer
            state.best_epoch_results[SCHEDULER_STATE] = state[SCHEDULER_STATE]  # computed by scheduler
            if self.checkpoint:
                atomic_save(state.best_epoch_results, Path(state.exp_path, BEST_CHECKPOINT_FILENAME))
        else:
            best_metric = state.best_epoch_results[score_or_loss][self.monitor]

            if self.operator(metric_to_compare, best_metric):
                state.update(best_epoch_results=state.epoch_results)
                state.best_epoch_results[BEST_EPOCH] = state.epoch
                state.best_epoch_results[score_or_loss][self.monitor] = metric_to_compare
                state.best_epoch_results[MODEL_STATE] = copy.deepcopy(state.model.state_dict())
                state.best_epoch_results[OPTIMIZER_STATE] = state[OPTIMIZER_STATE]  # computed by optimizer
                state.best_epoch_results[SCHEDULER_STATE] = state[SCHEDULER_STATE]  # computed by scheduler
                if self.checkpoint:
                    atomic_save(state.best_epoch_results, Path(state.exp_path, BEST_CHECKPOINT_FILENAME))

        # Regarless of improvement or not
        stop_training = self.stop(state, score_or_loss, metric_to_compare)
        state.update(stop_training=stop_training)

    def stop(self, state, score_or_loss, metric):
        """
        Returns true when the early stopping technique decides it is time to stop.
        :param state: the State object
        :param score_or_loss: whether to check in scores or losses
        :param metric: the metric to consider
        :return:
        """
        raise NotImplementedError('Sublass EarlyStopper and implement this method!')


class PatienceEarlyStopper(EarlyStopper):
    """ Early Stopper that implements patience """

    def __init__(self, monitor, mode, patience=30, checkpoint=False):
        super().__init__(monitor, mode, checkpoint)
        self.patience = patience

    def stop(self, state, score_or_loss, metric):
        best_epoch = state.best_epoch_results[BEST_EPOCH]
        stop_training = (state.epoch - best_epoch) >= self.patience
        return stop_training
