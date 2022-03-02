import copy
import operator
from pathlib import Path

from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State
from pydgn.training.util import atomic_save


class EarlyStopper(EventHandler):
    """
    EarlyStopper is the main event handler for optimizers. Just create a subclass that implements an early stopping
    method.

    Args:
        monitor (str): the metric to monitor. The format is ``[TRAINING|VALIDATION]_[METRIC NAME]``, where
        ``TRAINING`` and ``VALIDATION`` are defined in ``pydgn.static``
        mode (str): can be ``MIN`` or ``MAX`` (as defined in ``pydgn.static``)
        checkpoint (bool): whether we are interested in the checkpoint of the "best" epoch or not
    """
    def __init__(self, monitor: str, mode: str, checkpoint: bool=False):
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

    def on_epoch_end(self, state: State):
        """
        At the end of an epoch, check that the validation score improves over the current best validation score.
        If so, store the necessary info in a dictionary and save it into the "best_epoch_results" property of the state.
        If it is time to stop, updates the stop_training field of the state.

        Args:
            state (:class:`~training.event.state.State`): object holding training information
        """
        # it is possible that we evaluate every `n` epochs
        if not (self.monitor in state.epoch_results[SCORES] or self.monitor in state.epoch_results[LOSSES]):
            return

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

    def stop(self, state: State, score_or_loss: str, metric: str) -> bool:
        """
        Returns true when the early stopping technique decides it is time to stop.

        Args:
            state (:class:`~training.event.state.State`): object holding training information
            score_or_loss (str): whether to monitor scores or losses
            metric (str): the metric to consider. The format is ``[TRAINING|VALIDATION]_[METRIC NAME]``, where
                          ``TRAINING`` and ``VALIDATION`` are defined in ``pydgn.static``

        Returns:
            a boolean specifying whether training should be stopped or not
        """
        raise NotImplementedError('Sublass EarlyStopper and implement this method!')


class PatienceEarlyStopper(EarlyStopper):
    """
    Early Stopper that implements patience

    Args:
        monitor (str): the metric to monitor. The format is ``[TRAINING|VALIDATION]_[METRIC NAME]``, where
        ``TRAINING`` and ``VALIDATION`` are defined in ``pydgn.static``
        mode (str): can be ``MIN`` or ``MAX`` (as defined in ``pydgn.static``)
        patience (int): the number of epochs of patience
        checkpoint (bool): whether we are interested in the checkpoint of the "best" epoch or not
    """

    def __init__(self, monitor, mode, patience=30, checkpoint=False):
        super().__init__(monitor, mode, checkpoint)
        self.patience = patience

    def stop(self, state, score_or_loss, metric):
        # do not start with patience until you have evaluated at least once
        if not hasattr(state, BEST_EPOCH_RESULTS):
            return False

        best_epoch = state.best_epoch_results[BEST_EPOCH]
        stop_training = (state.epoch - best_epoch) >= self.patience
        return stop_training
