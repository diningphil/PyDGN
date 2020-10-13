import operator
import copy
from pathlib import Path

from training.util import atomic_save
from training.event.handler import EventHandler


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

        if 'min' in mode:
            self.operator = operator.le
        elif 'max' in mode:
            self.operator = operator.ge
        else:
            raise NotImplementedError('Mode not understood by early stopper.')

        assert 'test' not in monitor, "Do not apply early stopping to the test set!"

    def on_epoch_end(self, state):
        """
        At the end of an epoch, check that the validation score improves over the current best validation score.
        If so, store the necessary info in a dictionary and save it into the "best_epoch_results" property of the state.
        If it is time to stop, updates the stop_training field of the state.
        :param state: the State object that is shared by the TrainingEngine during training
        """

        assert self.monitor in state.epoch_results['scores'], f'{self.monitor} not found in epoch_results'

        metric_to_compare = state.epoch_results['scores'][self.monitor]

        if not hasattr(state, 'best_epoch_results'):
            state.update(best_epoch_results=state.epoch_results)
            state.best_epoch_results['best_epoch'] = state.epoch
            state.best_epoch_results['scores'][self.monitor] = state.epoch_results['scores'][self.monitor]
            state.best_epoch_results['model_state'] = copy.deepcopy(state.model.state_dict())
            state.best_epoch_results['optimizer_state'] = state.optimizer_state  # computed by optimizer
            state.best_epoch_results['scheduler_state'] = state['scheduler_state']  # computed by scheduler
            stop_training = False
            if self.checkpoint:
                atomic_save(state.best_epoch_results, Path(state.exp_path,
                           state.BEST_CHECKPOINT_FILENAME))
        else:
            best_metric = state.best_epoch_results['scores'][self.monitor]

            if self.operator(metric_to_compare, best_metric):
                state.update(best_epoch_results=state.epoch_results)
                state.best_epoch_results['best_epoch'] = state.epoch
                state.best_epoch_results['scores'][self.monitor] = metric_to_compare
                state.best_epoch_results['model_state'] = copy.deepcopy(state.model.state_dict())
                state.best_epoch_results['optimizer_state'] = state.optimizer_state  # computed by optimizer
                state.best_epoch_results['scheduler_state'] = state['scheduler_state']  # computed by scheduler
                if self.checkpoint:
                    atomic_save(state.best_epoch_results, Path(state.exp_path,
                               state.BEST_CHECKPOINT_FILENAME))

        # Regarless of improvement or not
        stop_training = self.stop(state, metric_to_compare)
        state.update(stop_training=stop_training)


    def stop(self, state, metric):
        """
        Returns true when the early stopping technique decides it is time to stop.
        :param state: the State object
        :param metric: the metric to consider
        :return:
        """
        raise NotImplementedError('Sublass EarlyStopper and implement this method!')


class PatienceEarlyStopper(EarlyStopper):
    """ Early Stopper that implements patience """
    def __init__(self, monitor, mode, patience=30, checkpoint=False):
        super().__init__(monitor, mode, checkpoint)
        self.patience = patience

    def stop(self, state, metric):
        """
        Returns true when the early stopping technique decides it is time to stop.
        :param state: the State object
        :param metric: the metric to consider
        :return: a boolean indicating whether to stop the training or not
        """
        best_epoch = state.best_epoch_results['best_epoch']
        stop_training = (state.epoch - best_epoch) >= self.patience
        return stop_training
