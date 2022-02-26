import copy

from pydgn.experiment.util import s2c
from pydgn.training.event.handler import EventHandler


class Optimizer(EventHandler):
    """
    Optimizer is the main event handler for optimizers. Just pass a PyTorch scheduler together with its arguments in the
    configuration file.
    """
    __name__ = 'optimizer'

    def __init__(self, model, optimizer_class_name, accumulate_gradients=False, **kwargs):
        super().__init__()
        self.optimizer = s2c(optimizer_class_name)(model.parameters(), **kwargs)
        self.accumulate_gradients = accumulate_gradients

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state
        :param state_dict: the optimizer state
        :return:
        """
        self.optimizer.load_state_dict(state_dict)

    def on_fit_start(self, state):
        """
        Load scheduler from state if any
        :param state: the shared State object
        """
        if state.optimizer_state is not None:
            self.optimizer.load_state_dict(state.optimizer_state)

    def on_training_epoch_start(self, state):
        """
        Zeroes the gradient at the start of each epoch if gradient needs to be accumulated
        :param state: the shared State object
        """
        if self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_start(self, state):
        """
        Zeroes the gradient at the start of each (mini-)batch if gradient does not need to be accumulated
        :param state: the shared State object
        """
        if not self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        """
        Optimized the model at the end of each (mini-)batch if gradient does not need to be accumulated
        :param state: the shared State object
        """
        if not self.accumulate_gradients:
            self.optimizer.step()

    def on_training_epoch_end(self, state):
        """
        Stores the optimizer at the end of each epoch. If gradient needs to be accumulated performs an optimization step
        :param state: the shared State object
        """
        if self.accumulate_gradients:
            self.optimizer.step()

    def on_epoch_end(self, state):
        """
        Stores the optimizer at the end of each epoch
        :param state: the shared State object
        """
        state.update(optimizer_state=copy.deepcopy(self.optimizer.state_dict()))
