import copy

from pydgn.experiment.util import s2c
from pydgn.model.interface import ModelInterface
from pydgn.training.event.handler import EventHandler


class Optimizer(EventHandler):
    """
    Optimizer is the main event handler for optimizers.
    Just pass a PyTorch optimizer together with its arguments in the configuration file.

    Args:
        model (:class:`~pydgn.model.interface.ModelInterface`): the model that has to be trained
        optimizer_class_name (str): dotted path to the optimizer class to use
        accumulate_gradients (bool): if ``True``, accumulate mini-batch gradients to perform a batch gradient update without loading the entire batch in memory
        kwargs (dict): additional parameters for the specific optimizer
    """
    def __init__(self,
                 model: ModelInterface,
                 optimizer_class_name: str,
                 accumulate_gradients: bool=False,
                 **kwargs: dict):
        super().__init__()
        self.optimizer = s2c(optimizer_class_name)(model.parameters(), **kwargs)
        self.accumulate_gradients = accumulate_gradients

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def on_fit_start(self, state):
        if state.optimizer_state is not None:
            self.optimizer.load_state_dict(state.optimizer_state)

    def on_training_epoch_start(self, state):
        if self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_start(self, state):
        if not self.accumulate_gradients:
            self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        if not self.accumulate_gradients:
            self.optimizer.step()

    def on_training_epoch_end(self, state):
        if self.accumulate_gradients:
            self.optimizer.step()

    def on_epoch_end(self, state):
        state.update(optimizer_state=copy.deepcopy(self.optimizer.state_dict()))
