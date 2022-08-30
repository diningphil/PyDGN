from torch.optim.optimizer import Optimizer

from pydgn.experiment.util import s2c
from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State


class Scheduler(EventHandler):
    """
    Scheduler is the main event handler for schedulers. Just pass a PyTorch
    scheduler together with its arguments in the configuration file.

    Args:
        scheduler_class_name (str): dotted path to class name of the scheduler
        optimizer (:class:`torch.optim.optimizer`): the Pytorch optimizer to
            use. **This is automatically recovered by PyDGN when
            providing an optimizer**
        kwargs: additional parameters for the specific scheduler to be used

    """

    def __init__(
        self, scheduler_class_name: str, optimizer: Optimizer, **kwargs: dict
    ):
        self.scheduler = s2c(scheduler_class_name)(optimizer, **kwargs)

    def on_fit_start(self, state: State):
        """
        Loads the scheduler state if already present in the state_dict
        of a checkpoint

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        if self.scheduler and state.scheduler_state is not None:
            self.scheduler.load_state_dict(state.scheduler_state)

    def on_epoch_end(self, state: State):
        """
        Updates the scheduler state with the current one for checkpointing

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        state.update(scheduler_state=self.scheduler.state_dict())


class EpochScheduler(Scheduler):
    """
    Implements a scheduler which uses epochs to modify the step size
    """

    def on_training_epoch_end(self, state: State):
        """
        Performs a scheduler's step at the end of the training epoch.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.scheduler.step()


class MetricScheduler(Scheduler):
    """
    Implements a scheduler which uses variations in the metric of interest
    to modify the step size

    Args:
        scheduler_class_name (str): dotted path to class name of the scheduler
        use_loss (str): whether to monitor scores or losses
        monitor (str): the metric to monitor. The format is
            ``[TRAINING|VALIDATION]_[METRIC NAME]``, where
            ``TRAINING`` and ``VALIDATION`` are defined in ``pydgn.static``
        optimizer (:class:`torch.optim.optimizer`): the Pytorch optimizer
            to use. **This is automatically recovered by PyDGN when
            providing an optimizer**
        kwargs: additional parameters for the specific scheduler to be used
    """

    def __init__(
        self,
        scheduler_class_name: str,
        use_loss: bool,
        monitor: str,
        optimizer: Optimizer,
        **kwargs: dict,
    ):
        self.scheduler = s2c(scheduler_class_name)(optimizer, **kwargs)
        self.use_loss = use_loss
        self.monitor = monitor

    def on_epoch_end(self, state: State):
        """
        Updates the state of the scheduler according to a metric to monitor
        at each epoch. Finally, loads the scheduler state if already present
        in the state_dict of a checkpoint

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """

        monitor_main_key = LOSSES if self.use_loss else SCORES
        assert (
            self.monitor in state.epoch_results[monitor_main_key]
        ), f"{self.monitor} not found in epoch_results"
        metric = state.epoch_results[monitor_main_key][self.monitor]
        self.scheduler.step(metric)

        # Do not forget to store dict
        super().on_epoch_end(state)
