from pydgn.experiment.util import s2c
from pydgn.static import *
from pydgn.training.event.handler import EventHandler


class Scheduler(EventHandler):
    """
    Scheduler is the main event handler for schedulers. Just pass a PyTorch scheduler together with its arguments in the
    configuration file.
    """
    __name__ = 'scheduler'

    def __init__(self, scheduler_class_name, optimizer, **kwargs):
        self.scheduler = s2c(scheduler_class_name)(optimizer, **kwargs)

    def on_fit_start(self, state):
        """
        Load scheduler from state if any
        :param state: the shared State object
        """
        if self.scheduler and SCHEDULER_STATE in state:
            self.scheduler.load_state_dict(state.scheduler_state)

    def on_epoch_end(self, state):
        """
        Save the scheduler state into the state object
        :param state: the shared State object
        """
        state.update(scheduler_state=self.scheduler.state_dict())


class EpochScheduler(Scheduler):
    __name__ = 'epoch-based scheduler'

    def on_training_epoch_end(self, state):
        self.scheduler.step(state.epoch)


class MetricScheduler(Scheduler):
    __name__ = 'metric-based scheduler'

    def __init__(self, scheduler_class_name, use_loss, monitor, optimizer, **kwargs):
        self.scheduler = s2c(scheduler_class_name)(optimizer, **kwargs)
        self.use_loss = use_loss
        self.monitor = monitor

    def on_epoch_end(self, state):
        monitor_main_key = LOSSES if self.use_loss else SCORES
        assert self.monitor in state.epoch_results[monitor_main_key], f'{self.monitor} not found in epoch_results'
        metric = state.epoch_results[monitor_main_key][self.monitor]
        self.scheduler.step(metric)

        # Do not forget to store dict
        super().on_epoch_end(state)
