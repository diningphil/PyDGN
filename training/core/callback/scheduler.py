from config.utils import s2c
from training.core.event.handler import EventHandler


class Scheduler(EventHandler):
    """
    Scheduler is the main event handler for schedulers. Just pass a PyTorch scheduler together with its arguments in the
    configuration file.
    """
    def __init__(self, scheduler_class_name, optimizer, **kwargs):
        self.scheduler = s2c(scheduler_class_name)(optimizer, **kwargs)

    def on_fit_start(self, state):
        """
        Load scheduler from state if any
        :param state: the shared State object
        """
        if self.scheduler and 'scheduler_state' in state:
            self.scheduler.load_state_dict(state.scheduler_state)

    def on_training_epoch_end(self, state):
        """
        Perform a scheduler step
        :param state: the shared State object
        """
        self.scheduler.step(state.epoch)

    def on_epoch_end(self, state):
        """
        Save the scheduler state into the state object
        :param state: the shared State object
        """
        state.update(scheduler_state=self.scheduler.state_dict())
