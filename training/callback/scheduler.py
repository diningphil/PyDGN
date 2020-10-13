from experiment.experiment import s2c
from training.event.handler import EventHandler


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
        if self.scheduler and 'scheduler_state' in state:
            self.scheduler.load_state_dict(state.scheduler_state)

    def on_epoch_end(self, state):
        """
        Save the scheduler state into the state object
        :param state: the shared State object
        """
        state.update(scheduler_state=self.scheduler.state_dict())

    def on_training_epoch_end(self, state):
        raise NotImplementedError("You should subclass the Scheduler Class to implement pass the appropriate arguments to the scheduler.step() method.")


class EpochScheduler(EventHandler):
    """
    Scheduler is the main event handler for schedulers. Just pass a PyTorch scheduler together with its arguments in the
    configuration file.
    """
    __name__ = 'epoch-based scheduler'

    def on_training_epoch_end(self, state):
        """
        Perform a scheduler step
        :param state: the shared State object
        """
        self.scheduler.step(state.epoch)
