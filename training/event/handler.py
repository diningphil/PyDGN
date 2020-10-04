class EventHandler:
    """ Simple class implementing the publisher/subscribe pattern for training. This class provides the main methods
        that a subscriber should implement. Each subscriber can make use of the training.core.event.state.State object
        that is passed to each method. """
    def on_fetch_data(self, state):
        pass

    def on_fit_start(self, state):
        pass

    def on_fit_end(self, state):
        pass

    def on_epoch_start(self, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_training_epoch_start(self, state):
        pass

    def on_training_epoch_end(self, state):
        pass

    def on_eval_epoch_start(self, state):
        pass

    def on_eval_epoch_end(self, state):
        pass

    def on_training_batch_start(self, state):
        pass

    def on_training_batch_end(self, state):
        pass

    def on_eval_batch_start(self, state):
        pass

    def on_eval_batch_end(self, state):
        pass

    def on_forward(self, state):
        pass

    def on_backward(self, state):
        pass

    def on_compute_metrics(self, state):
        pass
