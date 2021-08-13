class EventHandler:
    """ Simple class implementing the publisher/subscribe pattern for training. This class provides the main methods
        that a subscriber should implement. Each subscriber can make use of the training.core.event.state.State object
        that is passed to each method. """

    ON_FETCH_DATA = "on_fetch_data"
    ON_FIT_START = "on_fit_start"
    ON_FIT_END = "on_fit_end"
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    ON_TRAINING_EPOCH_START = "on_training_epoch_start"
    ON_TRAINING_EPOCH_END = "on_training_epoch_end"
    ON_EVAL_EPOCH_START = "on_eval_epoch_start"
    ON_EVAL_EPOCH_END = "on_eval_epoch_end"
    ON_TRAINING_BATCH_START = "on_training_batch_start"
    ON_TRAINING_BATCH_END = "on_training_batch_end"
    ON_EVAL_BATCH_START = "on_eval_batch_start"
    ON_EVAL_BATCH_END = "on_eval_batch_end"
    ON_FORWARD = 'on_forward'
    ON_BACKWARD = "on_backward"
    ON_COMPUTE_METRICS = "on_compute_metrics"

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
