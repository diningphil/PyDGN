from pydgn.training.event.state import State


class EventHandler:
    r"""
    Interface that adheres to the publisher/subscribe pattern for training. It defines the main methods
    that a subscriber should implement. Each subscriber can make use of the
    :class:`~training.event.state.State` object that is passed to each method, so detailed knowledge
    about that object is required.

    This class defines a set of callbacks that should cover a sufficient number of use cases. These are meant to work
    closely with the :class:`~training.callback.engine.TrainingEngine` object, which implements the overall training and
    evaluation process. This training engine is fairly general to accomodate a number of situations, so we expect we
    won't need to change it much to deal with static graph problems.

    We list below some pre/post conditions for each method that depend on the current implementation of the main
    training engine :class:`~training.callback.engine.TrainingEngine`. These are clearly not strict conditions, but
    they can help design new training engines with their own publisher/subscriber patterns or create subclasses
    of :class:`~training.callback.engine.TrainingEngine` that require special modifications.
    """

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

    def on_fetch_data(self, state: State):
        """
        Load the next batch of data, possibly applying some kind of additional pre-processing not
        included in the :mod:`~pydgn.data.transform` package.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The data loader is contained in  ``state.loader_iterable`` and the minibatch ID (i.e., a counter) is stored
            in``state.id_batch``

        Post-condition:
            The ``state`` object now has a field ``batch_input`` with the next batch of data
        """
        pass

    def on_fit_start(self, state: State):
        """
        Initialize an object at the beginning of the training phase, e.g., the internals of an optimizer,
        using the information contained in ``state``.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.initial_epoch``: the initial epoch from which to start/resume training
              * ``state.stop_training``: do/don't train the model
              * ``state.optimizer_state``: the internal state of the optimizer (can be ``None``)
              * ``state.scheduler_state``: the internal state of the scheduler (can be  ``None``)
              * ``state.best_epoch_results``: a dictionary with the best results computed so far (can be used when resuming training, either for early stopping or to keep some information about the last checkpoint).

        """
        pass

    def on_fit_end(self, state: State):
        """
        Training has ended, free all resources, e.g., close Tensorboard writers.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        """
        pass

    def on_epoch_start(self, state: State):
        """
        Initialize/reset some internal state at the start of a training/evaluation epoch.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            * The following fields have been initialized:
              * ``state.epoch``: the current epoch
              * ``state.return_node_embeddings``: do/don't return node_embeddings for each graph at the end of the epoch

        """
        pass

    def on_epoch_end(self, state: State):
        """
        Perform bookkeeping operations at the end of an epoch, e.g., early stopping, plotting, etc.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
                * ``state.epoch_loss``: a dictionary containing the aggregated loss value across all minibatches
                * ``state.epoch_score``: a dictionary containing the aggregated score value across all minibatches

        Post-condition:
            The following fields have been initialized:
                * ``state.stop_training``: do/don't train the model
                * ``state.optimizer_state``: the internal state of the optimizer (can be ``None``)
                * ``state.scheduler_state``: the internal state of the scheduler (can be  ``None``)
                * ``state.best_epoch_results``: a dictionary with the best results computed so far (can be used when resuming training, either for early stopping or to keep some information about the last checkpoint).

        """
        pass

    def on_training_epoch_start(self, state: State):
        """
        Initialize/reset some internal state at the start of a training epoch.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.set``: it must be set to :const:`~pydgn.static.TRAINING`

        """
        pass

    def on_training_epoch_end(self, state: State):
        """
        Initialize/reset some internal state at the end of a training epoch.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Post-condition:
            The following fields have been initialized:
              * ``state.epoch_loss``: a dictionary containing the aggregated loss value across all minibatches
              * ``state.epoch_score``: a dictionary containing the aggregated score value across all minibatches

        """
        pass

    def on_eval_epoch_start(self, state: State):
        """
        Initialize/reset some internal state at the start of an evaluation epoch.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.set``: the dataset type (can be :const:`~pydgn.static.TRAINING`, :const:`~pydgn.static.VALIDATION` or :const:`~pydgn.static.TEST`)

        """
        pass

    def on_eval_epoch_end(self, state: State):
        """
        Initialize/reset some internal state at the end of an evaluation epoch.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Post-condition:
            The following fields have been initialized:
              * ``state.epoch_loss``: a dictionary containing the aggregated loss value across all minibatches
              * ``state.epoch_score``: a dictionary containing the aggregated score value across all minibatches

        """
        pass

    def on_training_batch_start(self, state: State):
        """
        Initialize/reset some internal state before training on a new minibatch of data.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.set``: it must be set to :const:`~pydgn.static.TRAINING`
              * ``state.batch_input``: the input to be fed to the model
              * ``state.batch_targets``: the ground truth values to be fed to the model (if any, ow a dummy value can be used)
              * ``state.batch_num_graphs``: the total number of graphs in the minibatch
              * ``state.batch_num_nodes``: the total number of nodes in the minibatch
              * ``state.batch_num_targets``: the total number of ground truth values in the minibatch

        """
        pass

    def on_training_batch_end(self, state: State):
        """
        Initialize/reset some internal state after training on a new minibatch of data.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.set``: it must be set to :const:`~pydgn.static.TRAINING`
              * ``state.batch_num_graphs``: the total number of graphs in the minibatch
              * ``state.batch_num_nodes``: the total number of nodes in the minibatch
              * ``state.batch_num_targets``: the total number of ground truth values in the minibatch
              * ``state.batch_loss``: a dictionary holding the loss of the minibatch
              * ``state.batch_loss_extra``: a dictionary containing extra info, e.g., intermediate loss scores etc.
              * ``state.batch_score``: a dictionary holding the score of the minibatch

        """
        pass

    def on_eval_batch_start(self, state: State):
        """
        Initialize/reset some internal state before evaluating on a new minibatch of data.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.set``: the dataset type (can be :const:`~pydgn.static.TRAINING`, :const:`~pydgn.static.VALIDATION` or :const:`~pydgn.static.TEST`)
              * ``state.batch_input``: the input to be fed to the model
              * ``state.batch_targets``: the ground truth values to be fed to the model (if any, ow a dummy value can be used)
              * ``state.batch_num_graphs``: the total number of graphs in the minibatch
              * ``state.batch_num_nodes``: the total number of nodes in the minibatch
              * ``state.batch_num_targets``: the total number of ground truth values in the minibatch

        """
        pass

    def on_eval_batch_end(self, state: State):
        """
        Initialize/reset some internal state after evaluating on a new minibatch of data.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.set``: the dataset type (can be :const:`~pydgn.static.TRAINING`, :const:`~pydgn.static.VALIDATION` or :const:`~pydgn.static.TEST`)
              * ``state.batch_num_graphs``: the total number of graphs in the minibatch
              * ``state.batch_num_nodes``: the total number of nodes in the minibatch
              * ``state.batch_num_targets``: the total number of ground truth values in the minibatch
              * ``state.batch_loss``: a dictionary holding the loss of the minibatch
              * ``state.batch_loss_extra``: a dictionary containing extra info, e.g., intermediate loss scores etc.
              * ``state.batch_score``: a dictionary holding the score of the minibatch

        """
        pass

    def on_forward(self, state: State):
        """
        Feed the input data to the model.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.batch_input``: the input to be fed to the model
              * ``state.batch_targets``: the ground truth values to be fed to the model (if any, ow a dummy value can be used)

        Post-condition:
            The following fields have been initialized:
              * ``state.batch_outputs``: the output produced the model (a tuple of values)

        """
        pass

    def on_backward(self, state: State):
        """
        Updates the parameters of the model using loss information.

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.batch_loss``: a dictionary holding the loss of the minibatch

        """
        pass

    def on_compute_metrics(self, state: State):
        """
        Computes the metrics of interest using the output and ground truth information obtained so far.
        The loss-related subscriber MUST be called before the score-related one

        Args:
            state (:class:`~training.event.state.State`): object holding training information

        Pre-condition:
            The following fields have been initialized:
              * ``state.batch_input``: the input to be fed to the model
              * ``state.batch_targets``: the ground truth values to be fed to the model (if any, ow a dummy value can be used)
              * ``state.batch_outputs``: the output produced the model (a tuple of values)

        Post-condition:
            The following fields have been initialized:
              * ``state.batch_loss``: a dictionary holding the loss of the minibatch
              * ``state.batch_loss_extra``: a dictionary containing extra info, e.g., intermediate loss scores etc.
              * ``state.batch_score``: a dictionary holding the score of the minibatch

        """
        pass
