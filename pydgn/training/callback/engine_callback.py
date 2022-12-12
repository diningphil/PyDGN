import copy
import os
from pathlib import Path

from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State
from pydgn.training.util import atomic_save


class EngineCallback(EventHandler):
    r"""
    Class responsible for fetching data and handling current-epoch checkpoints
     at training time.

    Args:
        store_last_checkpoint (bool): if ``True``, keep the model's
            checkpoint for the last training epoch
    """

    def __init__(self, store_last_checkpoint: bool):
        super().__init__()
        self.store_last_checkpoint = store_last_checkpoint

    # Allows to profile data loading
    def on_fetch_data(self, state: State):
        """
        Fetches next batch of data from loader and updates the `batch_input`
        field of the state

        Args:
            state (:class:`~training.event.state.State`): object holding
                training information
        """
        data = next(state.loader_iterable)
        state.update(batch_input=data)

    def on_forward(self, state: State):
        """
        Calls the forward method of the model and stores the outputs in the
        `batch_outputs` field of the state.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        # Forward pass
        outputs = state.model.forward(state.batch_input)
        state.update(batch_outputs=outputs)

    def on_epoch_end(self, state: State):
        """
        Stores the checkpoint in a dictionary with the following fields:

        * ``EPOCH`` (as defined in ``pydgn.static``)
        * ``MODEL_STATE`` (as defined in ``pydgn.static``)
        * ``OPTIMIZER_STATE`` (as defined in ``pydgn.static``)
        * ``SCHEDULER_STATE`` (as defined in ``pydgn.static``)
        * ``STOP_TRAINING`` (as defined in ``pydgn.static``)

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        # Save last checkpoint
        if self.store_last_checkpoint:
            if not os.path.exists(Path(state.exp_path)):
                os.makedirs(Path(state.exp_path))

            last_ckpt = {
                EPOCH: state.epoch,
                MODEL_STATE: copy.deepcopy(state.model.state_dict()),
                OPTIMIZER_STATE: getattr(state, OPTIMIZER_STATE, None),
                SCHEDULER_STATE: getattr(state, SCHEDULER_STATE, None),
                STOP_TRAINING: state.stop_training,
            }
            last_ckpt.update(state.epoch_results)
            atomic_save(
                last_ckpt, Path(state.exp_path, LAST_CHECKPOINT_FILENAME)
            )


class IterableEngineCallback(EngineCallback):
    r"""
    Class that extends :class:`pydgn.training.callback.EngineCallback`
    to the processing of Iterable-style datasets.
    Needs to be used together with the appropriate engine class.
    """

    def on_fetch_data(self, state: State):
        """
        Fetches next batch of data from loader (if any, as data comes
        from a stream of unknown length)
        and updates the `batch_input` field of the state

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        try:
            data = next(state.loader_iterable)
            state.update(batch_input=data)
        except StopIteration as e:
            state.update(stop_fetching=True)
            state.update(batch_input=None)


class TemporalEngineCallback(EngineCallback):
    r"""
    Class that extends :class:`pydgn.training.callback.EngineCallback`
    to the processing of temporal datasets.
    Needs to be used together with the appropriate engine class.
    """

    def on_forward(self, state):
        """
        Calls the forward method of the model and stores the outputs in the
        `batch_outputs` field of the state. In addition to the input, passes
        to the model the hidden state computed at the previous time step.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        # Forward pass, the last hidden state gets passed as additional
        # argument
        outputs = state.model.forward(
            state.batch_input, state.last_hidden_state
        )
        state.update(batch_outputs=outputs)
