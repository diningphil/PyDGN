import copy
import os
from pathlib import Path

from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.util import atomic_save


class EngineCallback(EventHandler):
    __name__ = "engine_callback"

    def __init__(self, store_last_checkpoint):
        super().__init__()
        self.store_last_checkpoint = store_last_checkpoint

    # Allows to profile data loading
    def on_fetch_data(self, state):
        data = state.loader_iterable.next()
        state.update(batch_input=data)

    def on_forward(self, state):
        # Forward pass
        outputs = state.model.forward(state.batch_input)
        state.update(batch_outputs=outputs)

    def on_epoch_end(self, state):
        # Save last checkpoint
        if self.store_last_checkpoint:
            if not os.path.exists(Path(state.exp_path)):
                os.makedirs(Path(state.exp_path))

            last_ckpt = {
                EPOCH: state.epoch,
                MODEL_STATE: copy.deepcopy(state.model.state_dict()),
                OPTIMIZER_STATE: getattr(state, OPTIMIZER_STATE, None),
                SCHEDULER_STATE: getattr(state, SCHEDULER_STATE, None),
                STOP_TRAINING: state.stop_training}
            last_ckpt.update(state.epoch_results)
            atomic_save(last_ckpt, Path(state.exp_path, LAST_CHECKPOINT_FILENAME))
