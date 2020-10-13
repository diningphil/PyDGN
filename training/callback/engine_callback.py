import os
import copy
from pathlib import Path
from training.util import atomic_save
from training.event.handler import EventHandler


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
                         'epoch': state.epoch,
                         'model_state': copy.deepcopy(state.model.state_dict()),
                         'optimizer_state': state.optimizer_state,
                         'scheduler_state': state['scheduler_state'],
                         'stop_training': state.stop_training }
            last_ckpt.update(state.epoch_results)
            atomic_save(last_ckpt,
                        Path(state.exp_path, state.LAST_CHECKPOINT_FILENAME))
