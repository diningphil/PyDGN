from pydgn.training.callback.engine_callback import EngineCallback


class DynamicEngineCallback(EngineCallback):
    __name__ = "dynamic_engine_callback"

    def __init__(self, store_last_checkpoint):
        super().__init__(store_last_checkpoint)

    def on_forward(self, state):
        # Forward pass
        outputs = state.model.forward(state.batch_input, state.last_hidden_state)
        state.update(batch_outputs=outputs)
