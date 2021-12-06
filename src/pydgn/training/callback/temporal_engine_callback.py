from pydgn.training.callback.engine_callback import EngineCallback


class TemporalEngineCallback(EngineCallback):
    __name__ = "temporal_engine_callback"

    def __init__(self, store_last_checkpoint):
        super().__init__(store_last_checkpoint)

    def on_forward(self, state):
        # Forward pass, the last hidden state gets passed as additional argument
        outputs = state.model.forward(state.batch_input, state.last_hidden_state)
        state.update(batch_outputs=outputs)
