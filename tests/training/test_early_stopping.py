from torch.nn import Linear

from pydgn.model.interface import ModelInterface
from pydgn.static import LOSSES, SCORES, BEST_EPOCH, MIN, MAX
from pydgn.training.callback.early_stopping import PatienceEarlyStopper
from pydgn.training.event.state import State


class FakeModel(ModelInterface):
    def __init__(self):
        super().__init__(0, 0, 0, None, None)
        self.lin = Linear(10, 10)


def test_early_stopping_patience():

    for use_as_loss in [False, True]:
        for patience in [2, 10]:

            early_stopper = PatienceEarlyStopper(
                "validation_main_loss"
                if use_as_loss
                else "validation_main_score",
                mode=MIN if use_as_loss else MAX,
                patience=patience,
                checkpoint=False,
            )

            state = State(model=FakeModel(), optimizer=None, device="cpu")

            # Update state with epoch results
            epoch_results = {LOSSES: {}, SCORES: {}}
            state.update(epoch_results=epoch_results)

            num_epochs = 30
            for epoch in range(1, num_epochs + 1):
                state.update(epoch=epoch)

                state.epoch_results[LOSSES].update(
                    {f"validation_main_loss": epoch}
                )
                state.epoch_results[SCORES].update(
                    {f"validation_main_score": epoch}
                )

                early_stopper.on_epoch_end(state)

                if state.stop_training:
                    break

            if use_as_loss:
                # implies MIN is used. For this test we should stop and exit
                # the loop after patience epochs (the best epoch will always
                # be the first)
                assert state.best_epoch_results[BEST_EPOCH] == epoch - patience
            else:
                assert state.best_epoch_results[BEST_EPOCH] == num_epochs
