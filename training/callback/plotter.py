import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from pydgn.static import *
from pydgn.training.event.handler import EventHandler


class Plotter(EventHandler):
    """
    Plotter is the main event handler for plotting at training time.
    """
    __name__ = 'plotter'

    def __init__(self, exp_path, **kwargs):
        super().__init__()
        self.exp_path = exp_path

        if not os.path.exists(Path(self.exp_path, TENSORBOARD)):
            os.makedirs(Path(self.exp_path, TENSORBOARD))
        self.writer = SummaryWriter(log_dir=Path(self.exp_path, 'tensorboard'))

    def on_epoch_end(self, state):

        for k, v in state.epoch_results[LOSSES].items():
            loss_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = ' '.join(k.split('_')[1:])
            if TRAINING in k:
                loss_scalars[f'{TRAINING}'] = v
            elif VALIDATION in k:
                loss_scalars[f'{VALIDATION}'] = v
            elif TEST in k:
                loss_scalars[f'{TEST}'] = v

            self.writer.add_scalars(loss_name, loss_scalars, state.epoch)

        for k, v in state.epoch_results[SCORES].items():
            score_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            score_name = ' '.join(k.split('_')[1:])
            if TRAINING in k:
                score_scalars[f'{TRAINING}'] = v
            elif VALIDATION in k:
                score_scalars[f'{VALIDATION}'] = v
            elif TEST in k:
                score_scalars[f'{TEST}'] = v

            self.writer.add_scalars(score_name, score_scalars, state.epoch)

    def on_fit_end(self, state):
        self.writer.close()
