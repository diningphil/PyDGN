import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State


class Plotter(EventHandler):
    r"""
    Plotter is the main event handler for plotting at training time.

    Args:
        exp_path (str): path where to store the Tensorboard logs
        use_wandb (bool): whether to upload tensorboard to wandb
        keargs (dict): additional arguments that may depend on the plotter
    """
    def __init__(self, exp_path: str, use_wandb: bool, **kwargs: dict):
        super().__init__()
        self.exp_path = exp_path

        if not os.path.exists(Path(self.exp_path, TENSORBOARD)):
            os.makedirs(Path(self.exp_path, TENSORBOARD))
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "To use the Weights and Biases Logger please install wandb."
                    "Run `pip install wandb` to install it."
                )
            self._wandb.tensorboard.patch(root_logdir=self.exp_path, tensorboard_x=False, pytorch=True)
            # Initialize a W&B run 
            if self._wandb.run is None:
                self._wandb.init(project="PyDGN")
        self.writer = SummaryWriter(log_dir=Path(self.exp_path, 'tensorboard'))


    def on_epoch_end(self, state: State):

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

    def on_fit_end(self, state: State):
        self.writer.close()
