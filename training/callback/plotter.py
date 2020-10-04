import os
from pathlib import Path

import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from training.engine import TrainingEngine
from training.event.handler import EventHandler


class Plotter(EventHandler):
    """
    Plotter is the main event handler for plotting at training time.
    """
    __name__ = 'plotter'

    def __init__(self, exp_path, **kwargs):
        super().__init__()
        self.exp_path = exp_path

        if not os.path.exists(Path(self.exp_path, 'tensorboard')):
            os.makedirs(Path(self.exp_path, 'tensorboard'))
        self.writer = SummaryWriter(log_dir=Path(self.exp_path, 'tensorboard'))

    def on_epoch_end(self, state):

        for k, v in state.epoch_results['losses'].items():
            loss_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = ' '.join(k.split('_')[1:])
            if TrainingEngine.TRAINING in k:
                loss_scalars[f'{TrainingEngine.TRAINING}'] = v
            elif TrainingEngine.VALIDATION in k:
                loss_scalars[f'{TrainingEngine.VALIDATION}'] = v
            elif TrainingEngine.TEST in k:
                loss_scalars[f'{TrainingEngine.TEST}'] = v

            self.writer.add_scalars(loss_name, loss_scalars, state.epoch)


        for k, v in state.epoch_results['scores'].items():
            score_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            score_name = ' '.join(k.split('_')[1:])
            if TrainingEngine.TRAINING in k:
                score_scalars[f'{TrainingEngine.TRAINING}'] = v
            elif TrainingEngine.VALIDATION in k:
                score_scalars[f'{TrainingEngine.VALIDATION}'] = v
            elif TrainingEngine.TEST in k:
                score_scalars[f'{TrainingEngine.TEST}'] = v

            self.writer.add_scalars(score_name, score_scalars, state.epoch)


    def on_fit_end(self, state):
        self.writer.close()
