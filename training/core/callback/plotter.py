from pathlib import Path
import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns

from training.core.event.handler import EventHandler


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
        self.train_writer = SummaryWriter(log_dir=Path(self.exp_path, 'tensorboard', 'training'))
        self.valid_writer = SummaryWriter(log_dir=Path(self.exp_path, 'tensorboard', 'validation'))

    def on_epoch_end(self, state):

        train_loss = state.epoch_results['training_loss']
        self.train_writer.add_scalar('Loss', train_loss, state.epoch)

        val_loss = state.epoch_results['validation_loss']
        if val_loss is not None:
            self.valid_writer.add_scalar('Loss', val_loss, state.epoch)

        for k, v in state.epoch_results['scores'].items():
            # Remove training/validation/test prefix (coupling with Engine)
            score_name = ' '.join(k.split('_')[1:])
            if 'train' in k:
                self.train_writer.add_scalar(f'{score_name}', v, state.epoch)
            elif 'valid' in k:
                self.valid_writer.add_scalar(f'{score_name}', v, state.epoch)

