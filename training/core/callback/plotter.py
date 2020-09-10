from pathlib import Path
import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
from training.core.engine import TrainingEngine
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
        self.writer = SummaryWriter(log_dir=Path(self.exp_path, 'tensorboard'))

    def on_epoch_end(self, state):

        loss_scalars = {}

        train_loss = state.epoch_results[f'{TrainingEngine.TRAINING}_loss']
        #self.writer.add_scalar(f'{TrainingEngine.TRAINING}/Loss', train_loss, state.epoch)
        loss_scalars[f'{TrainingEngine.TRAINING}'] = train_loss

        val_loss = state.epoch_results[f'{TrainingEngine.VALIDATION}_loss']
        if val_loss is not None:
            #self.writer.add_scalar(f'{TrainingEngine.VALIDATION}/Loss', val_loss, state.epoch)
            loss_scalars[f'{TrainingEngine.VALIDATION}'] = val_loss

        test_loss = state.epoch_results[f'{TrainingEngine.TEST}_loss']
        if test_loss is not None:
            #self.writer.add_scalar(f'{TrainingEngine.TEST}/Loss', test_loss, state.epoch)
            loss_scalars[f'{TrainingEngine.TEST}'] = test_loss

        self.writer.add_scalars('Loss', loss_scalars, state.epoch)

        for k, v in state.epoch_results['scores'].items():

            score_scalars = {}

            # Remove training/validation/test prefix (coupling with Engine)
            score_name = ' '.join(k.split('_')[1:])
            if 'train' in k:
                #self.writer.add_scalar(f'{TrainingEngine.TRAINING}/{score_name}', v, state.epoch)
                score_scalars[f'{TrainingEngine.TRAINING}'] = v
            elif 'valid' in k:
                #self.writer.add_scalar(f'{TrainingEngine.VALIDATION}/{score_name}', v, state.epoch)
                score_scalars[f'{TrainingEngine.VALIDATION}'] = v
            elif 'test' in k:
                #self.writer.add_scalar(f'{TrainingEngine.TEST}/{score_name}', v, state.epoch)
                score_scalars[f'{TrainingEngine.TEST}'] = v

            self.writer.add_scalars(score_name, score_scalars, state.epoch)


    def on_fit_end(self, state):
        self.writer.close()
