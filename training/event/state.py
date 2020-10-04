import torch
import numpy as np


class State:
    """ Any object of this class contains training information that is handled and modified by the training engine
        (see training.core.engine) as well as by the EventHandler callbacks (see training.core.callbacks). """
    TRAINING = 'training'
    EVALUATION = 'evaluation'
    LAST_CHECKPOINT_FILENAME = 'last_checkpoint.pth'
    BEST_CHECKPOINT_FILENAME = 'best_checkpoint.pth'

    def __init__(self, model, optimizer, **values):

        self.update(**values)

        self.initial_epoch = 0
        self.epoch = self.initial_epoch
        self.model = model
        self.optimizer = optimizer
        self.stop_training = False

    def __getitem__(self, name):
        return getattr(self, name, None)

    def __contains__(self, name):
        return name in self.__dict__

    def update(self, **values):
        for name, value in values.items():
            setattr(self, name, value)
