import torch
import numpy as np


class State:
    """ Any object of this class contains training information that is handled and modified by the training engine
        (see training.core.engine) as well as by the EventHandler callbacks (see training.core.callbacks). """
    TRAINING = 'training'
    EVALUATION = 'evaluation'

    def __init__(self, model, optimizer, **values):

        self.update(**values)

        self.model = model
        self.optimizer = optimizer
        self.mode = None  # it should be training or evaluation
        self.set = None  # it should be training, validation or test
        self.epoch = 1

        self.stop_training = False

        self.batch_loss = None
        self.batch_score = None
        self.batch_targets = None
        self.batch_predictions = None

        self.epoch_loss = None
        self.epoch_score = None
        self.epoch_data_list = None

        self.train_loss = None
        self.train_score = None
        self.train_embeddings_tuple = None

        self.val_loss = None
        self.val_score = None
        self.val_embeddings_tuple = None

        self.val_loss = None
        self.val_score = None
        self.val_embeddings_tuple = None

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.__dict__

    def update(self, **values):
        for name, value in values.items():
            setattr(self, name, value)