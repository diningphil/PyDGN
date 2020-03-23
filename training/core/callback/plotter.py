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
    def __init__(self, exp_path, **kwargs):
        super().__init__()
        self.exp_path = exp_path