import torch

from pydgn.experiment.util import s2c
from pydgn.training.event.handler import EventHandler


class GradientClipping(EventHandler):
    """
    GradientClipping is the main event handler for gradient clippers. Just pass a PyTorch scheduler together with its
     arguments in the configuration file.
    """
    __name__ = 'gradient clipper'

    def __init__(self, gradient_clipping_class_name, **kwargs):
        """
        Istantiates the gradient clipper object
        :param gradient_clipping_class_name:
        :param kwargs:
        """
        self.gradient_clipper = s2c(gradient_clipping_class_name)(**kwargs)

    def on_backward(self, state):
        """
        Clip the gradient using the gradient clipper
        :param state: the State object shared during training
        """
        self.gradient_clipper.clip_gradients(state.model.parameters())


class StandardGradientClipper:
    def __init__(self, factor):
        self.factor = factor

    def clip_gradients(self, parameters):
        torch.nn.utils.clip_grad_norm_(parameters, self.factor)
