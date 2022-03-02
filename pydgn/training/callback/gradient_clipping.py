from pydgn.experiment.util import s2c
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State


class GradientClipper(EventHandler):
    r"""
    GradientClipper is the main event handler for gradient clippers. Just pass a PyTorch scheduler together with its
    arguments in the configuration file.

    Args:
        gradient_clipper_class_name (str): the dotted path to the gradient clipper class name
        kwargs (dict): additional arguments
    """
    def __init__(self, gradient_clipper_class_name: str, **kwargs: dict):
        self.gradient_clipper = s2c(gradient_clipper_class_name)(**kwargs)

    def on_backward(self, state: State):
        self.gradient_clipper.clip_gradients(state.model.parameters())
