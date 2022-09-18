from torch.nn.utils import clip_grad_value_

from pydgn.experiment.util import s2c
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State


class GradientClipper(EventHandler):
    r"""
    GradientClipper is the main event handler for gradient clippers.
    Just pass a PyTorch scheduler together with its
    arguments in the configuration file.

    Args:
        clip_value (float): the gradient will be clipped in
            [-clip_value, clip_value]
        kwargs (dict): additional arguments
    """

    def __init__(self, clip_value: float, **kwargs: dict):
        self.clip_value = clip_value

    def on_backward(self, state: State):
        """
        Clips the gradients of the model before the weights are updated.

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        clip_grad_value_(state.model.parameters(), clip_value=self.clip_value)
