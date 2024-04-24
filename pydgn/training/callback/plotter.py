import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State


class Plotter(EventHandler):
    r"""
    Plotter is the main event handler for plotting at training time.

    Args:
        exp_path (str): path where to store the Tensorboard logs
        store_on_disk (bool): whether to store all metrics on disk.
            Defaults to False
        kwargs (dict): additional arguments that may depend on the plotter
    """

    def __init__(self,
                 exp_path: str,
                 store_on_disk: bool = False,
                 **kwargs: dict):
        super().__init__()
        self.exp_path = exp_path
        self.store_on_disk = store_on_disk

        if not os.path.exists(Path(self.exp_path, TENSORBOARD)):
            os.makedirs(Path(self.exp_path, TENSORBOARD))
        self.writer = SummaryWriter(log_dir=Path(self.exp_path, "tensorboard"))

        self.stored_metrics = {"losses": {}, "scores": {}}
        self.stored_metrics_path = Path(self.exp_path, "metrics_data.torch")
        if os.path.exists(self.stored_metrics_path):
            self.stored_metrics = torch.load(self.stored_metrics_path)


    def on_epoch_end(self, state: State):
        """
        Writes Training, Validation and (if any) Test metrics to Tensorboard

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        for k, v in state.epoch_results[LOSSES].items():
            loss_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = " ".join(k.split("_")[1:])
            if TRAINING in k:
                loss_scalars[f"{TRAINING}"] = v
            elif VALIDATION in k:
                loss_scalars[f"{VALIDATION}"] = v
            elif TEST in k:
                loss_scalars[f"{TEST}"] = v

            self.writer.add_scalars(loss_name, loss_scalars, state.epoch)

            if self.store_on_disk:
                t = "losses"
                if not k in self.stored_metrics[t]:
                    self.stored_metrics[t][k] = [v.item()]
                else:
                    self.stored_metrics[t][k].append(v.item())

        for k, v in state.epoch_results[SCORES].items():
            score_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            score_name = " ".join(k.split("_")[1:])
            if TRAINING in k:
                score_scalars[f"{TRAINING}"] = v
            elif VALIDATION in k:
                score_scalars[f"{VALIDATION}"] = v
            elif TEST in k:
                score_scalars[f"{TEST}"] = v

            self.writer.add_scalars(score_name, score_scalars, state.epoch)

            if self.store_on_disk:
                t = "scores"
                if not k in self.stored_metrics[t]:
                    self.stored_metrics[t][k] = [v.item()]
                else:
                    self.stored_metrics[t][k].append(v.item())

        if self.store_on_disk:
            try:
                torch.save(self.stored_metrics, self.stored_metrics_path)
            except RuntimeError as e:
                print(e)


    def on_fit_end(self, state: State):
        """
        Frees resources by closing the Tensorboard writer

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.writer.close()


class WandbPlotter(EventHandler):
    r"""
    EventHandler subclass for logging to Weights & Biases

    Args:
        wandb_project (str): Project Name for W&B
        wandb_entity (str): Entity Name for W&B
        kwargs (dict): additional arguments that may depend on the plotter
    """

    def __init__(
        self, exp_path: str, wandb_project, wandb_entity, **kwargs: dict
    ):
        super().__init__()
        self.exp_path = exp_path

        try:
            import wandb

            self._wandb = wandb
            self._wandb.require(experiment="service")
            self._wandb.setup()
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                name=self.exp_path, project=wandb_project, entity=wandb_entity
            )

    def on_epoch_end(self, state: State):
        """
        Writes Training, Validation and (if any) Test metrics to WandB

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """

        for k, v in state.epoch_results[LOSSES].items():
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = " ".join(k.split("_")[1:])
            if TRAINING in k:
                self._wandb.log({f"Train/{loss_name}": v})
            elif VALIDATION in k:
                self._wandb.log({f"Valid/{loss_name}": v})
            elif TEST in k:
                self._wandb.log({f"Test/{loss_name}": v})

        for k, v in state.epoch_results[SCORES].items():
            # Remove training/validation/test prefix (coupling with Engine)
            score_name = " ".join(k.split("_")[1:])
            if TRAINING in k:
                self._wandb.log({f"Train/{score_name}": v})
            elif VALIDATION in k:
                self._wandb.log({f"Valid/{score_name}": v})
            elif TEST in k:
                self._wandb.log({f"Test/{score_name}": v})

    def on_fit_end(self, state: State):
        """
        Frees resources by closing the WandB writer

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self._wandb.finish()
