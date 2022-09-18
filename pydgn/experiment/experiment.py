import random
from typing import Callable, Tuple, List

import numpy as np
import torch

from pydgn.data.provider import DataProvider
from pydgn.evaluation.config import Config
from pydgn.evaluation.util import return_class_and_args
from pydgn.experiment.util import s2c
from pydgn.log.logger import Logger
from pydgn.model.interface import ModelInterface, ReadoutInterface
from pydgn.static import DEFAULT_ENGINE_CALLBACK
from pydgn.training.engine import TrainingEngine


class Experiment:
    r"""
    Class that handles a single experiment.

    Args:
        model_configuration (dict): the dictionary holding the
            experiment-specific configuration
        exp_path (str): path to the experiment folder
        exp_seed (int): the experiment's seed to use
    """

    def __init__(
        self, model_configuration: dict, exp_path: str, exp_seed: int
    ):
        self.model_config = Config(model_configuration)
        self.exp_path = exp_path
        self.exp_seed = exp_seed
        # Set seed here to aid reproducibility
        np.random.seed(self.exp_seed)
        torch.manual_seed(self.exp_seed)
        torch.cuda.manual_seed(self.exp_seed)
        random.seed(self.exp_seed)

        # torch.use_deterministic_algorithms(True) for future versions of
        # Pytorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _return_class_and_args(
        config: Config, key: str
    ) -> Tuple[Callable[..., object], dict]:
        r"""
        Returns the class and arguments associated to a specific key in the
        configuration file.

        Args:
            config: the configuration dictionary
            key: a string representing a particular class in the
                configuration dictionary

        Returns:
            a tuple (class, dict of arguments), or (None, None) if the key
            is not present in the config dictionary
        """
        if key not in config or config[key] is None:
            return None, None
        elif isinstance(config[key], str):
            return s2c(config[key]), {}
        elif isinstance(config[key], dict):
            return (
                s2c(config[key]["class_name"]),
                config[key]["args"] if "args" in config[key] else {},
            )
        else:
            raise NotImplementedError(
                "Parameter has not been formatted properly"
            )

    def _create_model(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        readout_classname: str,
        config: Config,
    ) -> ModelInterface:
        r"""
        Instantiates a model that implements the
        :class:`~pydgn.model.model.ModelInterface` interface

        Args:
            dim_node_features (int): number of node features
            dim_edge_features (int): number of edge features
            dim_target (int): target dimension
            readout_classname (str): string containing the model's class
            config (:class:`~pydgn.evaluation.config.Config`):
                the configuration dictionary

        Returns:
            a model that implements the
            :class:`~pydgn.model.model.ModelInterface` interface
        """
        model = s2c(config["model"])(
            dim_node_features=dim_node_features,
            dim_edge_features=dim_edge_features,
            dim_target=dim_target,
            readout_class=s2c(readout_classname)
            if readout_classname is not None
            else None,
            config=config,
        )

        # move to device
        # model .to() may not return anything
        model.to(self.model_config.device)
        return model

    def create_supervised_model(
        self, dim_node_features: int, dim_edge_features: int, dim_target: int
    ) -> ModelInterface:
        r"""
        Instantiates a **supervised** model that implements the
        :class:`~pydgn.model.model.ModelInterface` interface,
        using the ``supervised_config`` field in the configuration file.

        Args:
            dim_node_features (int): number of node features
            dim_edge_features (int): number of edge features
            dim_target (int): target dimension

        Returns:
            a model that implements the
            :class:`~pydgn.model.model.ModelInterface` interface
        """
        readout_classname = (
            self.model_config.supervised_config["readout"]
            if "readout" in self.model_config.supervised_config
            else None
        )
        return self._create_model(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_classname,
            self.model_config.supervised_config,
        )

    def create_supervised_readout(
        self, dim_node_features: int, dim_edge_features: int, dim_target: int
    ) -> ReadoutInterface:
        r"""
        Instantiates an **supervised** readout that implements the
        :class:`~pydgn.model.model.ReadoutInterface` interface,
        using the ``supervised_config`` field in the configuration file.

        Args:
            dim_node_features (int): number of node features
            dim_edge_features (int): number of edge features
            dim_target (int): target dimension

        Returns:
            a model that implements the
            :class:`~pydgn.model.model.ReadoutInterface` interface
        """
        return s2c(self.model_config.supervised_config["readout"])(
            dim_node_features=dim_node_features,
            dim_edge_features=dim_edge_features,
            dim_target=dim_target,
            config=self.model_config.supervised_config,
        )

    def create_unsupervised_model(
        self, dim_node_features: int, dim_edge_features: int, dim_target: int
    ) -> ModelInterface:
        r"""
        Instantiates an **unsupervised** model that implements the
        :class:`~pydgn.model.model.ModelInterface` interface,
        using the ``unsupervised_config`` field in the configuration file.

        Args:
            dim_node_features (int): number of node features
            dim_edge_features (int): number of edge features
            dim_target (int): target dimension

        Returns:
            a model that implements the
            :class:`~pydgn.model.model.ModelInterface` interface
        """
        readout_classname = (
            self.model_config.unsupervised_config["readout"]
            if "readout" in self.model_config.unsupervised_config
            else None
        )
        return self._create_model(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_classname,
            self.model_config.unsupervised_config,
        )

    def create_incremental_model(
        self,
        dim_node_features: (int),
        dim_edge_features: (int),
        dim_target: (int),
        depth: (int),
        prev_outputs_to_consider: List[int],
    ) -> ModelInterface:
        r"""
        Instantiates a layer of an incremental architecture.
        It assumes the config file has a field ``layer_config``
        and another ``layer_config.arbitrary_function_config``
        that holds any kind of information for the arbitrary
        function of an incremental architecture

        Args:
            dim_node_features: input node features
            dim_edge_features: input edge features
            dim_target: target size
            depth: current depth of the architecture
            prev_outputs_to_consider: A list of previous layers to consider,
                e.g., [1,2] means the last two previous layers.

        Returns:
            a layer of a model that implements the
            :class:`~pydgn.model.model.ModelInterface` interface
        """
        readout_classname = self.model_config.layer_config.get("readout", None)
        self.model_config.layer_config["depth"] = depth
        self.model_config.layer_config[
            "prev_outputs_to_consider"
        ] = prev_outputs_to_consider
        return self._create_model(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_classname,
            self.model_config.layer_config,
        )

    def _create_engine(
        self,
        config: Config,
        model: ModelInterface,
        device: str,
        evaluate_every: int,
        reset_eval_model_hidden_state: bool,
    ) -> TrainingEngine:
        r"""
        Utility that instantiates the training engine. It looks for
        pre-defined fields in the configuration file, i.e. ``loss``,
        ``scorer``, ``optimizer``, ``scheduler``, ``gradient_clipper``,
        ``early_stopper`` and ``plotter``, all of which should be classes
        implementing the :class:`~pydgn.training.event.handler.EventHandler`
        interface

        Args:
            config (:class:`~pydgn.evaluation.config.Config`):
                the configuration dictionary
            model: the  model that needs be trained
            device (str): the string with the CUDA device to be used,
                or ``cpu``
            evaluate_every (int): number of epochs after which to
                log information
            reset_eval_model_hidden_state (bool): [temporal graph learning]
                Used when we want to reset the state after performing
                previous inference. It should be ``False`` when we are
                dealing with a single temporal graph sequence,
                because we don't want to reset the hidden state after
                processing the previous [training/validation] time steps.

        Returns:
            a :class:`~pydgn.training.engine.TrainingEngine` object
        """

        loss_class, loss_args = return_class_and_args(config, "loss")
        loss_args.update(device=device)
        loss = (
            loss_class(use_as_loss=True, **loss_args)
            if loss_class is not None
            else None
        )

        scorer_class, scorer_args = return_class_and_args(config, "scorer")
        scorer_args.update(device=device)
        scorer = (
            scorer_class(use_as_loss=False, **scorer_args)
            if scorer_class is not None
            else None
        )

        optim_class, optim_args = return_class_and_args(config, "optimizer")
        optimizer = (
            optim_class(model=model, **optim_args)
            if optim_class is not None
            else None
        )

        sched_class, sched_args = return_class_and_args(config, "scheduler")
        if sched_args is not None:
            sched_args["optimizer"] = optimizer.optimizer
        scheduler = (
            sched_class(**sched_args) if sched_class is not None else None
        )
        # Remove the optimizer obj ow troubles when dumping the config file
        if sched_args is not None:
            sched_args.pop("optimizer", None)

        grad_clip_class, grad_clip_args = return_class_and_args(
            config, "gradient_clipper"
        )
        grad_clipper = (
            grad_clip_class(**grad_clip_args)
            if grad_clip_class is not None
            else None
        )

        early_stop_class, early_stop_args = return_class_and_args(
            config, "early_stopper"
        )
        early_stopper = (
            early_stop_class(**early_stop_args)
            if early_stop_class is not None
            else None
        )

        plot_class, plot_args = return_class_and_args(config, "plotter")
        plotter = (
            plot_class(exp_path=self.exp_path, **plot_args)
            if plot_class is not None
            else None
        )

        store_last_checkpoint = config.get("checkpoint", False)
        engine_class, engine_args = return_class_and_args(config, "engine")
        engine_callback = s2c(
            engine_args.get("engine_callback", DEFAULT_ENGINE_CALLBACK)
        )

        engine = engine_class(
            engine_callback=engine_callback,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scorer=scorer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            gradient_clipper=grad_clipper,
            device=device,
            plotter=plotter,
            exp_path=self.exp_path,
            evaluate_every=evaluate_every,
            store_last_checkpoint=store_last_checkpoint,
            reset_eval_model_hidden_state=reset_eval_model_hidden_state,
        )
        return engine

    def create_supervised_engine(
        self, model: ModelInterface
    ) -> TrainingEngine:
        r"""
        Instantiates the training engine by using the ``supervised_config``
        key in the config file

        Args:
            model: the  model that needs be trained

        Returns:
            a :class:`~pydgn.training.engine.TrainingEngine` object
        """
        device = self.model_config.device
        evaluate_every = self.model_config.evaluate_every
        reset_eval_model_hidden_state = self.model_config.get(
            "reset_eval_model_hidden_state", True
        )

        return self._create_engine(
            self.model_config.supervised_config,
            model,
            device,
            evaluate_every,
            reset_eval_model_hidden_state,
        )

    def create_unsupervised_engine(
        self, model: ModelInterface
    ) -> TrainingEngine:
        r"""
        Instantiates the training engine by using the ``unsupervised_config``
        key in the config file

        Args:
            model: the  model that needs be trained

        Returns:
            a :class:`~pydgn.training.engine.TrainingEngine` object
        """
        device = self.model_config.device
        evaluate_every = self.model_config.evaluate_every
        reset_eval_model_hidden_state = self.model_config.get(
            "reset_eval_model_hidden_state", True
        )
        return self._create_engine(
            self.model_config.unsupervised_config,
            model,
            device,
            evaluate_every,
            reset_eval_model_hidden_state,
        )

    def create_incremental_engine(
        self, model: ModelInterface
    ) -> TrainingEngine:
        r"""
        Instantiates the training engine by using the ``layer_config``
        key in the config file

        Args:
            model: the  model that needs be trained

        Returns:
            a :class:`~pydgn.training.engine.TrainingEngine` object
        """
        device = self.model_config.device
        evaluate_every = self.model_config.evaluate_every
        reset_eval_model_hidden_state = self.model_config.get(
            "reset_eval_model_hidden_state", True
        )
        return self._create_engine(
            self.model_config.layer_config,
            model,
            device,
            evaluate_every,
            reset_eval_model_hidden_state,
        )

    def run_valid(self, dataset_getter, logger) -> Tuple[dict, dict]:
        r"""
        This function returns the training and validation results for a
        `model selection run`.
        **Do not attempt to load the test set inside this method!**
        **If possible, rely on already available subclasses of this class**.

        Args:
            dataset_getter (:class:`~pydgn.data.provider.DataProvider`):
                a data provider
            logger (:class:`~pydgn.log.logger.Logger`): the logger

        Returns:
            a tuple of training and test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``pydgn.static``)
            * ``SCORE`` (as defined in ``pydgn.static``)

            For instance, training_results[SCORE] is a dictionary itself
            with other fields to be used by the evaluator.
        """
        raise NotImplementedError("You must implement this function!")

    def run_test(
        self, dataset_getter: DataProvider, logger: Logger
    ) -> Tuple[dict, dict, dict]:
        """
        This function returns the training, validation and test results
        for a `final run`.
        **Do not use the test to train the model nor for
        early stopping reasons!**
        **If possible, rely on already available subclasses of this class**.

        Args:
            dataset_getter (:class:`~pydgn.data.provider.DataProvider`):
                a data provider
            logger (:class:`~pydgn.log.logger.Logger`): the logger

        Returns:
            a tuple of training,validation,test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``pydgn.static``)
            * ``SCORE`` (as defined in ``pydgn.static``)

            For instance, training_results[SCORE] is a dictionary itself
            with other fields to be used by the evaluator.
        """
        raise NotImplementedError("You must implement this function!")
