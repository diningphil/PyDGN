import os
from pathlib import Path
from typing import Callable, List, Union, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from pydgn.log.logger import Logger
from pydgn.model.interface import ModelInterface
from pydgn.static import *
from pydgn.training.callback.early_stopping import EarlyStopper
from pydgn.training.callback.engine_callback import EngineCallback
from pydgn.training.callback.gradient_clipping import GradientClipper
from pydgn.training.callback.metric import Metric
from pydgn.training.callback.optimizer import Optimizer
from pydgn.training.callback.plotter import Plotter
from pydgn.training.callback.scheduler import Scheduler
from pydgn.training.event.dispatcher import EventDispatcher
from pydgn.training.event.handler import EventHandler
from pydgn.training.event.state import State
from pydgn.training.profiler import Profiler


def log(msg, logger: Logger):
    if logger is not None:
        logger.log(msg)


class TrainingEngine(EventDispatcher):
    """
    This is the most important class when it comes to training a model. It implements the :class:`~pydgn.training.event.dispatcher.EventDispatcher` interface,
    which means that after registering some callbacks in a given order, it will proceed to trigger specific events that
    will result in the shared :class:`~pydgn.training.event.state.State` object being updated by the callbacks. Callbacks implement the EventHandler
    interface, and they receive the shared State object when any event is triggered. Knowing the order in which
    callbacks are called is important.
    The order is:

    * loss function
    * score function
    * gradient clipper
    * optimizer
    * early stopper
    * scheduler
    * plotter

    Args:
        engine_callback ( Callable[..., :class:`~pydgn.training.callback.engine_callback.EngineCallback`]): the engine callback object to be used for data fetching and checkpoints (or even other purposes if necessary)
        model (:class:`~pydgn.model.interface.ModelInterface`): the model to be trained
        loss (:class:`~pydgn.training.callback.metric.Metric`): the loss to be used
        optimizer (:class:`~pydgn.training.callback.optimizer.Optimizer`): the optimizer to be used
        scorer (:class:`~pydgn.training.callback.metric.Metric`): the score to be used
        scheduler (:class:`~pydgn.training.callback.scheduler.Scheduler`): the scheduler to be used Default is ``None``.
        early_stopper (:class:`~pydgn.training.callback.early_stopping.EarlyStopper`): the early stopper to be used. Default is ``None``.
        gradient_clipper (:class:`~pydgn.training.callback.gradient_clipper.GradientClipper`): the gradient clipper to be used. Default is ``None``.
        device (str): the device on which to train. Default is ``cpu``.
        plotter (:class:`~pydgn.training.callback.plotter.Plotter`): the plotter to be used. Default is ``None``.
        exp_path (str): the path of the experiment folder. Default is ``None`` but it is always instantiated.
        evaluate_every(int): the frequency of logging epoch results. Default is ``1``.
        store_last_checkpoint (bool): whether to store a checkpoint at the end of each epoch. Allows to resume training from last epoch. Default is ``False``.
    """
    def __init__(self,
                 engine_callback: Callable[...,EngineCallback],
                 model: ModelInterface,
                 loss: Metric,
                 optimizer: Optimizer,
                 scorer: Metric,
                 scheduler: Scheduler=None,
                 early_stopper: EarlyStopper=None,
                 gradient_clipper: GradientClipper=None,
                 device: str='cpu',
                 plotter: Plotter=None,
                 exp_path: str=None,
                 evaluate_every: int=1,
                 store_last_checkpoint: bool=False):
        super().__init__()

        self.engine_callback = engine_callback
        self.model = model
        self.loss_fun = loss
        self.optimizer = optimizer
        self.score_fun = scorer
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.gradient_clipper = gradient_clipper
        self.device = device
        self.plotter = plotter
        self.exp_path = exp_path
        self.evaluate_every = evaluate_every
        self.store_last_checkpoint = store_last_checkpoint
        self.training = False

        self.profiler = Profiler(threshold=1e-5)

        # Now register the callbacks (IN THIS ORDER, WHICH IS KNOWN TO THE USER)
        # Decorate with a profiler
        self.callbacks = [self.profiler(c) for c in [self.loss_fun, self.score_fun,
                                                     self.gradient_clipper, self.optimizer,
                                                     self.early_stopper, self.scheduler, self.plotter] if
                          c is not None]  # filter None callbacks

        # Add an Engine specific callback to profile different passages of _loop
        self.callbacks.append(self.profiler(self.engine_callback(store_last_checkpoint=self.store_last_checkpoint)))

        for c in self.callbacks:
            self.register(c)

        self.state = State(self.model, self.optimizer, self.device)  # Initialize the state
        self.state.update(exp_path=self.exp_path)

    # TODO in general, there are no guarantees that y will be sufficient to determine the task. This may have to be changed in the future.
    def _to_data_list(self, x: torch.Tensor, batch: torch.Tensor, y: torch.Tensor) -> List[Data]:
        """
        Converts model outputs back to a list of Data elements. Useful for incremental architectures.

        Args:
            x (:class:`torch.Tensor`): tensor holding information of different nodes/graphs embeddings
            batch (:class:`torch.Tensor`): the usual PyG batch tensor. Used to split node/graph embeddings graph-wise.
            y (:class:`torch.Tensor`): target labels, used to determine whether the task is graph prediction or node prediction

        Returns:
            a list of PyG Data objects (with only ``x`` and ``y`` attributes)
        """
        data_list = []

        _, counts = torch.unique_consecutive(batch, return_counts=True)
        cumulative = torch.cumsum(counts, dim=0)

        is_graph_prediction = y.shape[0] == len(cumulative)

        y = y.unsqueeze(1) if y.dim() == 1 else y

        data_list.append(Data(x=x[:cumulative[0]],
                              y=y[0].unsqueeze(0) if is_graph_prediction else y[:cumulative[0]]))
        for i in range(1, len(cumulative)):
            g = Data(x=x[cumulative[i - 1]:cumulative[i]],
                     y=y[i].unsqueeze(0) if is_graph_prediction else y[cumulative[i - 1]:cumulative[i]])
            data_list.append(g)

        return data_list

    def _to_list(self, data_list: List[Data],
                 embeddings: Union[Tuple[torch.Tensor], torch.Tensor],
                 batch: torch.Tensor,
                 edge_index: torch.Tensor,
                 y: torch.Tensor) -> List[Data]:
        """
        Extends the ``data_list`` list of PyG Data objects with new samples.

        Args:
            data_list: a list of PyG Data objects (with only ``x`` and ``y`` attributes)
            embeddings (:class:`torch.Tensor`): tensor holding information of different nodes/graphs embeddings
            batch (:class:`torch.Tensor`): the usual PyG batch tensor. Used to split node/graph embeddings graph-wise.
            edge_index:
            y (:class:`torch.Tensor`): target labels, used to determine whether the task is graph prediction or node prediction

        Returns:
            a list of PyG Data objects (with only ``x`` and ``y`` attributes)
        """
        # Crucial: Detach the embeddings to free the computation graph!!
        if isinstance(embeddings, tuple):
            embeddings = tuple([e.detach().cpu() if e is not None else None for e in embeddings])
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        else:
            raise NotImplementedError('Embeddings not understood, should be torch.Tensor or Tuple of torch.Tensor')

        # Convert embeddings back to a list of torch_geometric Data objects
        # Needs also y information to (possibly) use them as a tensor dataset
        # CRUCIAL: remember, the loader could have shuffled the data, so we
        # need to carry y around as well
        if data_list is None:
            data_list = []
        data_list.extend(self._to_data_list(embeddings, batch, y))
        return data_list

    def _num_targets(self, targets: torch.Tensor) -> int:
        """
        Computes the number of targets (**different from the dimension of each target**).

        Args:
            targets (:class:`torch.Tensor`: the ground truth tensor

        Returns:
             an integer with the number of targets to predict
        """
        if targets is None:
            return 0
        assert isinstance(targets, torch.Tensor), "Expecting a tensor as target"
        num_targets = targets.shape[0]
        return num_targets

    def set_device(self):
        """
        Moves the model and the loss metric to the proper device.
        """
        self.model.to(self.device)
        self.loss_fun.to(self.device)

    def set_training_mode(self):
        """
        Sets the model and the internal state in ``TRAINING`` mode
        """
        self.model.train()
        self.training = True

    def set_eval_mode(self):
        """
        Sets the model and the internal state in ``EVALUATION`` mode
        """
        self.model.eval()
        self.training = False


    def _loop_helper(self):
        # Move data to device
        data = self.state.batch_input
        if isinstance(data, list):
            # used by incremental construction
            data = [d.to(self.device) for d in data]
            _data = data[0]
        else:
            # standard case
            data = data.to(self.device)
            _data = data
        batch_idx, edge_index, targets = _data.batch, _data.edge_index, _data.y

        # Helpful when you need to access again the input batch, e.g for some continual learning strategy
        self.state.update(batch_input=data)
        self.state.update(batch_targets=targets)

        num_graphs = _data.num_graphs
        num_nodes = _data.num_nodes
        self.state.update(batch_num_graphs=num_graphs)
        self.state.update(batch_num_nodes=num_nodes)

        # Can we do better? we force the user to use y
        assert targets is not None, "You must provide a target value, even a dummy one, for the engine to infer whether this is a node or link or graph problem."
        num_targets = self._num_targets(targets)

        self.state.update(batch_num_targets=num_targets)

        if self.training:
            self._dispatch(EventHandler.ON_TRAINING_BATCH_START, self.state)
        else:
            self._dispatch(EventHandler.ON_EVAL_BATCH_START, self.state)

        self._dispatch(EventHandler.ON_FORWARD, self.state)

        # EngineCallback will store the outputs in state.batch_outputs
        output = self.state.batch_outputs

        # Change into tuple if not, loss and score expect a tuple
        # where the first element consists of the model's predictions
        if not isinstance(output, tuple):
            output = (output,)
        else:
            if len(output) > 1 and self.state.return_node_embeddings:
                # Embeddings should be in position 2 of the output
                embeddings = output[1]

                data_list = self._to_list(self.state.epoch_data_list, embeddings,
                                          batch_idx, edge_index, targets)

                # I am extending the data list, not replacing! Hence the name "epoch" data list
                self.state.update(epoch_data_list=data_list)

        self._dispatch(EventHandler.ON_COMPUTE_METRICS, self.state)

        if self.training:
            self._dispatch(EventHandler.ON_BACKWARD, self.state)
            self._dispatch(EventHandler.ON_TRAINING_BATCH_END, self.state)
        else:
            self._dispatch(EventHandler.ON_EVAL_BATCH_END, self.state)


    # loop over all data (i.e. computes an epoch)
    def _loop(self, loader: DataLoader):
        """
        Main method that computes a pass over the dataset using the data loader provided.

        Args:
            loader (:class:`torch_geometric.loader.DataLoader`): the loader to be used
        """
        # Reset epoch state (DO NOT REMOVE)
        self.state.update(epoch_data_list=None)
        self.state.update(epoch_loss=None)
        self.state.update(epoch_score=None)
        self.state.update(loader_iterable=iter(loader))

        # Loop over data
        for id_batch in range(len(loader)):

            self.state.update(id_batch=id_batch)
            # EngineCallback will store fetched data in state.batch_input
            self._dispatch(EventHandler.ON_FETCH_DATA, self.state)

            self._loop_helper()

    def _train(self, loader):
        self.set_training_mode()

        self._dispatch(EventHandler.ON_TRAINING_EPOCH_START, self.state)

        self._loop(loader)

        self._dispatch(EventHandler.ON_TRAINING_EPOCH_END, self.state)

        assert self.state.epoch_loss is not None
        loss, score = self.state.epoch_loss, self.state.epoch_score
        return loss, score, None

    def infer(self, loader: DataLoader, set: str) -> Tuple[dict, dict, List[Data]]:
        """
        Performs an evaluation step on the data.

        Args:
            loader (:class:`torch_geometric.loader.DataLoader`): the loader to be used
            set (str): the type of dataset being used, can be ``TRAINING``, ``VALIDATION`` or ``TEST`` (as defined in ``pydgn.static``)

        Returns:
             a tuple (loss dict, score dict, list of :class:`torch_geometric.data.Data` objects with ``x`` and ``y`` attributes only).
             The data list can be used, for instance, in semi-supervised experiments or in incremental architectures
        """
        self.set_eval_mode()
        self.state.update(set=set)

        self._dispatch(EventHandler.ON_EVAL_EPOCH_START, self.state)

        with torch.no_grad():
            self._loop(loader)  # state has been updated

        self._dispatch(EventHandler.ON_EVAL_EPOCH_END, self.state)

        assert self.state.epoch_loss is not None
        loss, score, data_list = self.state.epoch_loss, self.state.epoch_score, self.state.epoch_data_list

        # Add the main loss we want to return as a special key
        main_loss_name = self.loss_fun.get_main_metric_name()
        loss[MAIN_LOSS] = loss[main_loss_name]

        # Add the main score we want to return as a special key
        # Needed by the experimental evaluation framework
        main_score_name = self.score_fun.get_main_metric_name()
        score[MAIN_SCORE] = score[main_score_name]

        return loss, score, data_list

    def train(self,
              train_loader: DataLoader,
              validation_loader: DataLoader=None,
              test_loader: DataLoader=None,
              max_epochs: int=100,
              zero_epoch: bool=False,
              logger: Logger=None) -> Tuple[dict, dict, List[Data], dict, dict, List[Data], dict, dict, List[Data]]:
        """
        Trains the model and regularly evaluates on validation and test data (if given).
        May perform early stopping and checkpointing.

        Args:
            train_loader (:class:`torch_geometric.loader.DataLoader`): the DataLoader associated with training data
            validation_loader (:class:`torch_geometric.loader.DataLoader`): the DataLoader associated with validation data, if any
            test_loader (:class:`torch_geometric.loader.DataLoader`):  the DataLoader associated with test data, if any
            max_epochs (int): maximum number of training epochs. Default is ``100``
            zero_epoch: if ``True``, starts again from epoch 0 and resets optimizer and scheduler states. Default is ``False``
            logger: the logger

        Returns:
             a tuple (train_loss, train_score, train_embeddings, validation_loss, validation_score, validation_embeddings, test_loss, test_score, test_embeddings)
        """

        try:
            # Initialize variables
            val_loss, val_score, val_embeddings_tuple = None, None, None
            test_loss, test_score, test_embeddings_tuple = None, None, None

            self.set_device()

            # Restore training from last checkpoint if possible!
            ckpt_filename = Path(self.exp_path, LAST_CHECKPOINT_FILENAME)
            best_ckpt_filename = Path(self.exp_path, BEST_CHECKPOINT_FILENAME)
            is_early_stopper_ckpt = self.early_stopper.checkpoint if self.early_stopper is not None else False
            # If one changes the options in the config file, the existence of a checkpoint is not enough to
            # decide whether to resume training or not!
            if os.path.exists(ckpt_filename) and (self.store_last_checkpoint or is_early_stopper_ckpt):
                self._restore_checkpoint_and_best_results(ckpt_filename, best_ckpt_filename, zero_epoch)
                log(f'START AGAIN FROM EPOCH {self.state.initial_epoch}', logger)

            self._dispatch(EventHandler.ON_FIT_START, self.state)

            # In case we already have a trained model
            epoch = self.state.initial_epoch

            # Loop over the entire dataset dataset
            for epoch in range(self.state.initial_epoch, max_epochs):
                self.state.update(epoch=epoch)
                self.state.update(return_node_embeddings=False)

                self._dispatch(EventHandler.ON_EPOCH_START, self.state)

                self.state.update(set=TRAINING)
                _, _, _ = self._train(train_loader)

                # Update state with epoch results
                epoch_results = {
                    LOSSES: {},
                    SCORES: {}
                }

                if ((epoch+1) >= self.evaluate_every and (epoch+1) % self.evaluate_every == 0) or epoch == 0:
                    # Compute training output (necessary because on_backward has been called)
                    train_loss, train_score, _ = self.infer(train_loader, TRAINING)

                    # Compute validation output
                    if validation_loader is not None:
                        val_loss, val_score, _ = self.infer(validation_loader, VALIDATION)

                    # Compute test output for visualization purposes only (e.g. to debug an incorrect data split for link prediction)
                    if test_loader is not None:
                        test_loss, test_score, _ = self.infer(test_loader, TEST)

                    epoch_results[LOSSES].update({f'{TRAINING}_{k}': v for k, v in train_loss.items()})
                    epoch_results[SCORES].update({f'{TRAINING}_{k}': v for k, v in train_score.items()})

                    if validation_loader is not None:
                        epoch_results[LOSSES].update({f'{VALIDATION}_{k}': v for k, v in val_loss.items()})
                        epoch_results[SCORES].update({f'{VALIDATION}_{k}': v for k, v in val_score.items()})
                        val_msg_str = f', VL loss: {val_loss} VL score: {val_score}'
                    else:
                        val_msg_str = ''

                    if test_loader is not None:
                        epoch_results[LOSSES].update({f'{TEST}_{k}': v for k, v in test_loss.items()})
                        epoch_results[SCORES].update({f'{TEST}_{k}': v for k, v in test_score.items()})
                        test_msg_str = f', TE loss: {test_loss} TE score: {test_score}'
                    else:
                        test_msg_str = ''

                    # Log performances
                    msg = f'Epoch: {epoch + 1}, TR loss: {train_loss} TR score: {train_score}' + val_msg_str + test_msg_str
                    log(msg, logger)

                # Update state with the result of this epoch
                self.state.update(epoch_results=epoch_results)

                # We can apply early stopping here
                self._dispatch(EventHandler.ON_EPOCH_END, self.state)

                if self.state.stop_training:
                    log(f"Stopping at epoch {self.state.epoch}.", logger)
                    break

            # Needed to indicate that training has ended
            self.state.update(stop_training=True)

            # We reached the max # of epochs, get best scores from the early stopper (if any) and restore best model
            if self.early_stopper is not None:
                ber = self.state.best_epoch_results
                # Restore the model according to the best validation score!
                self.model.load_state_dict(ber[MODEL_STATE])
                self.optimizer.load_state_dict(ber[OPTIMIZER_STATE])
            else:
                self.state.update(best_epoch_results={BEST_EPOCH: epoch})
                ber = self.state.best_epoch_results

            self.state.update(return_node_embeddings=True)

            # Compute training output
            train_loss, train_score, train_embeddings_tuple = self.infer(train_loader, TRAINING)
            # ber[f'{TRAINING}_loss'] = train_loss
            ber[f'{TRAINING}{EMB_TUPLE_SUBSTR}'] = train_embeddings_tuple
            ber.update({f'{TRAINING}_{k}': v for k, v in train_loss.items()})
            ber.update({f'{TRAINING}_{k}': v for k, v in train_score.items()})

            # Compute validation output
            if validation_loader is not None:
                val_loss, val_score, val_embeddings_tuple = self.infer(validation_loader, VALIDATION)
                # ber[f'{VALIDATION}_loss'] = val_loss
                ber[f'{VALIDATION}{EMB_TUPLE_SUBSTR}'] = val_embeddings_tuple
                ber.update({f'{TRAINING}_{k}': v for k, v in val_loss.items()})
                ber.update({f'{TRAINING}_{k}': v for k, v in val_score.items()})

            # Compute test output
            if test_loader is not None:
                test_loss, test_score, test_embeddings_tuple = self.infer(test_loader, TEST)
                # ber[f'{TEST}_loss'] = test_loss
                ber[f'{TEST}{EMB_TUPLE_SUBSTR}'] = test_embeddings_tuple
                ber.update({f'{TEST}_{k}': v for k, v in test_loss.items()})
                ber.update({f'{TEST}_{k}': v for k, v in test_score.items()})

            self._dispatch(EventHandler.ON_FIT_END, self.state)

            self.state.update(return_node_embeddings=False)

            log(f'Chosen is Epoch {ber[BEST_EPOCH]+1} TR loss: {train_loss} TR score: {train_score}, VL loss: {val_loss} VL score: {val_score} '
                f'TE loss: {test_loss} TE score: {test_score}', logger)

            self.state.update(set=None)

        except (KeyboardInterrupt, RuntimeError, FileNotFoundError) as e:
            report = self.profiler.report()
            print(str(e))
            log(str(e), logger)
            log(report, logger)
            raise e
            exit(0)

        # Log profile results
        report = self.profiler.report()
        log(report, logger)

        return train_loss, train_score, train_embeddings_tuple, \
               val_loss, val_score, val_embeddings_tuple, \
               test_loss, test_score, test_embeddings_tuple

    def _restore_checkpoint_and_best_results(self, ckpt_filename, best_ckpt_filename, zero_epoch):
        # When changing exp config from cuda to cpu, cuda will not be available to pytorch (due to Ray management
        # of resources). Hence, we need to specify explicitly the map location as cpu.
        # The other way around (cpu to cuda) poses no problem since GPUs are visible.
        ckpt_dict = torch.load(ckpt_filename, map_location='cpu' if self.device =='cpu' else None)

        self.state.update(initial_epoch=int(ckpt_dict[EPOCH]) + 1 if not zero_epoch else 0)
        self.state.update(stop_training=ckpt_dict[STOP_TRAINING])

        model_state = ckpt_dict[MODEL_STATE]

        # Needed only when moving from cpu to cuda (due to changes in config file). Move all parameters to cuda.
        for param in model_state.keys():
            model_state[param] = model_state[param].to(self.device)

        self.model.load_state_dict(model_state)

        if not zero_epoch:
            optimizer_state = ckpt_dict[OPTIMIZER_STATE]
            self.state.update(optimizer_state=optimizer_state)

        if os.path.exists(best_ckpt_filename):
            best_ckpt_dict = torch.load(best_ckpt_filename, map_location='cpu' if self.device =='cpu' else None)
            self.state.update(best_epoch_results=best_ckpt_dict)

        if self.scheduler is not None and not zero_epoch:
            scheduler_state = ckpt_dict[SCHEDULER_STATE]
            assert scheduler_state is not None
            self.state.update(scheduler_state=scheduler_state)


class DataStreamTrainingEngine(TrainingEngine):

    # loop over all data (i.e. computes an epoch)
    def _loop(self, loader: DataLoader):
        # Reset epoch state (DO NOT REMOVE)
        self.state.update(epoch_data_list=None)
        self.state.update(epoch_loss=None)
        self.state.update(epoch_score=None)
        self.state.update(loader_iterable=iter(loader))

        # Loop over data
        id_batch = 0
        self.state.update(stop_fetching=False)
        while not self.state.stop_fetching:

            self.state.update(id_batch=id_batch)
            id_batch += 1

            # EngineCallback will store fetched data in state.batch_input
            self._dispatch(EventHandler.ON_FETCH_DATA, self.state)

            if self.state.stop_fetching:
                return

            self._loop_helper()


class LinkPredictionSingleGraphEngine(TrainingEngine):
    """
    Specific engine for link prediction tasks. Here, we expect target values in the form of tuples: ``(_, pos_edges, neg_edges)``,
    where ``pos_edges`` and ``neg_edges`` have been generated by the splitter and provided by the data provider.
    """
    def _to_data_list(self, x, batch, y):
        data_list = []
        # Return node embeddings and their original class, if any (or dumb value which is required nonetheless)
        y = y[0]
        data_list.append(Data(x=x, y=y[0]))
        return data_list

    def _to_list(self, data_list, embeddings, batch, edge_index, y):
        assert isinstance(y, list), "Expecting a list of (_, pos_edges, neg_edges)"

        # Crucial: Detach the embeddings to free the computation graph!!
        if isinstance(embeddings, tuple):
            embeddings = tuple([e.detach().cpu() if e is not None else None for e in embeddings])
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        else:
            raise NotImplementedError('Embeddings not understood, should be torch.Tensor or Tuple of torch.Tensor')

        # Convert embeddings back to a list of torch_geometric Data objects
        # Needs also y information to (possibly) use them as a tensor dataset
        # (remember, the loader could have shuffled the data)
        if data_list is None:
            data_list = []
        data_list.extend(self._to_data_list(embeddings, batch, y))
        return data_list

    def _num_targets(self, targets):
        assert isinstance(targets, list), "Expecting a list of (_, pos_edges, neg_edges)"
        # positive links + negative links provided separately
        num_targets = targets[1].shape[1] + targets[2].shape[1]
        # Just a single graph for link prediction
        return num_targets



