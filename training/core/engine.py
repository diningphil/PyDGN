import time
import torch
from training.core.event.dispatcher import EventDispatcher
from training.core.event.state import State
from utils.batch_utils import to_data_list, extend_lists, to_tensor_lists


def log(msg, logger):
    if logger is not None:
        logger.log(msg)
    print(msg)


class TrainingEngine(EventDispatcher):
    """
    This is the most important class when it comes to training a model. It implements the EventDispatcher interface,
    which means that after registering some callbacks in a given order, it will proceed to trigger specific events that
    will result in the shared State object being updated by the callbacks. Callbacks implement the EventHandler
    interface, and they receive the shared State object when any event is triggered. Knowing the order in which
    callbacks are called is important.
    The order is: loss function - score function - gradient clipper - optimizer - early stopper - scheduler - plotter
    """

    def __init__(self, model, loss, optimizer, scorer=None,
                 scheduler=None, early_stopper=None, gradient_clipping=None, device='cpu', plotter=None):
        """
        Initializes the engine
        :param model: the model to be trained
        :param loss: an object of a subclass of training.core.callback.Loss
        :param optimizer: an object of a subclass of training.core.callback.Optimizer
        :param scorer: an object of a subclass of training.core.callback.Scorer
        :param scheduler: an object of a subclass of training.core.callback.Scheduler
        :param early_stopper: an object of a subclass of training.core.callback.EarlyStopper
        :param gradient_clipping: an object of a subclass of training.core.callback.GradientClipper
        :param device: the device on which to train.
        :param plotter: an object of a subclass of training.core.callback.Plotter
        """
        super().__init__()

        self.model = model
        self.loss_fun = loss
        self.optimizer = optimizer
        self.score_fun = scorer
        self.scheduler = scheduler
        self.early_stopper = early_stopper
        self.gradient_clipping = gradient_clipping
        self.device = device
        self.plotter = plotter

        # Now register the callbacks (IN THIS ORDER, WHICH IS KNOWN TO THE USER)
        self.callbacks = [c for c in [self.loss_fun, self.score_fun,
                          self.gradient_clipping, self.optimizer,
                          self.early_stopper, self.scheduler, self.plotter] if c is not None]  # filter None callbacks
        for c in self.callbacks:
            self.register(c)

    def _to_list(self, data_list, embeddings, batch, edge_index, y):
        if isinstance(embeddings, tuple):
            embeddings = tuple([e.detach() if e is not None else None for e in embeddings])
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach()
        else:
            raise NotImplementedError('Embeddings not understood, should be Tensor or Tuple of Tensors')

        if data_list is None:
            data_list = []
        # Crucial: Detach the embeddings to free the computation graph!!
        data_list.extend(to_data_list(embeddings, batch, y))
        return data_list

    def set_device(self):
        self.model.to(self.device)
        self.loss_fun.to(self.device)

    def set_training_mode(self):
        self.model.train()
        self.state.update(mode=State.TRAINING)
        self.state.update(set='training')

    def set_eval_mode(self):
        self.model.eval()
        self.state.update(mode=State.EVALUATION)
    
    def set_validation_mode(self):
        self.set_eval_mode()
        self.state.update(set='validation')

    def set_test_mode(self):
        self.set_eval_mode()
        self.state.update(set='test')

    # loop over all data (i.e. computes an epoch)
    def _loop(self, loader, return_node_embeddings=False):

        assert self.state.mode in [State.TRAINING, State.EVALUATION], "Mode has not been set correctly"
        training = self.state.mode == State.TRAINING

        start_timer = time.time()
        
        if training:
            self.set_training_mode()
        else:
            self.set_eval_mode()  # Mode has already been set in the train method

        # Reset epoch state (DO NOT REMOVE)
        self.state.update(epoch_data_list=None)
        self.state.update(epoch_loss=None)
        self.state.update(epoch_score=None)

        # Loop over data
        for data in loader:

            # Move data to device
            if isinstance(data, list):  # for incremental construction
                data = [d.to(self.device) for d in data]
                orig_data = data[0]
            else:
                data = data.to(self.device)
                orig_data = data

            self.state.update(batch_num_graphs=orig_data.num_graphs)

            if training:
                self._dispatch('on_training_batch_start', self.state)
            else:
                self._dispatch('on_eval_batch_start', self.state)

            # Forward pass
            output = self.model.forward(data)

            # Change into tuple if not, losses expect a tuple of elements
            # the first of which consists of the model's predictions
            if not isinstance(output, tuple):
                output = (output,)
            else:
                if len(output) > 1 and return_node_embeddings:
                    # Embeddings should be in position 2 of the output
                    embeddings = output[1]
                    data_list = self._to_list(self.state.epoch_data_list, embeddings,
                                                         orig_data.batch, orig_data.edge_index, orig_data.y)
                    # I am extending the data list, not replacing! Hence the name "epoch" data list
                    self.state.update(epoch_data_list=data_list)

            # Need to store loss, score and (possibly) embeddings
            loss = self.loss_fun(orig_data.y, *output)
            score = self.score_fun(orig_data.y, *output).detach() if self.score_fun else None

            self.state.update(batch_loss=loss)
            self.state.update(batch_score=score)
            self.state.update(batch_targets=orig_data.y)
            self.state.update(batch_predictions=output[0].detach() if output[0] is not None else None)

            if training:
                self._dispatch('on_backward', self.state)
                self._dispatch('on_training_batch_end', self.state)
            else:
                self._dispatch('on_eval_batch_end', self.state)

        # Reset variables to avoid potential graph memory leaks        
        self.state.update(batch_targets=None)
        self.state.update(batch_prediction=None)

        end_timer = time.time()
        self.state.update(epoch_elapsed=end_timer-start_timer)

    def _train(self, loader):

        self._dispatch('on_training_epoch_start', self.state)

        self._loop(loader, return_node_embeddings=False)

        self._dispatch('on_training_epoch_end', self.state)

        assert self.state.epoch_loss is not None
        loss, score = self.state.epoch_loss, self.state.epoch_score
        return loss, score, None

    def infer(self, loader, return_node_embeddings=True):
        """
        Performs an inference step
        :param loader: the DataLoader
        :param return_node_embeddings: whether the model should compute node embeddings or not
        :return: a tuple (loss, score, embeddings)
        """

        self._dispatch('on_eval_epoch_start', self.state)

        self._loop(loader, return_node_embeddings=return_node_embeddings)  # state has been updated

        self._dispatch('on_eval_epoch_end', self.state)

        assert self.state.epoch_loss is not None
        loss, score, data_list = self.state.epoch_loss, self.state.epoch_score, self.state.epoch_data_list
        return loss, score, data_list

        return loss, score, embeddings

    def train(self, train_loader, validation_loader=None, test_loader=None, max_epochs=100, logger=None, log_every=1):
        """
        Trains the model
        :param train_loader: the DataLoader associated with training data
        :param validation_loader: the DataLoader associated with validation data, if any
        :param test_loader:  the DataLoader associated with test data, if any
        :param max_epochs: maximum number of training epochs
        :param logger: a log.Logger for logging purposes
        :param log_every: interval of epochs between one log and the next
        :return: a tuple (train_loss, train_score, train_embeddings, validation_loss, validation_score, validation_embeddings, test_loss, test_score, test_embeddings)
        """

        # Initialize variables
        val_loss, val_score, val_embeddings_tuple = None, None, None
        test_loss, test_score, test_embeddings_tuple = None, None, None

        self.state = State(self.model, self.optimizer)  # Initialize the state
        self.set_device()
        self._dispatch('on_fit_start', self.state)

        # Loop over the entire dataset dataset
        for epoch in range(1, max_epochs+1):

            self.state.update(epoch=epoch)

            # logger.log(f"Starting epoch {self.state.epoch}.")
            self._dispatch('on_epoch_start', self.state)

            self.set_training_mode()
            # logger.log("Training mode set.")
            _, _, _ = self._train(train_loader)

            # Compute training output (necessary because on_backward has been called)
            self.set_training_mode()
            self.set_eval_mode()
            train_loss, train_score, _ = self.infer(train_loader, return_node_embeddings=False)

            if self.early_stopper:  # Early stopping can also be performed on training!
                # Compute validation output
                if validation_loader is not None:
                    self.set_validation_mode()
                    val_loss, val_score, _ = self.infer(validation_loader, return_node_embeddings=False)

            # Update state with epoch results
            epoch_results = {
                'train_loss': train_loss,
                'train_score': train_score,
                'val_loss': val_loss,  # can be None
                'val_score': val_score  # can be None
            }
                
            # Update state with the result of this epoch
            self.state.update(epoch_results=epoch_results)

            # We can apply early stopping here
            self._dispatch('on_epoch_end', self.state)

            # Log performances
            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR score: {train_score}, VL loss: {val_loss} VL score: {val_score} ' \
                        f'TE loss: {test_loss} TE score: {test_score}'
                print(msg)
                logger.log(msg)

            if self.state.stop_training:
                print(f"Stopping at epoch {self.state.epoch}.")
                logger.log(f"Stopping at epoch {self.state.epoch}.")
                break

        # We reached the max # of epochs, get best scores from the early stopper (if any) and restore best model
        if self.early_stopper is not None:

            ber = self.state.best_epoch_results
            best_epoch = ber['best_epoch']

            # Restore the model according to the best validation score!
            self.model.load_state_dict(ber['model_state'])
            self.optimizer.load_state_dict(ber['optimizer_state'])

        else:
            self.state.update(best_epoch_results={'best_epoch': epoch})
            ber = self.state.best_epoch_results
        
        # Compute training output
        self.set_training_mode()
        self.set_eval_mode()
        train_loss, train_score, train_embeddings_tuple = self.infer(train_loader)
        ber['train_loss'] = train_loss
        ber['train_score'] = train_score
        ber['train_embeddings_tuple'] = train_embeddings_tuple

        # Compute validation output
        if validation_loader is not None:
            self.set_validation_mode()
            val_loss, val_score, val_embeddings_tuple = self.infer(validation_loader)
            ber['val_loss'] = val_loss
            ber['val_score'] = val_score
            ber['val_embeddings_tuple'] = val_embeddings_tuple

        # Compute test output
        if test_loader is not None:
            self.set_test_mode()
            test_loss, test_score, test_embeddings_tuple = self.infer(test_loader)
            ber['test_loss'] = test_loss
            ber['test_score'] = test_score
            ber['test_embeddings_tuple'] = test_embeddings_tuple
        
        self._dispatch('on_fit_end', self.state)

        log(f'Chosen is TR loss: {train_loss} TR score: {train_score}, VL loss: {val_loss} VL score: {val_score} TE loss: {test_loss} TE score: {test_score}', logger)
        return train_loss, train_score, train_embeddings_tuple, \
                val_loss, val_score, val_embeddings_tuple, \
                test_loss, test_score, test_embeddings_tuple


class IncrementalTrainingEngine(TrainingEngine):
    def __init__(self, model, loss, **kwargs):
        super().__init__(model, loss, **kwargs)

    def infer(self, loader, return_node_embeddings=True):
        """
        Extends the infer method of Training Engine to update the State variable "compute_intermediate_outputs".
        :param loader:
        :param return_node_embeddings:
        :return:
        """
        self.state.update(compute_intermediate_outputs=return_node_embeddings)
        return super().infer(loader, return_node_embeddings)

    def _to_list(self, data_list, embeddings, batch, edge_index, y):

        if isinstance(embeddings, tuple):
            embeddings = tuple([e.detach() if e is not None else None for e in embeddings])
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach()
        else:
            raise NotImplementedError('Embeddings not understood, should be Tensor or Tuple of Tensors')

        data_list = extend_lists(data_list, to_tensor_lists(embeddings, batch, edge_index, y))
        return data_list