import os
from pathlib import Path

from pydgn.static import *
from pydgn.training.event.handler import EventHandler
from pydgn.training.engine import TrainingEngine
from torch_geometric.data import Data


def log(msg, logger):
    if logger is not None:
        logger.log(msg)


class GraphSequenceTrainingEngine(TrainingEngine):
    """
    Assumes that the model can return None predictions (i.e., output[0] below)
    whenever no values should be predicted according to the mask field in the
    Data object representing a snapshot
    """

    def __init__(self, engine_callback, model, loss, optimizer, scorer=None,
                 scheduler=None, early_stopper=None, gradient_clipping=None,
                 device='cpu', plotter=None, exp_path=None,
                 log_every=1, store_last_checkpoint=False, reset_eval_state=False):
        super().__init__(engine_callback, model, loss, optimizer, scorer,
                     scheduler, early_stopper, gradient_clipping, device,
                     plotter, exp_path, log_every, store_last_checkpoint)
        self.reset_eval_state = reset_eval_state

    # loop over all data (i.e. computes an epoch)
    def _loop(self, loader):

        # Reset epoch state (DO NOT REMOVE)
        self.state.update(epoch_data_list=None)
        self.state.update(epoch_loss=None)
        self.state.update(epoch_score=None)
        self.state.update(loader_iterable=iter(loader))
        # Initialize the model state at time step 0
        t = 0
        self.state.update(time_step=t)
        all_y = []
        prev_hidden_states = []

        # This is specific to the single graph sequence scenario
        num_timesteps_per_batch = len(loader)
        self.state.update(num_timesteps_per_batch=num_timesteps_per_batch)

        # Loop over data
        for id_batch in range(len(loader)):
            self.state.update(id_batch=id_batch)

            # EngineCallback will store fetched data in state.batch_input
            self._dispatch(EventHandler.ON_FETCH_DATA, self.state)

            # Move data to device
            # data is a list of snapshots
            data = self.state.batch_input

            if self.training:
                self._dispatch(EventHandler.ON_TRAINING_BATCH_START, self.state)
            else:
                self._dispatch(EventHandler.ON_EVAL_BATCH_START, self.state)

            for snapshot in data:

                _snapshot = snapshot
                snapshot = snapshot.to(self.device)

                edge_index, targets = _snapshot.edge_index, _snapshot.y

                # Helpful when you need to access again the input batch, e.g for some continual learning strategy
                self.state.update(batch_input=_snapshot)
                self.state.update(batch_targets=targets)
                self.state.update(time_step=t)

                num_nodes = _snapshot.x.shape[0]
                self.state.update(batch_num_nodes=num_nodes)

                # Can we do better? we force the user to use y
                assert targets is not None, "You must provide a target value, even a dummy one, for the engine to infer whether this is a node or link or graph problem."
                num_targets = self._num_targets(targets)

                self.state.update(batch_num_targets=num_targets)

                self._dispatch(EventHandler.ON_FORWARD, self.state)

                # EngineCallback will store the outputs in state.batch_outputs
                output = self.state.batch_outputs

                # make sure we have embeddings to store as model state
                assert isinstance(output, tuple) and len(output) > 1

                all_y.append(targets)

                # Update previous hidden states
                self.state.update(last_hidden_state=output[1])

                # Save main memory if node embeddings are not needed
                if self.state.return_node_embeddings:
                    prev_hidden_states.append(output[1].detach().cpu())

                self._dispatch(EventHandler.ON_COMPUTE_METRICS, self.state)

                # Increment global time counter
                t = t+1
                self.state.update(time_step=t)

            if self.training:
                self._dispatch(EventHandler.ON_BACKWARD, self.state)

                # After backward detach the last hidden state
                last_hidden_state = self.state.last_hidden_state.detach()
                self.state.update(last_hidden_state=last_hidden_state)

                self._dispatch(EventHandler.ON_TRAINING_BATCH_END, self.state)
            else:
                self._dispatch(EventHandler.ON_EVAL_BATCH_END, self.state)

        if self.state.return_node_embeddings:
            # Model state will hold a list of tensors, one per time step
            data_list = [Data(x=emb_t, y=all_y[i]) for i,emb_t in enumerate(prev_hidden_states)]
            self.state.update(epoch_data_list=data_list)

    def train(self, train_loader, validation_loader=None, test_loader=None, max_epochs=100, zero_epoch=False,
              logger=None):
        """
        Trains the model
        :param train_loader: the DataLoader associated with training data
        :param validation_loader: the DataLoader associated with validation data, if any
        :param test_loader:  the DataLoader associated with test data, if any
        :param max_epochs: maximum number of training epochs
        :param zero_epoch: if True, starts again from epoch 0 and resets optimizer and scheduler states.
        :param logger: a log.Logger for logging purposes
        :return: a tuple (train_loss, train_score, train_embeddings, validation_loss, validation_score, validation_embeddings, test_loss, test_score, test_embeddings)
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

                # Initialize the state of the model
                self.state.update(last_hidden_state=None)

                self._dispatch(EventHandler.ON_EPOCH_START, self.state)

                self.state.update(set=TRAINING)
                _, _, _ = self._train(train_loader)

                # Compute training output (necessary because on_backward has been called)
                train_loss, train_score, _ = self.infer(train_loader, TRAINING)

                # Used when we want to reset the state after performing previous
                # inference. Default should be false since we are dealing with
                # a single temporal graph sequence
                if self.reset_eval_state:
                    self.state.update(last_hidden_state=None)

                # Compute validation output
                if validation_loader is not None:
                    val_loss, val_score, _ = self.infer(validation_loader, VALIDATION)

                # Used when we want to reset the state after performing previous
                # inference. Default should be false since we are dealing with
                # a single temporal graph sequence
                if self.reset_eval_state:
                    self.state.update(last_hidden_state=None)

                # Compute test output for visualization purposes only (e.g. to debug an incorrect data split for link prediction)
                if test_loader is not None:
                    test_loss, test_score, _ = self.infer(test_loader, TEST)

                # Update state with epoch results
                epoch_results = {
                    LOSSES: {},
                    SCORES: {}
                }

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

                # Update state with the result of this epoch
                self.state.update(epoch_results=epoch_results)

                # We can apply early stopping here
                self._dispatch(EventHandler.ON_EPOCH_END, self.state)

                # Log performances
                if epoch % self.log_every == 0 or epoch == 1:
                    msg = f'Epoch: {epoch + 1}, TR loss: {train_loss} TR score: {train_score}' + val_msg_str + test_msg_str
                    log(msg, logger)

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

            # Initialize the state of the model again before the final evaluation
            self.state.update(last_hidden_state=None)

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

            log(f'Chosen is TR loss: {train_loss} TR score: {train_score}, VL loss: {val_loss} VL score: {val_score} '
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
