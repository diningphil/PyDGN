import os

from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from pydgn.experiment.experiment import Experiment
from pydgn.static import LOSS, SCORE


class SemiSupervisedTask(Experiment):
    """
    Class that implements a semi-supervised experiment. There is an ``unsupervised_config`` field in the configuration
    file that is used, together with the field ``model``, to produce unsupervised embeddings. These are later used
    by a **readout** extracted from a ``supervised_config`` field in the configuration file to perform the supervised
    task.
    """
    def __init__(self, model_configuration, exp_path, exp_seed):
        super(SemiSupervisedTask, self).__init__(model_configuration, exp_path, exp_seed)
        self.root_exp_path = exp_path  # to distinguish unsup. and sup. exp paths

    def run_valid(self, dataset_getter, logger):
        unsupervised_config = self.model_config.unsupervised_config
        supervised_config = self.model_config.supervised_config

        # -------------------------------------- PART I: Unsupervised Training --------------------------------------- #

        batch_size = unsupervised_config['batch_size']
        shuffle = unsupervised_config['shuffle'] \
            if 'shuffle' in unsupervised_config else True

        # Instantiate the Dataset
        train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=shuffle)

        # Call this after the loaders: the datasets may need to be instantiated with additional parameters
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Change exp path to allow Stop & Resume
        self.exp_path = os.path.join(self.root_exp_path, f'unsupervised_training')

        # Istantiate the Model
        model = self.create_unsupervised_model(dim_node_features, dim_edge_features, dim_target)

        # Istantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
        unsupervised_training_engine = self.create_unsupervised_engine(model)

        _, _, train_data_list, \
        _, _, val_data_list, \
        _, _, _ = unsupervised_training_engine.train(train_loader=train_loader,
                                                      validation_loader=val_loader,
                                                      test_loader=None,
                                                      max_epochs=unsupervised_config['epochs'],
                                                      logger=logger)

        # --------------------------------------- PART II: Supervised Training --------------------------------------- #

        # Get the embedding size from the first graph in the dataset
        embedding_size = train_data_list[0].x.shape[1]

        # Get information from the supervised configuration
        batch_size = supervised_config['batch_size']
        shuffle = supervised_config['shuffle'] \
            if 'shuffle' in supervised_config else True
        collate_fn = lambda data_list: Batch.from_data_list(data_list)

        # Instantiate the Embedding Dataset for supervised learning
        train_loader = DataLoader(train_data_list, batch_size, shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_data_list, batch_size, shuffle, collate_fn=collate_fn)

        # Change exp path to allow Stop & Resume
        self.exp_path = os.path.join(self.root_exp_path, f'supervised_training')

        # Instantiate the Model
        model = self.create_supervised_model(dim_node_features=embedding_size, dim_edge_features=0,
                                             dim_target=dim_target)

        # Instantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_engine = self.create_supervised_engine(model)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        _, _, _ = supervised_training_engine.train(train_loader=train_loader,
                                                    validation_loader=val_loader,
                                                    test_loader=None,
                                                    max_epochs=supervised_config['epochs'],
                                                    logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        return train_res, val_res

    def run_test(self, dataset_getter, logger):
        unsupervised_config = self.model_config.unsupervised_config
        supervised_config = self.model_config.supervised_config

        # -------------------------------------- PART I: Unsupervised Training --------------------------------------- #

        batch_size = unsupervised_config['batch_size']
        shuffle = unsupervised_config['shuffle'] \
            if 'shuffle' in unsupervised_config else True

        # Instantiate the Dataset
        train_loader = dataset_getter.get_outer_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_outer_val(batch_size=batch_size, shuffle=shuffle)
        test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle)

        # Call this after the loaders: the datasets may need to be instantiated with additional parameters
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Change exp path to allow Stop & Resume
        self.exp_path = os.path.join(self.root_exp_path, f'unsupervised_training')

        # Instantiate the Model
        model = self.create_unsupervised_model(dim_node_features, dim_edge_features, dim_target)

        # Instantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
        unsupervised_training_engine = self.create_unsupervised_engine(model)

        _, _, train_data_list, \
        _, _, val_data_list, \
        _, _, test_data_list = unsupervised_training_engine.train(train_loader=train_loader,
                                                                   validation_loader=val_loader,
                                                                   test_loader=test_loader,
                                                                   max_epochs=unsupervised_config['epochs'],
                                                                   logger=logger)

        # --------------------------------------- PART II: Supervised Training --------------------------------------- #

        # Get the embedding size from the first graph in the dataset
        embedding_size = train_data_list[0].x.shape[1]

        # Get information from the supervised configuration
        batch_size = supervised_config['batch_size']
        shuffle = supervised_config['shuffle'] \
            if 'shuffle' in supervised_config else True
        collate_fn = lambda data_list: Batch.from_data_list(data_list)

        # Instantiate the Embedding Dataset for supervised learning
        train_loader = DataLoader(train_data_list, batch_size, shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_data_list, batch_size, shuffle, collate_fn=collate_fn)
        test_loader = DataLoader(test_data_list, batch_size, shuffle, collate_fn=collate_fn)

        # Change exp path to allow Stop & Resume
        self.exp_path = os.path.join(self.root_exp_path, f'supervised_training')

        # Instantiate the Model
        model = self.create_supervised_model(dim_node_features=embedding_size, dim_edge_features=0,
                                             dim_target=dim_target)

        # Instantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_engine = self.create_supervised_engine(model)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        test_loss, test_score, _ = supervised_training_engine.train(train_loader=train_loader,
                                                                     validation_loader=val_loader,
                                                                     test_loader=test_loader,
                                                                     max_epochs=supervised_config['epochs'],
                                                                     logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}
        return train_res, val_res, test_res
