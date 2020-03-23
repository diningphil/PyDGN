from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from experiments.experiment import Experiment


class SemiSupervisedTask(Experiment):

    def __init__(self, model_configuration, exp_path):
        super(SemiSupervisedTask, self).__init__(model_configuration, exp_path)

    def run_valid(self, dataset_getter, logger, other=None):
        """
        This function returns the training and validation scores
        :return: (training score, validation score)
        """
        unsupervised_config = self.model_config.unsupervised_config
        supervised_config = self.model_config.supervised_config

        # -------------------------------------- PART I: Unsupervised Training --------------------------------------- #

        batch_size = unsupervised_config['batch_size']
        shuffle = unsupervised_config['shuffle'] \
            if 'shuffle' in unsupervised_config else True

        # Instantiate the Dataset
        dim_features = dataset_getter.get_dim_features()
        dim_target = dataset_getter.get_dim_target()
        train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=shuffle)

        # Istantiate the Model
        model = self.create_unsupervised_model(dim_features, None)

        # Istantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
        unsupervised_training_wrapper = self.create_unsupervised_wrapper(model)

        _, _, train_data_list, \
        _, _, val_data_list, \
        _, _, _ = unsupervised_training_wrapper.train(train_loader=train_loader,
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
        collate_fn=lambda data_list: Batch.from_data_list(data_list)

        # Instantiate the Embedding Dataset for supervised learning
        train_loader = DataLoader(train_data_list, batch_size, shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_data_list, batch_size, shuffle, collate_fn=collate_fn)

        # Instantiate the Model
        model = self.create_supervised_predictor(embedding_size, dim_target)

        # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_wrapper = self.create_supervised_wrapper(model)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        _, _, _ = supervised_training_wrapper.train(train_loader=train_loader,
                                                       validation_loader=val_loader,
                                                       test_loader=None,
                                                       max_epochs=supervised_config['epochs'],
                                                       logger=logger)

        return float(train_score), float(val_score)

    def run_test(self, dataset_getter, logger, other=None):
        """
        This function returns the training and test score. DO NOT USE THE TEST TO TRAIN OR FOR EARLY STOPPING REASONS!
        :return: (training score, test score)
        """
        unsupervised_config = self.model_config.unsupervised_config
        supervised_config = self.model_config.supervised_config

        # -------------------------------------- PART I: Unsupervised Training --------------------------------------- #

        batch_size = unsupervised_config['batch_size']
        shuffle = unsupervised_config['shuffle'] \
            if 'shuffle' in unsupervised_config else True

        # Instantiate the Dataset
        dim_features = dataset_getter.get_dim_features()
        dim_target = dataset_getter.get_dim_target()
        train_loader = dataset_getter.get_outer_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_outer_val(batch_size=batch_size, shuffle=shuffle)
        test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle)

        # Instantiate the Model
        model = self.create_unsupervised_model(dim_features, None)

        # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
        unsupervised_training_wrapper = self.create_unsupervised_wrapper(model)

        _, _, train_data_list, \
        _, _, val_data_list, \
        _, _, test_data_list  = unsupervised_training_wrapper.train(train_loader=train_loader,
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
        collate_fn=lambda data_list: Batch.from_data_list(data_list)

        # Instantiate the Embedding Dataset for supervised learning
        train_loader = DataLoader(train_data_list, batch_size, shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_data_list, batch_size, shuffle, collate_fn=collate_fn)
        test_loader = DataLoader(test_data_list, batch_size, shuffle, collate_fn=collate_fn)

        # Instantiate the Model
        model = self.create_supervised_predictor(embedding_size, dim_target)

        # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_wrapper = self.create_supervised_wrapper(model)

        train_loss, train_score, _, \
        _, _, _, \
        test_loss, test_score, _ = supervised_training_wrapper.train(train_loader=train_loader,
                                                       validation_loader=val_loader,
                                                       test_loader=test_loader,
                                                       max_epochs=supervised_config['epochs'],
                                                       logger=logger)

        return float(train_score), float(test_score)
