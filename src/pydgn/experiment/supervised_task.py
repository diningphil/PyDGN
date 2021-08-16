from pydgn.experiment.experiment import Experiment


class SupervisedTask(Experiment):

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation scores
        :return: (training score, validation score)
        """

        batch_size = self.model_config.supervised_config['batch_size']
        shuffle = self.model_config.supervised_config['shuffle'] \
            if 'shuffle' in self.model_config.supervised_config else True

        # Instantiate the Dataset
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()
        train_loader = dataset_getter.get_inner_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_inner_val(batch_size=batch_size, shuffle=shuffle)

        # Instantiate the Model
        model = self.create_supervised_model(dim_node_features, dim_edge_features, dim_target)

        # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_wrapper = self.create_supervised_wrapper(model)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        test_loss, test_score, _ = supervised_training_wrapper.train(
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=None,
            max_epochs=self.model_config.supervised_config['epochs'],
            logger=logger)
        return train_score, val_score

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training and test score. DO NOT USE THE TEST TO TRAIN OR FOR EARLY STOPPING REASONS!
        :return: (training score, test score)
        """
        batch_size = self.model_config.supervised_config['batch_size']
        shuffle = self.model_config.supervised_config['shuffle'] \
            if 'shuffle' in self.model_config.supervised_config else True

        # Instantiate the Dataset
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()
        train_loader = dataset_getter.get_outer_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_outer_val(batch_size=batch_size, shuffle=shuffle)
        test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle)

        # Instantiate the Model
        model = self.create_supervised_model(dim_node_features, dim_edge_features, dim_target)

        # Instantiate the wrapper (it handles the training loop and the inference phase by abstracting the specifics)
        supervised_training_wrapper = self.create_supervised_wrapper(model)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        test_loss, test_score, _ = supervised_training_wrapper.train(
            train_loader=train_loader,
            validation_loader=val_loader,
            test_loader=test_loader,
            max_epochs=self.model_config.supervised_config['epochs'],
            logger=logger)

        return train_score, test_score
