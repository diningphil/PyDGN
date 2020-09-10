import random
from config.base import Config
from config.utils import s2c


class Experiment:
    """
    Experiment provides useful utilities to support supervised, semi-supervised and incremental tasks on graphs
    """

    def __init__(self, model_configuration, exp_path):
        self.model_config = Config(model_configuration)
        self.exp_path = exp_path

    def _return_class_and_args(self, config, key):
        """
        Returns the class and arguments associated to a specific key in the configuration file.
        :param config: the configuration dictionary
        :param key: a string representing a particular class in the configuration dictionary
        :return: a tuple (class, arguments) or (None, None) if the key is not present in the config dictionary
        """
        if key not in config or config[key] is None:
            return None, None
        elif isinstance(config[key], str):
            return s2c(config[key]), {}
        elif isinstance(config[key], dict):
            return s2c(config[key]['class_name']), config[key]['args'] if 'args' in config[key] else {}
        else:
            raise NotImplementedError('Parameter has not been formatted properly')

    def _create_model(self, dim_node_features, dim_edge_features, dim_target, predictor_classname, config):
        """
        Instantiates a model that implements a fixed interface in models.predictors
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param predictor_classname: string containing the model's class
        :param config: the configuration dictionary
        :return: a generic model, see the models.predictors sub-package.
        """
        return s2c(self.model_config.model)(dim_node_features=dim_node_features,
                                            dim_edge_features=dim_edge_features,
                                            dim_target=dim_target,
                                            predictor_class=s2c(predictor_classname)
                                            if predictor_classname is not None else None,
                                            config=config)

    def create_supervised_model(self, dim_node_features, dim_edge_features, dim_target):
        """ Instantiates a supervised model """
        predictor_classname = self.model_config.supervised_config['predictor'] \
            if 'predictor' in self.model_config.supervised_config else None
        return self._create_model(dim_node_features, dim_edge_features, dim_target, predictor_classname, self.model_config.supervised_config)

    def create_supervised_predictor(self, dim_node_features, dim_edge_features, dim_target):
        """ Directly instantiates a supervised predictor for semi-supervised tasks """
        return s2c(self.model_config.supervised_config['predictor'])(dim_node_features=dim_node_features,
                                                                     dim_edge_features=dim_edge_features,
                                                                     dim_target=dim_target,
                                                                     config=self.model_config.supervised_config)

    def create_unsupervised_model(self, dim_node_features, dim_edge_features, dim_target):
        """ Instantiates an unsupervised model """
        predictor_classname = self.model_config.unsupervised_config['predictor'] \
            if 'predictor' in self.model_config.supervised_config else None
        return self._create_model(dim_node_features, dim_edge_features, dim_target, predictor_classname, self.model_config.unsupervised_config)

    def create_incremental_model(self, dim_node_features, dim_edge_features, dim_target,
                                 depth, prev_outputs_to_consider):
        """
        Instantiates a layer (model) of an incremental architecture. It assumes the config file has a field called
        'arbitrary_function_config' that holds any kind of information for the arbitrary function of an incremental
        architecture
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param depth: current depth of the architecture
        :param prev_outputs_to_consider: A list of previous layers to consider, e.g. [1,2] means the last
        two previous layers.
        :return: an incremental model
        """
        predictor_classname = self.model_config.layer_config['arbitrary_function_config']['predictor'] \
            if 'predictor' in self.model_config.layer_config['arbitrary_function_config'] else None
        self.model_config.layer_config['depth'] = depth
        self.model_config.layer_config['prev_outputs_to_consider'] = prev_outputs_to_consider
        return self._create_model(dim_node_features, dim_edge_features, dim_target,
                                  predictor_classname, self.model_config.layer_config)

    def _create_wrapper(self, config, model, wrapper_classname, device, log_every, checkpoint):
        """
        Instantiates the training engine (see training.core.engine). It looks for pre-defined fields in the
        configuration file, i.e. 'loss', 'scorer', 'optimizer', 'scheduler', 'gradient_clipping', 'early_stopper' and
        'plotter', all of which should be classes implementing the EventHandler interface
        (see training.core.event.handler subpackage)
        :param config: the configuration dictionary
        :param model: the core model that need be trained
        :param wrapper_classname: string containing the dotted path of the class associated with the training engine
        :param device:
        :param log_every:
        :param checkpoint:
        :return: a training engine implementing (see training.core.engine.TrainingEngine for the base class
        doing most of the work)
        """

        loss_class, loss_args = self._return_class_and_args(config, 'loss')
        loss = loss_class(**loss_args) if loss_class is not None else None

        scorer_class, scorer_args = self._return_class_and_args(config, 'scorer')
        scorer = scorer_class(**scorer_args) if scorer_class is not None else None

        optim_class, optim_args = self._return_class_and_args(config, 'optimizer')
        optimizer = optim_class(model=model, **optim_args) if optim_class is not None else None

        sched_class, sched_args = self._return_class_and_args(config, 'scheduler')
        if sched_args is not None:
            sched_args['optimizer'] = optimizer.optimizer
        scheduler = sched_class(**sched_args) if sched_class is not None else None

        grad_clip_class, grad_clip_args = self._return_class_and_args(config, 'gradient_clipping')
        grad_clipper = grad_clip_class(**grad_clip_args) if grad_clip_class is not None else None

        early_stop_class, early_stop_args = self._return_class_and_args(config, 'early_stopper')
        early_stopper = early_stop_class(**early_stop_args) if early_stop_class is not None else None

        plot_class, plot_args = self._return_class_and_args(config, 'plotter')
        plotter = plot_class(exp_path=self.exp_path, **plot_args) if plot_class is not None else None

        wrapper = s2c(wrapper_classname)(model=model, loss=loss,
                                         optimizer=optimizer, scorer=scorer, scheduler=scheduler,
                                         early_stopper=early_stopper, gradient_clipping=grad_clipper,
                                         device=device, plotter=plotter, exp_path=self.exp_path, log_every=log_every,
                                         checkpoint=checkpoint)
        return wrapper

    def create_supervised_wrapper(self, model):
        """ Instantiates the training engine by using the 'supervised_config' key in the config file """
        device = self.model_config.device
        wrapper_classname = self.model_config.supervised_config['wrapper']
        log_every = self.model_config.log_every
        checkpoint = self.model_config.supervised_config.get('checkpoint', False)
        return self._create_wrapper(self.model_config.supervised_config, model, wrapper_classname, device, log_every, checkpoint)

    def create_unsupervised_wrapper(self, model):
        """ Instantiates the training engine by using the 'unsupervised_config' key in the config file """
        device = self.model_config.device
        log_every = self.model_config.log_every
        wrapper_classname = self.model_config.unsupervised_config['wrapper']
        checkpoint = self.model_config.unsupervised_config.get('checkpoint', False)
        return self._create_wrapper(self.model_config.unsupervised_config, model, wrapper_classname, device, log_every, checkpoint)

    def create_incremental_wrapper(self, model):
        """ Instantiates the training engine by using the 'layer_config' key in the config file """
        device = self.model_config.device
        log_every = self.model_config.log_every
        wrapper_classname = self.model_config.layer_config['wrapper']
        checkpoint = self.model_config.layer_config.get('checkpoint', False)
        return self._create_wrapper(self.model_config.layer_config, model, wrapper_classname, device, log_every, checkpoint)

    def run_valid(self, get_train_val, logger, other=None):
        """
        This function returns the training and validation scores
        :return: (training score, validation score)
        """
        raise NotImplementedError('You must implement this function!')

    def run_test(self, get_train_val, get_test, logger, other=None):
        """
        This function returns the training and test score. DO NOT USE THE TEST TO TRAIN OR FOR EARLY STOPPING REASONS!
        :return: (training score, test score)
        """
        raise NotImplementedError('You must implement this function!')
