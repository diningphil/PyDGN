import os
import json
import operator
import numpy as np
import concurrent.futures
from copy import deepcopy

from log.Logger import Logger


class KFoldSelection:
    """
    Class implementing a K-Fold technique to do Model Selection
    """

    def __init__(self, inner_folds, max_processes, higher_is_better=True):
        """
        Initialized a K-Fold procedure for Model Selection (selection of best hyper-parameters configurations)
        :param inner_folds: The number K of inner TRAIN/VALIDATION folds. You should have generated the splits accordingly
        :param max_processes: the number of parallel processes to run all possible hyper-parameters configurations.
        """
        self.inner_folds = inner_folds
        self.max_processes = max_processes
        if higher_is_better:
            self.operator = operator.gt
        else:
            self.operator = operator.lt


        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'

    def process_results(self, folder, no_configurations):
        """
        Chooses the best hyper-parameters configuration using the HIGHEST validation mean score
        :param folder: a folder which holds all configurations results after K folds
        :param no_configurations: number of possible configurations
        """

        best_avg_vl = -float('inf')
        best_std_vl = 100.

        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(folder, self._CONFIG_BASE + str(i), self._CONFIG_FILENAME)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                avg_vl = config_dict['avg_VL_score']
                std_vl = config_dict['std_VL_score']

                if self.operator(avg_vl, best_avg_vl) or (best_avg_vl == avg_vl and best_std_vl > std_vl):
                    best_i = i
                    best_avg_vl = avg_vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        print('Model selection winner for experiment', folder, 'is config ', best_i, ':')
        for k in best_config.keys():
            print('\t', k, ':', best_config[k])

        return best_config

    def model_selection(self, dataset_getter, experiment_class, exp_path, model_configs, debug=False, other=None):
        """
        Performs model selection by launching each configuration in parallel, unless debug is True. Each process
        trains the same configuration for each inner fold.
        :param dataset_getter: an object that handles the creation of dataloaders. See datasets.provider.DataProvider
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param exp_path: The folder in which to store all results
        :param model_configs: an object storing all possible model configurations, e.g. config.base.Grid
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        :return: a dictionary holding the best configuration
        """

        exp_path = exp_path
        KFOLD_FOLDER = os.path.join(exp_path, str(self.inner_folds) + '_FOLD_MS')

        if not os.path.exists(KFOLD_FOLDER):
            os.makedirs(KFOLD_FOLDER)

        config_id = 0

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes)
        for config in model_configs:

            # I need to make a copy of this dictionary
            # It seems it gets shared between processes!
            cfg = deepcopy(config)

            # Create a separate folder for each experiment
            exp_config_name = os.path.join(KFOLD_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)

            json_config = os.path.join(exp_config_name, self._CONFIG_FILENAME)
            if not os.path.exists(json_config):
                if not debug:
                    pool.submit(self._model_selection_helper, dataset_getter, experiment_class, cfg,
                                exp_config_name, other)
                else:  # DEBUG
                    self._model_selection_helper(dataset_getter, experiment_class, cfg,
                                                 exp_config_name, other)
            else:
                # Do not recompute experiments for this fold.
                print(f"Config {json_config} already present! Skipping the experiment")

            config_id += 1

        pool.shutdown()

        best_config = self.process_results(KFOLD_FOLDER, config_id)

        with open(os.path.join(KFOLD_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(best_config, fp)

        return best_config

    def _model_selection_helper(self, dataset_getter, experiment_class, config, exp_config_name, other=None):
        """
        Helper method that runs model selection for a particular configuration. The configuration is repeated for each
        inner fold. Validation scores are averaged to compute the validation score of a single hyper-parameter
        configuration.
        :param dataset_getter: an object that handles the creation of dataloaders. See datasets.provider.DataProvider
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param config: Dictionary holding a specific hyper-parameter configuration
        :param exp_config_name: The folder in which to store all results for a specific configuration
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        """

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='a')

        logger.log('Configuration: ' + str(config))

        config_filename = os.path.join(exp_config_name, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        k_fold_dict = {
            'config': config,
            'folds': [{} for _ in range(self.inner_folds)],
            'avg_TR_score': 0.,
            'avg_VL_score': 0.,
            'std_TR_score': 0.,
            'std_VL_score': 0.
        }

        for k in range(self.inner_folds):

            dataset_getter.set_inner_k(k)

            fold_exp_folder = os.path.join(exp_config_name, 'FOLD_' + str(k + 1))
            # Create the experiment object which will be responsible for running a specific experiment
            experiment = experiment_class(config, fold_exp_folder)

            training_score, validation_score = experiment.run_valid(dataset_getter, logger, other)

            logger.log(str(k+1) + ' split, TR Score: ' + str(training_score) +
                       ' VL Score: ' + str(validation_score))

            k_fold_dict['folds'][k]['TR_score'] = training_score
            k_fold_dict['folds'][k]['VL_score'] = validation_score

        tr_scores = np.array([k_fold_dict['folds'][k]['TR_score'] for k in range(self.inner_folds)])
        vl_scores = np.array([k_fold_dict['folds'][k]['VL_score'] for k in range(self.inner_folds)])

        k_fold_dict['avg_TR_score'] = tr_scores.mean()
        k_fold_dict['std_TR_score'] = tr_scores.std()
        k_fold_dict['avg_VL_score'] = vl_scores.mean()
        k_fold_dict['std_VL_score'] = vl_scores.std()

        logger.log('TR avg is ' + str(k_fold_dict['avg_TR_score']) + ' std is ' + str(k_fold_dict['std_TR_score']) +
                   ' VL avg is ' + str(k_fold_dict['avg_VL_score']) + ' std is ' + str(k_fold_dict['std_VL_score']))

        with open(config_filename, 'w') as fp:
            json.dump(k_fold_dict, fp)
