import os
import time
import json
import pickle
import operator
from copy import deepcopy
import torch.multiprocessing as mp
import concurrent.futures

import numpy as np
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
        self.higher_is_better = higher_is_better
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

        best_avg_vl = -float('inf') if self.higher_is_better else float('inf')
        best_std_vl = float('inf')

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

        return best_config, best_i

    def model_selection(self, outer_fold_id, dataset_getter, experiment_class, exp_path, model_configs, debug, other, snd_queue):
        """
        Performs model selection by launching each configuration in parallel, unless debug is True. Each process
        trains the same configuration for each inner fold.
        :param outer_fold_id: id of the outer assessment fold
        :param dataset_getter: an object that handles the creation of dataloaders. See datasets.provider.DataProvider
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param exp_path: The folder in which to store all results
        :param model_configs: an object storing all possible model configurations, e.g. config.base.Grid
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        :param snd_queue: a queue to inform the main process about the progress
        :return: a dictionary holding the best configuration
        """

        exp_path = exp_path
        KFOLD_FOLDER = os.path.join(exp_path, str(self.inner_folds) + '_FOLD_MS')

        if not os.path.exists(KFOLD_FOLDER):
            os.makedirs(KFOLD_FOLDER)

        config_id = 0
        # See https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note
        mp_context = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes, mp_context=mp_context) as pool:
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
                        f = pool.submit(self._model_selection_helper, outer_fold_id, dataset_getter, experiment_class, cfg,
                                    config_id, exp_config_name, other, snd_queue)
                    else:  # DEBUG
                        self._model_selection_helper(outer_fold_id, dataset_getter, experiment_class, cfg,
                                                     config_id, exp_config_name, other, snd_queue)
                else:
                    # Do not recompute experiments for these folds.
                    # print(f"Config {json_config} already present! Skipping the experiment")
                    # Inform the main process about experiment completion
                    for k in range(self.inner_folds):
                        for msg_type in ["START_CONFIG", "END_CONFIG"]:
                            msg = dict(type=msg_type, outer_fold=outer_fold_id, config_id=config_id, inner_fold=k)
                            snd_queue.put(msg)

                config_id += 1

        best_config, best_config_id = self.process_results(KFOLD_FOLDER, config_id)

        with open(os.path.join(KFOLD_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(dict(best_config=best_config, best_config_id=best_config_id), fp)

        return best_config

    def _model_selection_helper(self, outer_fold_id, dataset_getter, experiment_class, config, config_id, exp_config_name, other, snd_queue):
        """
        Helper method that runs model selection for a particular configuration. The configuration is repeated for each
        inner fold. Validation scores are averaged to compute the validation score of a single hyper-parameter
        configuration.
        :param outer_fold_id: id of the outer assessment fold
        :param dataset_getter: an object that handles the creation of dataloaders. See datasets.provider.DataProvider
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param config: Dictionary holding a specific hyper-parameter configuration
        :param config_id: Configuration ID
        :param exp_config_name: The folder in which to store all results for a specific configuration
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        :param snd_queue: a queue to inform the main process about the progress
        """
        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='a')

        logger.log('Configuration: \n' + str(config))

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

            # Inform the main process about experiment completion
            msg = dict(type="START_CONFIG", outer_fold=outer_fold_id, config_id=config_id, inner_fold=k)
            snd_queue.put(msg)

            training_score, validation_score = experiment.run_valid(dataset_getter, logger, other)

            logger.log(str(k+1) + ' split, TR Score: ' + str(training_score) +
                       ' VL Score: ' + str(validation_score))

            for key in training_score.keys():
                k_fold_dict['folds'][k][f'TR_{key}'] = float(training_score[key])
                k_fold_dict['folds'][k][f'VL_{key}'] = float(validation_score[key])

            k_fold_dict['folds'][k]['TR_score'] = float(training_score['main_score'])
            k_fold_dict['folds'][k]['VL_score'] = float(validation_score['main_score'])

            # Inform the main process about experiment completion
            msg = dict(type="END_CONFIG", outer_fold=outer_fold_id, config_id=config_id, inner_fold=k)
            snd_queue.put(msg)

        for key in list(training_score.keys()) + ['score']:
            tr_scores = np.array([k_fold_dict['folds'][i][f'TR_{key}'] for i in range(self.inner_folds)])
            vl_scores = np.array([k_fold_dict['folds'][i][f'VL_{key}'] for i in range(self.inner_folds)])
            k_fold_dict[f'avg_TR_{key}'] = float(tr_scores.mean())
            k_fold_dict[f'std_TR_{key}'] = float(tr_scores.std())
            k_fold_dict[f'avg_VL_{key}'] = float(vl_scores.mean())
            k_fold_dict[f'std_VL_{key}'] = float(vl_scores.std())

        logger.log('TR avg is ' + str(k_fold_dict['avg_TR_score']) + ' std is ' + str(k_fold_dict['std_TR_score']) +
                   ' VL avg is ' + str(k_fold_dict['avg_VL_score']) + ' std is ' + str(k_fold_dict['std_VL_score']))

        with open(config_filename, 'w') as fp:
            json.dump(k_fold_dict, fp)

        # Now we can remove the last checkpoints to save space
        for k in range(self.inner_folds):
            ckpt_filename = os.path.join(exp_config_name, 'FOLD_' + str(k + 1), 'last_checkpoint.pth')
            os.remove(ckpt_filename)
