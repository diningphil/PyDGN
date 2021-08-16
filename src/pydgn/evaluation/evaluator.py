import json
import operator
import os
import os.path as osp
import random
import sys
import time
from copy import deepcopy

import numpy as np
import ray
import torch

from pydgn.evaluation.util import ProgressManager
from pydgn.experiment.util import s2c
from pydgn.log.Logger import Logger
from pydgn.static import *

# Ignore warnings
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


@ray.remote(num_cpus=1, num_gpus=int(os.environ.get(PYDGN_RAY_NUM_GPUS_PER_TASK)))
def run_valid(experiment_class, dataset_getter, config, config_id,
              fold_exp_folder, fold_results_torch_path, exp_seed, logger):
    if not osp.exists(fold_results_torch_path):
        start = time.time()
        experiment = experiment_class(config, fold_exp_folder, exp_seed)
        train_res, val_res = experiment.run_valid(dataset_getter, logger)
        elapsed = time.time() - start
        torch.save((train_res, val_res, elapsed), fold_results_torch_path)
    else:
        _, _, elapsed = torch.load(fold_results_torch_path)
    return dataset_getter.outer_k, dataset_getter.inner_k, config_id, elapsed


@ray.remote(num_cpus=1, num_gpus=int(os.environ.get(PYDGN_RAY_NUM_GPUS_PER_TASK)))
def run_test(experiment_class, dataset_getter, best_config, outer_k, i,
             final_run_exp_path, final_run_torch_path, exp_seed, logger):
    if not osp.exists(final_run_torch_path):
        start = time.time()
        experiment = experiment_class(best_config[CONFIG],
                                      final_run_exp_path, exp_seed)
        train_res, test_res = experiment.run_test(dataset_getter, logger)
        elapsed = time.time() - start
        torch.save((train_res, test_res, elapsed), final_run_torch_path)
    else:
        _, _, elapsed = torch.load(final_run_torch_path)
    return outer_k, i, elapsed


class RiskAssesser:
    """ Class implementing a K-Fold technique to do Risk Assessment and K-Fold Model Selection """

    def __init__(self, outer_folds, inner_folds, experiment_class, exp_path, splits_folder, splits_filepath,
                 model_configs,
                 final_training_runs, higher_is_better, gpus_per_task, base_seed=42):
        """
        Initializes a K-Fold procedure for Risk Assessment (estimate of the true generalization performances)
        :param outer_folds: The number K of outer TEST folds. You should have generated the splits accordingly
        :param outer_folds: The number K of inner VALIDATION folds. You should have generated the splits accordingly
        :param experiment_class: the experiment class to be instantiated
        :param exp_path: The folder in which to store all results
        :param splits_folder: The folder in which data splits are stored
        :param splits_filepath: The splits filepath with additional meta information
        :param model_configs: an object storing all possible model configurations, e.g. config.base.Grid
        :param final_training_runs: no of final training runs to mitigate bad initializations
        :param higher_is_better: How to find the best configuration during model selection
        :param gpus_per_task: Number of gpus to assign to each experiment
        :param base_seed: Seed used to generate experiments seeds. Used to replicate results
        """
        # REPRODUCIBILITY: https://pytorch.org/docs/stable/notes/randomness.html
        self.base_seed = base_seed
        # Impost the manual seed from the start
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)
        torch.cuda.manual_seed(self.base_seed)
        random.seed(self.base_seed)

        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.experiment_class = experiment_class

        # Iterator producing the list of all possible configs
        self.model_configs = model_configs
        self.final_training_runs = final_training_runs
        self.higher_is_better = higher_is_better
        if higher_is_better:
            self.operator = operator.gt
        else:
            self.operator = operator.lt
        self.gpus_per_task = gpus_per_task

        # Main folders
        self.exp_path = exp_path
        self.splits_folder = splits_folder

        # Splits filepath
        self.splits_filepath = splits_filepath

        # Model assessment filenames
        self._ASSESSMENT_FOLDER = osp.join(exp_path, MODEL_ASSESSMENT)
        self._OUTER_FOLD_BASE = 'OUTER_FOLD_'
        self._OUTER_RESULTS_FILENAME = 'outer_results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

        # Model selection filenames
        self._SELECTION_FOLDER = 'MODEL_SELECTION'
        self._INNER_FOLD_BASE = 'INNER_FOLD_'
        self._CONFIG_BASE = 'config_'
        self._CONFIG_RESULTS = 'config_results.json'
        self._WINNER_CONFIG = 'winner_config.json'

        # Used to keep track of the scheduled jobs
        self.outer_folds_job_list = []
        self.final_runs_job_list = []

    def risk_assessment(self, debug):
        """
        Performs model selection followed by risk assessment to evaluate the performances of a model.
        :param debug: If True, sequential execution is performed
        """
        if not osp.exists(self._ASSESSMENT_FOLDER):
            os.makedirs(self._ASSESSMENT_FOLDER)

        # Show progress
        with ProgressManager(self.outer_folds,
                             self.inner_folds,
                             len(self.model_configs),
                             self.final_training_runs,
                             show=not debug) as progress_manager:
            self.progress_manager = progress_manager

            # NOTE: Pre-computing seeds in advance simplifies the code
            # Pre-compute in advance the seeds for model selection to aid replicability
            self.model_selection_seeds = [[[random.randrange(2 ** 32 - 1) for _ in range(self.inner_folds)] for _ in
                                           range(len(self.model_configs))] for _ in range(self.outer_folds)]
            # Pre-compute in advance the seeds for the final runs to aid replicability
            self.final_runs_seeds = [[random.randrange(2 ** 32 - 1) for _ in range(self.final_training_runs)] for _ in
                                     range(self.outer_folds)]

            for outer_k in range(self.outer_folds):
                # Create a separate folder for each experiment
                kfold_folder = osp.join(self._ASSESSMENT_FOLDER, self._OUTER_FOLD_BASE + str(outer_k + 1))
                if not osp.exists(kfold_folder):
                    os.makedirs(kfold_folder)

                # Perform model selection. This determines a best config FOR EACH
                # of the k outer folds
                self.model_selection(kfold_folder, outer_k, debug)

                # Must stay separate from Ray distributed computing logic
                if debug:
                    self.run_final_model(outer_k, True)

            # We launched all model selection jobs, now it is time to wait
            if not debug:
                # This will also launch the final runs jobs once the model selection
                # for a specific outer folds is completed. It returns when
                # everything has completed
                self.wait_configs()

            # Produces the self._ASSESSMENT_FILENAME file
            self.process_outer_results()

    def wait_configs(self):
        no_model_configs = len(self.model_configs)
        skip_model_selection = no_model_configs == 1

        # Copy the list of jobs (only for model selection atm)
        waiting = [el for el in self.outer_folds_job_list]

        # Number of jobs to run for each OUTER fold
        n_inner_runs = self.inner_folds * no_model_configs

        # Counters to keep track of what has completed

        # keeps track of the model selection jobs completed for each outer fold
        outer_completed = [0 for _ in range(self.outer_folds)]
        # keeps track of the jobs completed for each inner fold
        inner_completed = [[0 for _ in range(no_model_configs)]
                           for _ in range(self.outer_folds)]

        # keeps track of the final run jobs completed for each outer fold
        final_runs_completed = [0 for _ in range(self.outer_folds)]

        if skip_model_selection:
            for outer_k in range(self.outer_folds):
                # no need to call process_inner_results() here
                self.run_final_model(outer_k, False)

                # Append the NEW jobs to the waiting list
                waiting.extend(self.final_runs_job_list[-self.final_training_runs:])

        while waiting:
            completed, waiting = ray.wait(waiting)

            for future in completed:
                is_model_selection_run = future in self.outer_folds_job_list

                if is_model_selection_run:  # Model selection
                    outer_k, inner_k, config_id, elapsed = ray.get(future)
                    self.progress_manager.update_state(dict(type=END_CONFIG,
                                                            outer_fold=outer_k,
                                                            inner_fold=inner_k,
                                                            config_id=config_id,
                                                            elapsed=elapsed))

                    ms_exp_path = osp.join(self._ASSESSMENT_FOLDER,
                                           self._OUTER_FOLD_BASE + str(outer_k + 1),
                                           self._SELECTION_FOLDER)
                    config_folder = osp.join(ms_exp_path,
                                             self._CONFIG_BASE + str(config_id + 1))

                    # if all inner folds completed, process that configuration
                    inner_completed[outer_k][config_id] += 1
                    if inner_completed[outer_k][config_id] == self.inner_folds:
                        self.process_config(config_folder, deepcopy(self.model_configs[config_id]))

                    # if model selection is complete, launch final runs
                    outer_completed[outer_k] += 1
                    if outer_completed[outer_k] == n_inner_runs:  # outer fold completed - schedule final runs
                        self.process_inner_results(ms_exp_path, len(self.model_configs))
                        self.run_final_model(outer_k, False)

                        # Append the NEW jobs to the waiting list
                        waiting.extend(self.final_runs_job_list[-self.final_training_runs:])

                elif future in self.final_runs_job_list:  # Risk ass. final runs
                    outer_k, run_id, elapsed = ray.get(future)
                    self.progress_manager.update_state(dict(type=END_FINAL_RUN,
                                                            outer_fold=outer_k,
                                                            run_id=run_id,
                                                            elapsed=elapsed))

                    final_runs_completed[outer_k] += 1
                    if final_runs_completed[outer_k] == self.final_training_runs:
                        # Time to produce self._OUTER_RESULTS_FILENAME
                        self.process_final_runs(outer_k)

    def model_selection(self, kfold_folder, outer_k, debug):
        """
        Performs model selection by launching each configuration in parallel, unless debug is True. Each process
        trains the same configuration for each inner fold.
        :param kfold_folder: The root folder for model selection
        :param outer_k: the current outer fold to consider
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        """
        model_selection_folder = osp.join(kfold_folder, self._SELECTION_FOLDER)

        # Create the dataset provider
        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(self.model_configs.data_root,
                                              self.splits_folder,
                                              self.splits_filepath,
                                              s2c(self.model_configs.dataset_class),
                                              self.model_configs.dataset_name,
                                              self.outer_folds,
                                              self.inner_folds,
                                              self.model_configs.num_dataloader_workers,
                                              self.model_configs.pin_memory)

        # Tell the data provider to take data relative
        # to a specific OUTER split
        dataset_getter.set_outer_k(outer_k)

        if not osp.exists(model_selection_folder):
            os.makedirs(model_selection_folder)

        # if the # of configs to try is 1, simply skip model selection
        if len(self.model_configs) > 1:

            # Launch one job for each inner_fold for each configuration
            for config_id, config in enumerate(self.model_configs):
                # I need to make a copy of this dictionary
                # It seems it gets shared between processes!
                cfg = deepcopy(config)

                # Create a separate folder for each configuration
                config_folder = osp.join(model_selection_folder,
                                         self._CONFIG_BASE + str(config_id + 1))
                if not osp.exists(config_folder):
                    os.makedirs(config_folder)

                for k in range(self.inner_folds):
                    # Create a separate folder for each fold for each config.
                    fold_exp_folder = osp.join(config_folder,
                                               self._INNER_FOLD_BASE + str(k + 1))
                    fold_results_torch_path = osp.join(fold_exp_folder,
                                                       f'fold_{str(k + 1)}_results.torch')

                    # Use pre-computed random seed for the experiment
                    exp_seed = self.model_selection_seeds[outer_k][config_id][k]
                    dataset_getter.set_exp_seed(exp_seed)

                    # Tell the data provider to take data relative
                    # to a specific INNER split
                    dataset_getter.set_inner_k(k)

                    logger = Logger(osp.join(fold_exp_folder,
                                             'experiment.log'), mode='a', debug=debug)
                    logger.log(json.dumps(dict(outer_k=dataset_getter.outer_k,
                                               inner_k=dataset_getter.inner_k,
                                               exp_seed=exp_seed,
                                               **config),
                                          sort_keys=False, indent=4))
                    if not debug:
                        # Launch the job and append to list of outer jobs
                        future = run_valid.remote(self.experiment_class,
                                                  dataset_getter, config,
                                                  config_id, fold_exp_folder,
                                                  fold_results_torch_path,
                                                  exp_seed, logger)
                        self.outer_folds_job_list.append(future)
                    else:  # debug mode
                        if not osp.exists(fold_results_torch_path):
                            start = time.time()
                            experiment = self.experiment_class(config, fold_exp_folder, exp_seed)
                            training_score, validation_score = experiment.run_valid(dataset_getter, logger)
                            elapsed = time.time() - start
                            torch.save((training_score, validation_score, elapsed),
                                       fold_results_torch_path)
                        else:
                            _, _, elapsed = torch.load(fold_results_torch_path)

                if debug:
                    self.process_config(config_folder, deepcopy(config))
            if debug:
                self.process_inner_results(model_selection_folder, len(self.model_configs))
        else:
            # Performing model selection for a single configuration is useless
            with open(osp.join(model_selection_folder, self._WINNER_CONFIG), 'w') as fp:
                json.dump(dict(best_config_id=0, config=self.model_configs[0]), fp, sort_keys=False, indent=4)

    def run_final_model(self, outer_k, debug):
        outer_folder = osp.join(self._ASSESSMENT_FOLDER,
                                self._OUTER_FOLD_BASE + str(outer_k + 1))
        config_fname = osp.join(outer_folder, self._SELECTION_FOLDER,
                                self._WINNER_CONFIG)

        with open(config_fname, 'r') as f:
            best_config = json.load(f)

        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(self.model_configs.data_root,
                                              self.splits_folder,
                                              self.splits_filepath,
                                              s2c(self.model_configs.dataset_class),
                                              self.model_configs.dataset_name,
                                              self.outer_folds,
                                              self.inner_folds,
                                              self.model_configs.num_dataloader_workers,
                                              self.model_configs.pin_memory)
        # Tell the data provider to take data relative
        # to a specific OUTER split
        dataset_getter.set_outer_k(outer_k)
        dataset_getter.set_inner_k(None)

        # Mitigate bad random initializations with more runs
        for i in range(self.final_training_runs):

            final_run_exp_path = osp.join(outer_folder, f"final_run{i + 1}")
            final_run_torch_path = osp.join(final_run_exp_path,
                                            f'run_{i + 1}_results.torch')

            # Use pre-computed random seed for the experiment
            exp_seed = self.final_runs_seeds[outer_k][i]
            dataset_getter.set_exp_seed(exp_seed)

            # Retrain with the best configuration and test
            # Set up a log file for this experiment (run in a separate process)
            logger = Logger(osp.join(final_run_exp_path,
                                     'experiment.log'), mode='a', debug=debug)
            logger.log(json.dumps(dict(outer_k=dataset_getter.outer_k,
                                       inner_k=dataset_getter.inner_k,
                                       exp_seed=exp_seed,
                                       **best_config),
                                  sort_keys=False, indent=4))

            if not debug:
                # Launch the job and append to list of final runs jobs
                future = run_test.remote(self.experiment_class, dataset_getter,
                                         best_config, outer_k, i,
                                         final_run_exp_path,
                                         final_run_torch_path, exp_seed, logger)
                self.final_runs_job_list.append(future)
            else:
                if not osp.exists(final_run_torch_path):
                    start = time.time()
                    experiment = self.experiment_class(best_config[CONFIG],
                                                       final_run_exp_path, exp_seed)
                    training_score, test_score = experiment.run_test(dataset_getter, logger)
                    elapsed = time.time() - start
                    torch.save((training_score, test_score, elapsed),
                               final_run_torch_path)
                else:
                    _, _, elapsed = torch.load(final_run_torch_path)
        if debug:
            self.process_final_runs(outer_k)

    def process_config(self, config_folder, config):
        config_filename = osp.join(config_folder, self._CONFIG_RESULTS)
        k_fold_dict = {
            CONFIG: config,
            FOLDS: [{} for _ in range(self.inner_folds)],
            AVG_TR_SCORE: 0.,
            AVG_VL_SCORE: 0.,
            STD_TR_SCORE: 0.,
            STD_VL_SCORE: 0.
        }

        for k in range(self.inner_folds):
            # Set up a log file for this experiment (run in a separate process)
            fold_exp_folder = osp.join(config_folder,
                                       self._INNER_FOLD_BASE + str(k + 1))
            fold_results_torch_path = osp.join(fold_exp_folder,
                                               f'fold_{str(k + 1)}_results.torch')

            training_score, validation_score, _ = torch.load(fold_results_torch_path)

            for key in training_score.keys():
                k_fold_dict[FOLDS][k][f'{TRAINING}_{key}'] = float(training_score[key])
                k_fold_dict[FOLDS][k][f'{VALIDATION}_{key}'] = float(validation_score[key])

            k_fold_dict[FOLDS][k][TR_SCORE] = float(training_score[MAIN_SCORE])
            k_fold_dict[FOLDS][k][VL_SCORE] = float(validation_score[MAIN_SCORE])

        for key in list(training_score.keys()) + [SCORE]:
            tr_scores = np.array([k_fold_dict[FOLDS][i][f'{TRAINING}_{key}']
                                  for i in range(self.inner_folds)])
            vl_scores = np.array([k_fold_dict[FOLDS][i][f'{VALIDATION}_{key}']
                                  for i in range(self.inner_folds)])
            k_fold_dict[f'{AVG}_{TRAINING}_{key}'] = float(tr_scores.mean())
            k_fold_dict[f'{STD}_{TRAINING}_{key}'] = float(tr_scores.std())
            k_fold_dict[f'{AVG}_{VALIDATION}_{key}'] = float(vl_scores.mean())
            k_fold_dict[f'{STD}_{VALIDATION}_{key}'] = float(vl_scores.std())

        with open(config_filename, 'w') as fp:
            json.dump(k_fold_dict, fp, sort_keys=False, indent=4)

    def process_inner_results(self, folder, no_configurations):
        """
        Chooses the best hyper-parameters configuration using the HIGHEST validation mean score
        :param folder: a folder which holds all configurations results after K folds
        :param no_configurations: number of possible configurations
        """
        best_avg_vl = -float('inf') if self.higher_is_better else float('inf')
        best_std_vl = float('inf')

        for i in range(1, no_configurations + 1):
            config_filename = osp.join(folder, self._CONFIG_BASE + str(i),
                                       self._CONFIG_RESULTS)

            with open(config_filename, 'r') as fp:
                config_dict = json.load(fp)

                avg_vl = config_dict[AVG_VL_SCORE]
                std_vl = config_dict[STD_VL_SCORE]

                if self.operator(avg_vl, best_avg_vl) or (best_avg_vl == avg_vl and best_std_vl > std_vl):
                    best_i = i
                    best_avg_vl = avg_vl
                    best_config = config_dict

        with open(osp.join(folder, self._WINNER_CONFIG), 'w') as fp:
            json.dump(dict(best_config_id=best_i, **best_config), fp, sort_keys=False, indent=4)

    def process_final_runs(self, outer_k):

        outer_folder = osp.join(self._ASSESSMENT_FOLDER,
                                self._OUTER_FOLD_BASE + str(outer_k + 1))
        config_fname = osp.join(outer_folder, self._SELECTION_FOLDER,
                                self._WINNER_CONFIG)

        with open(config_fname, 'r') as f:
            best_config = json.load(f)

            training_scores, test_scores = [], []
            for i in range(self.final_training_runs):

                final_run_exp_path = osp.join(outer_folder, f"final_run{i + 1}")
                final_run_torch_path = osp.join(final_run_exp_path,
                                                f'run_{i + 1}_results.torch')
                training_score, test_score, _ = torch.load(final_run_torch_path)
                training_scores.append(training_score)
                test_scores.append(test_score)

                tr_res = {}
                for k in training_score.keys():
                    tr_res[k] = np.mean([float(tr_run[k])
                                         for tr_run in training_scores])
                    tr_res[k + f'_{STD}'] = np.std([float(tr_run[k])
                                                    for tr_run in training_scores])

                te_res = {}
                for k in test_score.keys():
                    te_res[k] = np.mean([float(te_run[k])
                                         for te_run in test_scores])
                    te_res[k + f'_{STD}'] = np.std([float(te_run[k])
                                                    for te_run in test_scores])

        with open(osp.join(outer_folder, self._OUTER_RESULTS_FILENAME), 'w') as fp:
            json.dump({BEST_CONFIG: best_config,
                       OUTER_TRAIN: tr_res, OUTER_TEST: te_res},
                      fp, sort_keys=False, indent=4)

    def process_outer_results(self):
        """ Aggregates Outer Folds results and compute Training and Test mean/std """

        outer_tr_scores = []
        outer_ts_scores = []
        assessment_results = {}

        for i in range(1, self.outer_folds + 1):
            config_filename = osp.join(self._ASSESSMENT_FOLDER,
                                       self._OUTER_FOLD_BASE + str(i),
                                       self._OUTER_RESULTS_FILENAME)
            with open(config_filename, 'r') as fp:
                outer_fold_scores = json.load(fp)
                outer_tr_scores.append(outer_fold_scores[OUTER_TRAIN])
                outer_ts_scores.append(outer_fold_scores[OUTER_TEST])

                for k in outer_fold_scores[OUTER_TRAIN].keys():
                    # Do not want to average std of different final runs in different outer folds
                    if k[-3:] == STD:
                        continue
                    tr_scores = np.array([score[k] for score in outer_tr_scores])
                    ts_scores = np.array([score[k] for score in outer_ts_scores])
                    assessment_results[f'{AVG}_{TRAINING}_{k}'] = tr_scores.mean()
                    assessment_results[f'{STD}_{TRAINING}_{k}'] = tr_scores.std()
                    assessment_results[f'{AVG}_{TEST}_{k}'] = ts_scores.mean()
                    assessment_results[f'{STD}_{TEST}_{k}'] = ts_scores.std()

        with open(osp.join(self._ASSESSMENT_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp, sort_keys=False, indent=4)
