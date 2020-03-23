import os
import json

import numpy as np
import concurrent.futures

from config.utils import s2c
from log import Logger


class KFoldAssessment:
    """ Class implementing a K-Fold technique to do Risk Assessment """

    def __init__(self, outer_folds, model_selector, exp_path, model_configs, outer_processes=2, final_training_runs=3):
        """
        Initializes a K-Fold procedure for Risk Assessment (estimate of the true generalization performances)
        :param outer_folds: The number K of outer TEST folds. You should have generated the splits accordingly
        :param model_selector: any model selection procedure defined in evaluation.model_selection
        :param exp_path: The folder in which to store all results
        :param model_configs: an object storing all possible model configurations, e.g. config.base.Grid
        :param outer_processes: The number of folds to process in parallel.
        """
        self.outer_folds = outer_folds
        self.outer_processes = outer_processes
        self.model_selector = model_selector
        self.model_configs = model_configs  # Dictionary with key:list of possible values
        self.final_training_runs = final_training_runs

        # Create the experiments folder straight away
        self.exp_path = exp_path
        self.__NESTED_FOLDER = os.path.join(exp_path, str(self.outer_folds) + '_NESTED_CV')
        self.__OUTER_FOLD_BASE = 'OUTER_FOLD_'
        self._OUTER_RESULTS_FILENAME = 'outer_results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self):
        """ Aggregates Outer Folds results and compute Training and Test mean/std """

        outer_TR_scores = []
        outer_TS_scores = []
        assessment_results = {}

        for i in range(1, self.outer_folds+1):
            try:
                config_filename = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(i),
                                               self._OUTER_RESULTS_FILENAME)

                with open(config_filename, 'r') as fp:
                    outer_fold_scores = json.load(fp)

                    outer_TR_scores.append(outer_fold_scores['OUTER_TR'])
                    outer_TS_scores.append(outer_fold_scores['OUTER_TS'])

            except Exception as e:
                print(e)

        outer_TR_scores = np.array(outer_TR_scores)
        outer_TS_scores = np.array(outer_TS_scores)

        assessment_results['avg_TR_score'] = outer_TR_scores.mean()
        assessment_results['std_TR_score'] = outer_TR_scores.std()
        assessment_results['avg_TS_score'] = outer_TS_scores.mean()
        assessment_results['std_TS_score'] = outer_TS_scores.std()

        with open(os.path.join(self.__NESTED_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp)

    def risk_assessment(self, experiment_class, debug=False, other=None):
        """
        Starts multiple processes, each of which will perform a model selection step. Then processes all results
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        """
        if not os.path.exists(self.__NESTED_FOLDER):
            os.makedirs(self.__NESTED_FOLDER)

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.outer_processes)
        for outer_k in range(self.outer_folds):

            # Create a separate folder for each experiment
            kfold_folder = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(outer_k + 1))
            if not os.path.exists(kfold_folder):
                os.makedirs(kfold_folder)

            json_outer_results = os.path.join(kfold_folder, self._OUTER_RESULTS_FILENAME)
            if not os.path.exists(json_outer_results):
                if not debug:
                    pool.submit(self._risk_assessment_helper, outer_k,
                                experiment_class, kfold_folder, debug, other)
                else:  # DEBUG
                    self._risk_assessment_helper(outer_k, experiment_class, kfold_folder, debug, other)
            else:
                # Do not recompute experiments for this outer fold.
                print(f"File {json_outer_results} already present! Shutting down to prevent loss of previous experiments")
                continue

        pool.shutdown()  # wait the batch of configs to terminate

        self.process_results()

    def _risk_assessment_helper(self, outer_k, experiment_class, exp_path, debug=False, other=None):
        """
        Helper method that runs model selection for a particular fold. Once model selection completes, 3 final
        training runs are performed, and test scores averaged to compute the test score of a single outer fold.
        :param outer_k: The specific outer fold for which to perform model selection
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param exp_path: The folder in which to store all results
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        """

        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(self.model_configs.data_root, s2c(self.model_configs.dataset_class),
                                              self.model_configs.dataset_name, outer_folds=self.outer_folds)
        dataset_getter.set_outer_k(outer_k)

        best_config = self.model_selector.model_selection(dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, debug, other)

        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], exp_path)

        # Set up a log file for this experiment (run in a separate process)

        logger = Logger.Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        training_scores, test_scores = [], []

        # Mitigate bad random initializations
        for i in range(self.final_training_runs):
            training_score, test_score = experiment.run_test(dataset_getter, logger, other)
            print(f'Final training run {i + 1}: {training_score}, {test_score}')

            training_scores.append(training_score)
            test_scores.append(test_score)

        training_score = sum(training_scores) / self.final_training_runs
        test_score = sum(test_scores) / self.final_training_runs

        logger.log('End of Outer fold. TR score: ' + str(training_score) + ' TS score: ' + str(test_score))

        with open(os.path.join(exp_path, self._OUTER_RESULTS_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'OUTER_TR': training_score, 'OUTER_TS': test_score}, fp)


