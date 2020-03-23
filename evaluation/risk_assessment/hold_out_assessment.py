import os
import json

from config.utils import s2c
from log.Logger import Logger


class HoldOutAssessment:
    """
    Class implementing a Hold-out technique to do Risk Assessment
    """

    def __init__(self, model_selector, exp_path, model_configs, final_training_runs=3):
        """
        Initialized a Hold-out procedure for Risk Assessment (estimate of the true generalization performances)
        :param model_selector: any model selection procedure defined in evaluation.model_selection
        :param exp_path: The folder in which to store all results
        :param model_configs: an object storing all possible model configurations, e.g. config.base.Grid
        """
        self.model_configs = model_configs  # Dictionary with key:list of possible values
        self.model_selector = model_selector
        self.final_training_runs = final_training_runs

        # Create the experiments folder straight away
        self.exp_path = exp_path
        self._HOLDOUT_FOLDER = os.path.join(exp_path, 'HOLDOUT_ASS')
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def risk_assessment(self, experiment_class, debug=False, other=None):
        """
        Starts a model selection.
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        """

        json_outer_results = os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME)
        if not os.path.exists(json_outer_results):
            self._risk_assessment_helper(experiment_class, self._HOLDOUT_FOLDER, debug, other)
        else:
            # Do not recompute experiments for this outer fold.
            print(f"File {json_outer_results} already present! Shutting down to prevent loss of previous experiments")

    def _risk_assessment_helper(self, experiment_class, exp_path, debug=False, other=None):
        """
        Helper method that runs model selection for the train/validation sets. Once model selection completes, 3 final
        training runs are performed, and test scores averaged to compute the test score.
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param exp_path: The folder in which to store all results
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        """
        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(self.model_configs.data_root, s2c(self.model_configs.dataset_class),
                                              self.model_configs.dataset_name, outer_folds=1)
        dataset_getter.set_outer_k(0)  # needs to stay 0

        best_config = self.model_selector.model_selection(dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, debug, other)

        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], exp_path)

        # Set up a log file for this experiment (I am in a forked process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        training_scores, test_scores = [], []

        # Mitigate bad random initializations
        for i in range(self.final_training_runs):
            training_score, test_score = experiment.run_test(dataset_getter, logger, other)
            print(f'Final training run {i + 1}: {training_score}, {test_score}')

            training_scores.append(training_score)
            test_scores.append(test_score)

        training_score = sum(training_scores) / self.final_training_runs
        test_score = sum(test_scores) / self.final_training_runs

        logger.log('TR score: ' + str(training_score) + ' TS score: ' + str(test_score))

        with open(os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'HOLDOUT_TR': training_score, 'HOLDOUT_TS': test_score}, fp)