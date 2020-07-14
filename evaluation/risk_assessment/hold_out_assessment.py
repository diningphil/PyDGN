import os
import json
from pathlib import Path

import queue  # needed to catch empty exception
import torch.multiprocessing as mp
import concurrent.futures

from config.utils import s2c
from log.Logger import Logger
from evaluation.utils import ProgressManager


class HoldOutAssessment:
    """
    Class implementing a Hold-out technique to do Risk Assessment
    """

    def __init__(self, model_selector, exp_path, splits_folder, model_configs, final_training_runs=3):
        """
        Initialized a Hold-out procedure for Risk Assessment (estimate of the true generalization performances)
        :param model_selector: any model selection procedure defined in evaluation.model_selection
        :param exp_path: The folder in which to store all results
        :param model_configs: an object storing all possible model configurations, e.g. config.base.Grid
        """
        self.outer_folds = 1
        self.model_configs = model_configs  # Dictionary with key:list of possible values
        self.model_selector = model_selector
        self.final_training_runs = final_training_runs

        # Create the experiments folder straight away
        self.exp_path = exp_path
        self.splits_folder = Path(splits_folder)
        self._HOLDOUT_FOLDER = os.path.join(exp_path, 'HOLDOUT_ASS')
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def risk_assessment(self, experiment_class, debug=False, other=None):
        """
        Starts a model selection.
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        """
        if not os.path.exists(self._HOLDOUT_FOLDER):
            os.makedirs(self._HOLDOUT_FOLDER)
        json_outer_results = os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME)

        model_selections_done = 0
        def done_callback(future=None):
            nonlocal model_selections_done
            model_selections_done += 1
            # if future is not None:
            #    print(future.result())

        # Show progress
        with ProgressManager(self.outer_folds, self.model_selector.inner_folds, len(self.model_configs), self.final_training_runs) as progress:
            def read_all_msgs(q, timeout=1):
                try:
                    while True: # Read all messages
                        msg = q.get(timeout=timeout)
                        progress.update_state(msg)
                except queue.Empty:
                    pass
            '''
            Spawning rather than forking here prevents Pytorch from complaining
            See https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
            and https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
            It triggers a warning on a leaked semaphore which we will ignore for now.
            Also, https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note
            '''
            mp_context = mp.get_context('spawn')
            m = mp.Manager()
            rcv_queue = m.Queue()
            with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=mp_context) as pool:
                #if not os.path.exists(json_outer_results):
                if not debug:
                    f = pool.submit(self._risk_assessment_helper,
                                experiment_class, self._HOLDOUT_FOLDER, debug, other, rcv_queue)
                    f.add_done_callback(done_callback)
                else:  # DEBUG
                    self._risk_assessment_helper(experiment_class, self._HOLDOUT_FOLDER, debug, other, rcv_queue)
                    done_callback()
                    read_all_msgs(rcv_queue)
                #else:
                #    # Do not recompute experiments for this outer fold.
                #    print(f"File {json_outer_results} already present! Shutting down to prevent loss of previous experiments")
                #    done_callback()

                # Passive wait with timeout
                while model_selections_done < 1:
                    read_all_msgs(rcv_queue)
                # Deal with corner case where all configs terminate quickly
                read_all_msgs(rcv_queue)

    def _risk_assessment_helper(self, experiment_class, exp_path, debug, other, snd_queue):
        """
        Helper method that runs model selection for the train/validation sets. Once model selection completes, 3 final
        training runs are performed, and test scores averaged to compute the test score.
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param exp_path: The folder in which to store all results
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param other: this can be used to pass some additional information to the experiment in the form of a dict
        """
        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(self.model_configs.data_root, self.splits_folder, s2c(self.model_configs.dataset_class),
                                              self.model_configs.dataset_name, outer_folds=self.outer_folds, inner_folds=self.model_selector.inner_folds,
                                              num_workers=self.model_configs.num_dataloader_workers, pin_memory=self.model_configs.pin_memory)
        dataset_getter.set_outer_k(0)  # needs to stay 0

        best_config = self.model_selector.model_selection(0, dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, debug, other, snd_queue)

        if not os.path.exists(os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME)):
            # Retrain with the best configuration and test
            experiment = experiment_class(best_config['config'], exp_path)

            # Set up a log file for this experiment (I am in a forked process)
            logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

            training_scores, test_scores = [], []

            # Mitigate bad random initializations
            for i in range(self.final_training_runs):
                msg = dict(type="START_FINAL_RUN", outer_fold=0, run_id=i)
                snd_queue.put(msg)

                training_score, test_score = experiment.run_test(dataset_getter, logger, other)
                logger.log(f'Final training run {i + 1}: {training_score}, {test_score}')

                training_scores.append(training_score)
                test_scores.append(test_score)

                msg = dict(type="END_FINAL_RUN", outer_fold=0, run_id=i)
                snd_queue.put(msg)

            tr_res = {}
            for k in training_score.keys():
                tr_res[k] = sum([float(tr_run[k]) for tr_run in training_scores]) / self.final_training_runs

            te_res = {}
            for k in test_score.keys():
                te_res[k] = sum([float(te_run[k]) for te_run in test_scores]) / self.final_training_runs

            logger.log('TR score: ' + str(tr_res) + ' TS score: ' + str(te_res))

            with open(os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
                json.dump({'best_config': best_config, 'HOLDOUT_TR': tr_res, 'HOLDOUT_TS': te_res}, fp)
        else:
            for msg_type in ["START_FINAL_RUN", "END_FINAL_RUN"]:
                for run_id in range(self.final_training_runs):
                    msg = dict(type=msg_type, outer_fold=0, run_id=run_id)
                    snd_queue.put(msg)
