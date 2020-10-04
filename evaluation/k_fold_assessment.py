import os
import json
from pathlib import Path

import queue  # needed to catch empty exception
import torch.multiprocessing as mp
import concurrent.futures

import torch
import numpy as np
from experiments.experiment import s2c
from log.Logger import Logger
from evaluation.utils import ProgressManager


class KFoldAssessment:
    """ Class implementing a K-Fold technique to do Risk Assessment """

    def __init__(self, outer_folds, model_selector, exp_path, splits_folder, model_configs, outer_processes, final_training_runs):
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
        self.splits_folder = Path(splits_folder)
        #self.__NESTED_FOLDER = os.path.join(exp_path, str(self.outer_folds) + '_NESTED_CV')
        self._ASSESSMENT_FOLDER = os.path.join(exp_path, 'MODEL_ASSESSMENT')
        self._OUTER_FOLD_BASE = 'OUTER_FOLD_'
        self._OUTER_RESULTS_FILENAME = 'outer_results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self):
        """ Aggregates Outer Folds results and compute Training and Test mean/std """

        outer_TR_scores = []
        outer_TS_scores = []
        assessment_results = {}

        for i in range(1, self.outer_folds+1):
            try:
                config_filename = os.path.join(self._ASSESSMENT_FOLDER, self._OUTER_FOLD_BASE + str(i),
                                               self._OUTER_RESULTS_FILENAME)

                with open(config_filename, 'r') as fp:
                    outer_fold_scores = json.load(fp)

                    outer_TR_scores.append(outer_fold_scores['OUTER_TR'])
                    outer_TS_scores.append(outer_fold_scores['OUTER_TS'])

            except Exception as e:
                print(e)

        for k in outer_fold_scores['OUTER_TR'].keys():
            tr_scores = np.array([score[k] for score in outer_TR_scores])
            ts_scores = np.array([score[k] for score in outer_TS_scores])
            assessment_results[f'avg_TR_{k}'] = tr_scores.mean()
            assessment_results[f'std_TR_{k}'] = tr_scores.std()
            assessment_results[f'avg_TS_{k}'] = ts_scores.mean()
            assessment_results[f'std_TS_{k}'] = ts_scores.std()

        with open(os.path.join(self._ASSESSMENT_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp)

    def risk_assessment(self, experiment_class, use_cuda, debug=False):
        """
        Starts multiple processes, each of which will perform a model selection step. Then processes all results
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param use_cuda: whether cuda is to be used in the experiments (no multiprocessing)
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        """
        if not os.path.exists(self._ASSESSMENT_FOLDER):
            os.makedirs(self._ASSESSMENT_FOLDER)

        model_selections_done = 0
        def done_callback(future=None):
            nonlocal model_selections_done
            model_selections_done += 1
            # if future is not None:
            #    print(future.result())

        # Show progress
        with ProgressManager(self.outer_folds, self.model_selector.inner_folds, len(self.model_configs), self.final_training_runs, show=not debug) as progress:

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
            rcv_queue = m.Queue(maxsize=1000)
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.outer_processes if not use_cuda else 1, mp_context=mp_context) as pool:
                for outer_k in range(self.outer_folds):

                    # Create a separate folder for each experiment
                    kfold_folder = os.path.join(self._ASSESSMENT_FOLDER, self._OUTER_FOLD_BASE + str(outer_k + 1))
                    if not os.path.exists(kfold_folder):
                        os.makedirs(kfold_folder)

                    if not debug:
                        f = pool.submit(self._risk_assessment_helper, outer_k,
                                    experiment_class, kfold_folder, use_cuda, debug, rcv_queue)
                        f.add_done_callback(done_callback)
                    else:
                        self._risk_assessment_helper(outer_k, experiment_class, kfold_folder, use_cuda, debug, rcv_queue)
                        done_callback()
                        read_all_msgs(rcv_queue)

                if not debug:
                    # Passive wait with timeout
                    while model_selections_done < self.outer_folds:
                        read_all_msgs(rcv_queue)
                    # Deal with corner case where all configs terminate quickly
                    read_all_msgs(rcv_queue)

        self.process_results()

    def _risk_assessment_helper(self, outer_k, experiment_class, exp_path, use_cuda, debug, snd_queue):
        """
        Helper method that runs model selection for a particular fold. Once model selection completes, 3 final
        training runs are performed, and test scores averaged to compute the test score of a single outer fold.
        :param outer_k: The specific outer fold for which to perform model selection
        :param experiment_class: Class of the experiment to be run, see experiments.Experiment and its subclasses
        :param exp_path: The folder in which to store all results
        :param use_cuda: whether cuda is to be used in the experiments (no multiprocessing)
        :param debug: whether to run the procedure in debug mode (no multiprocessing)
        :param snd_queue: Queue object used only to SEND msg to parent process
        """
        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(self.model_configs.data_root, self.splits_folder, s2c(self.model_configs.dataset_class),
                                              self.model_configs.dataset_name, outer_folds=self.outer_folds, inner_folds=self.model_selector.inner_folds,
                                               num_workers=self.model_configs.num_dataloader_workers, pin_memory=self.model_configs.pin_memory)
        dataset_getter.set_outer_k(outer_k)

        best_config = self.model_selector.model_selection(outer_k, dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, use_cuda, debug, snd_queue)

        if not os.path.exists(os.path.join(exp_path, self._OUTER_RESULTS_FILENAME)):

            training_scores, test_scores = [], []

            # Mitigate bad random initializations
            for i in range(self.final_training_runs):
                final_run_exp_path = Path(exp_path / Path(f"final_run{i+1}"))
                final_run_torch_path = final_run_exp_path / Path(f'run_{i+1}_results.torch')

                msg = dict(type="START_FINAL_RUN", outer_fold=outer_k, run_id=i)
                snd_queue.put(msg)

                if not os.path.exists(final_run_torch_path):
                    # Retrain with the best configuration and test
                    experiment = experiment_class(best_config['config'], final_run_exp_path)
                    # Set up a log file for this experiment (run in a separate process)
                    logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
                    logger.log('Configuration: \n' + str(best_config['config']))

                    training_score, test_score = experiment.run_test(dataset_getter, logger)
                    logger.log(f'Final training run {i + 1}: {training_score}, {test_score}')

                    torch.save((training_score, test_score), final_run_torch_path)
                else:
                    training_score, test_score = torch.load(final_run_torch_path)

                training_scores.append(training_score)
                test_scores.append(test_score)

                msg = dict(type="END_FINAL_RUN", outer_fold=outer_k, run_id=i)
                snd_queue.put(msg)

            tr_res = {}
            for k in training_score.keys():
                tr_res[k] = np.mean([float(tr_run[k]) for tr_run in training_scores])
                tr_res[k+'_std'] = np.std([float(tr_run[k]) for tr_run in training_scores])

            te_res = {}
            for k in test_score.keys():
                te_res[k] = np.mean([float(te_run[k]) for te_run in test_scores])
                te_res[k+'_std'] = np.std([float(te_run[k]) for te_run in test_scores])

            logger.log('End of Outer fold. TR score: ' + str(tr_res) + ' TS score: ' + str(te_res))

            with open(os.path.join(exp_path, self._OUTER_RESULTS_FILENAME), 'w') as fp:
                json.dump({'best_config': best_config, 'OUTER_TR': tr_res, 'OUTER_TS': te_res}, fp)
        else:
            for msg_type in ["START_FINAL_RUN", "END_FINAL_RUN"]:
                for run_id in range(self.final_training_runs):
                    msg = dict(type=msg_type, outer_fold=outer_k, run_id=run_id)
                    snd_queue.put(msg)
