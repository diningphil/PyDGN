import json
import operator
import os
import os.path as osp
import random
import time
from copy import deepcopy
from typing import Tuple, Callable, Union

import numpy as np
import ray
import requests
import torch

from pydgn.data.provider import DataProvider
from pydgn.evaluation.config import Config
from pydgn.evaluation.grid import Grid
from pydgn.evaluation.random_search import RandomSearch
from pydgn.evaluation.util import ProgressManager
from pydgn.experiment.experiment import Experiment
from pydgn.experiment.util import s2c
from pydgn.log.logger import Logger
from pydgn.static import *


def send_telegram_update(bot_token: str, bot_chat_ID: str, bot_message: str):
    """
    Sends a message using Telegram APIs. Markdown can be used.

    Args:
        bot_token (str): token of the user's bot
        bot_chat_ID (str): identifier of the chat where to write the message
        bot_message (str): the message to be sent
    """
    send_text = (
        "https://api.telegram.org/bot"
        + str(bot_token)
        + "/sendMessage?chat_id="
        + str(bot_chat_ID)
        + "&parse_mode=Markdown&text="
        + str(bot_message)
    )
    response = requests.get(send_text)
    return response.json()


@ray.remote(
    num_cpus=1,
    num_gpus=float(os.environ.get(PYDGN_RAY_NUM_GPUS_PER_TASK, default=1)),
    max_calls=1
    # max_calls=1 --> the worker automatically exits after executing the task
    # (thereby releasing the GPU resources).
)
def run_valid(
    experiment_class: Callable[..., Experiment],
    dataset_getter: Callable[..., DataProvider],
    config: dict,
    config_id: int,
    fold_exp_folder: str,
    fold_results_torch_path: str,
    exp_seed: int,
    logger: Logger,
) -> Tuple[int, int, int, float]:
    r"""
    Ray job that performs a model selection run and returns bookkeeping
    information for the progress manager.

    Args:
        experiment_class
            (Callable[..., :class:`~pydgn.experiment.experiment.Experiment`]):
            the class of the experiment to instantiate
        dataset_getter
            (Callable[..., :class:`~pydgn.data.provider.DataProvider`]):
            the class of the data provider to instantiate
        config (dict): the configuration of this specific experiment
        config_id (int): the id of the configuration (for bookkeeping reasons)
        fold_exp_folder (str): path of the experiment root folder
        fold_results_torch_path (str): path where to store the
            results of the experiment
        exp_seed (int): seed of the experiment
        logger (:class:`~pydgn.log.logger.Logger`): a logger to log
            information in the appropriate file

    Returns:
        a tuple with outer fold id, inner fold id, config id, and time elapsed
    """
    if not osp.exists(fold_results_torch_path):
        start = time.time()
        experiment = experiment_class(config, fold_exp_folder, exp_seed)
        train_res, val_res = experiment.run_valid(dataset_getter, logger)
        elapsed = time.time() - start
        torch.save((train_res, val_res, elapsed), fold_results_torch_path)
    else:
        _, _, elapsed = torch.load(fold_results_torch_path)
    return dataset_getter.outer_k, dataset_getter.inner_k, config_id, elapsed


@ray.remote(
    num_cpus=1,
    num_gpus=float(os.environ.get(PYDGN_RAY_NUM_GPUS_PER_TASK, default=1)),
    max_calls=1
    # max_calls=1 --> the worker automatically exits after executing the task
    # (thereby releasing the GPU resources).
)
def run_test(
    experiment_class: Callable[..., Experiment],
    dataset_getter: Callable[..., DataProvider],
    best_config: dict,
    outer_k: int,
    i: int,
    final_run_exp_path: str,
    final_run_torch_path: str,
    exp_seed: int,
    logger: Logger,
) -> Tuple[int, int, float]:
    r"""
    Ray job that performs a risk assessment run and returns bookkeeping
    information for the progress manager.

    Args:
        experiment_class
            (Callable[..., :class:`~pydgn.experiment.experiment.Experiment`]):
            the class of the experiment to instantiate
        dataset_getter
            (Callable[..., :class:`~pydgn.data.provider.DataProvider`]):
            the class of the data provider to instantiate
        best_config (dict): the best configuration to use for this
            specific outer fold
        i (int): the id of the final run (for bookkeeping reasons)
        final_run_exp_path (str): path of the experiment root folder
        final_run_torch_path (str): path where to store the results
            of the experiment
        exp_seed (int): seed of the experiment
        logger (:class:`~pydgn.log.logger.Logger`): a logger to log
            information in the appropriate file

    Returns:
        a tuple with outer fold id, final run id, and time elapsed
    """
    if not osp.exists(final_run_torch_path):
        start = time.time()
        experiment = experiment_class(
            best_config[CONFIG], final_run_exp_path, exp_seed
        )
        res = experiment.run_test(dataset_getter, logger)
        elapsed = time.time() - start

        train_res, val_res, test_res = res
        torch.save(
            (train_res, val_res, test_res, elapsed), final_run_torch_path
        )
    else:
        res = torch.load(final_run_torch_path)
        elapsed = res[-1]
    return outer_k, i, elapsed


class RiskAssesser:
    r"""
    Class implementing a K-Fold technique to do Risk Assessment
    (estimate of the true generalization performances)
    and K-Fold Model Selection (select the best hyper-parameters
    for **each** external fold

    Args:
        outer_folds (int): The number K of outer TEST folds.
            You should have generated the splits accordingly
        outer_folds (int): The number K of inner VALIDATION folds.
            You should have generated the splits accordingly
        experiment_class
            (Callable[..., :class:`~pydgn.experiment.experiment.Experiment`]):
            the experiment class to be instantiated
        exp_path (str): The folder in which to store **all** results
        splits_filepath (str): The splits filepath with additional
            meta information
        model_configs
            (Union[:class:`~pydgn.evaluation.grid.Grid`,
            :class:`~pydgn.evaluation.random_search.RandomSearch`]):
            an object storing all possible model configurations,
            e.g., config.base.Grid
        final_training_runs (int): no of final training runs to mitigate
            bad initializations
        higher_is_better (bool): whether or not the best model
            for each external fold should be selected by higher
            or lower score values
        gpus_per_task (float): Number of gpus to assign to each
            experiment. Can be < ``1``.
        base_seed (int): Seed used to generate experiments seeds.
            Used to replicate results. Default is ``42``
    """

    def __init__(
        self,
        outer_folds: int,
        inner_folds: int,
        experiment_class: Callable[..., Experiment],
        exp_path: str,
        splits_filepath: str,
        model_configs: Union[Grid, RandomSearch],
        final_training_runs: int,
        higher_is_better: bool,
        gpus_per_task: float,
        base_seed: int = 42,
    ):

        # REPRODUCIBILITY:
        # https://pytorch.org/docs/stable/notes/randomness.html
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

        # Splits filepath
        self.splits_filepath = splits_filepath

        # Model assessment filenames
        self._ASSESSMENT_FOLDER = osp.join(exp_path, MODEL_ASSESSMENT)
        self._OUTER_FOLD_BASE = "OUTER_FOLD_"
        self._OUTER_RESULTS_FILENAME = "outer_results.json"
        self._ASSESSMENT_FILENAME = "assessment_results.json"

        # Model selection filenames
        self._SELECTION_FOLDER = "MODEL_SELECTION"
        self._INNER_FOLD_BASE = "INNER_FOLD_"
        self._CONFIG_BASE = "config_"
        self._CONFIG_RESULTS = "config_results.json"
        self._WINNER_CONFIG = "winner_config.json"

        # Used to keep track of the scheduled jobs
        self.outer_folds_job_list = []
        self.final_runs_job_list = []

        # telegram config
        tc = model_configs.telegram_config
        self.telegram_bot_token = (
            tc[TELEGRAM_BOT_TOKEN] if tc is not None else None
        )
        self.telegram_bot_chat_ID = (
            tc[TELEGRAM_BOT_CHAT_ID] if tc is not None else None
        )
        self.log_model_selection = (
            tc[LOG_MODEL_SELECTION] if tc is not None else None
        )
        self.log_final_runs = tc[LOG_FINAL_RUNS] if tc is not None else None

    def risk_assessment(self, debug: bool):
        r"""
        Performs risk assessment to evaluate the performances of a model.

        Args:
            debug: if ``True``, sequential execution is performed and logs are
                printed to screen
        """
        if not osp.exists(self._ASSESSMENT_FOLDER):
            os.makedirs(self._ASSESSMENT_FOLDER)

        # Show progress
        with ProgressManager(
            self.outer_folds,
            self.inner_folds,
            len(self.model_configs),
            self.final_training_runs,
            show=not debug,
        ) as progress_manager:
            self.progress_manager = progress_manager

            # NOTE: Pre-computing seeds in advance simplifies the code
            # Pre-compute in advance the seeds for model selection to aid
            # replicability
            self.model_selection_seeds = [
                [
                    [
                        random.randrange(2**32 - 1)
                        for _ in range(self.inner_folds)
                    ]
                    for _ in range(len(self.model_configs))
                ]
                for _ in range(self.outer_folds)
            ]
            # Pre-compute in advance the seeds for the final runs to aid
            # replicability
            self.final_runs_seeds = [
                [
                    random.randrange(2**32 - 1)
                    for _ in range(self.final_training_runs)
                ]
                for _ in range(self.outer_folds)
            ]

            for outer_k in range(self.outer_folds):
                # Create a separate folder for each experiment
                kfold_folder = osp.join(
                    self._ASSESSMENT_FOLDER,
                    self._OUTER_FOLD_BASE + str(outer_k + 1),
                )
                if not osp.exists(kfold_folder):
                    os.makedirs(kfold_folder)

                # Perform model selection. This determines a best config
                # FOR EACH of the k outer folds
                self.model_selection(kfold_folder, outer_k, debug)

                # Must stay separate from Ray distributed computing logic
                if debug:
                    self.run_final_model(outer_k, True)

            # We launched all model selection jobs, now it is time to wait
            if not debug:
                # This will also launch the final runs jobs once the model
                # selection for a specific outer folds is completed.
                # It returns when everything has completed
                self.wait_configs()

            # Produces the self._ASSESSMENT_FILENAME file
            self.process_outer_results()

    def wait_configs(self):
        r"""
        Waits for configurations to terminate and updates the state of the
        progress manager
        """
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
        inner_completed = [
            [0 for _ in range(no_model_configs)]
            for _ in range(self.outer_folds)
        ]

        # keeps track of the final run jobs completed for each outer fold
        final_runs_completed = [0 for _ in range(self.outer_folds)]

        if skip_model_selection:
            for outer_k in range(self.outer_folds):
                # no need to call process_inner_results() here
                self.run_final_model(outer_k, False)

                # Append the NEW jobs to the waiting list
                waiting.extend(
                    self.final_runs_job_list[-self.final_training_runs :]
                )

        while waiting:
            completed, waiting = ray.wait(waiting)

            for future in completed:
                is_model_selection_run = future in self.outer_folds_job_list

                if is_model_selection_run:  # Model selection
                    outer_k, inner_k, config_id, elapsed = ray.get(future)
                    self.progress_manager.update_state(
                        dict(
                            type=END_CONFIG,
                            outer_fold=outer_k,
                            inner_fold=inner_k,
                            config_id=config_id,
                            elapsed=elapsed,
                        )
                    )

                    ms_exp_path = osp.join(
                        self._ASSESSMENT_FOLDER,
                        self._OUTER_FOLD_BASE + str(outer_k + 1),
                        self._SELECTION_FOLDER,
                    )
                    config_folder = osp.join(
                        ms_exp_path, self._CONFIG_BASE + str(config_id + 1)
                    )

                    # if all inner folds completed, process that configuration
                    inner_completed[outer_k][config_id] += 1
                    if inner_completed[outer_k][config_id] == self.inner_folds:
                        self.process_config(
                            config_folder,
                            deepcopy(self.model_configs[config_id]),
                        )

                    # if model selection is complete, launch final runs
                    outer_completed[outer_k] += 1
                    if (
                        outer_completed[outer_k] == n_inner_runs
                    ):  # outer fold completed - schedule final runs
                        self.process_inner_results(
                            ms_exp_path, outer_k, len(self.model_configs)
                        )
                        self.run_final_model(outer_k, False)

                        # Append the NEW jobs to the waiting list
                        waiting.extend(
                            self.final_runs_job_list[
                                -self.final_training_runs :
                            ]
                        )

                elif (
                    future in self.final_runs_job_list
                ):  # Risk ass. final runs
                    outer_k, run_id, elapsed = ray.get(future)
                    self.progress_manager.update_state(
                        dict(
                            type=END_FINAL_RUN,
                            outer_fold=outer_k,
                            run_id=run_id,
                            elapsed=elapsed,
                        )
                    )

                    final_runs_completed[outer_k] += 1
                    if (
                        final_runs_completed[outer_k]
                        == self.final_training_runs
                    ):
                        # Time to produce self._OUTER_RESULTS_FILENAME
                        self.process_final_runs(outer_k)

    def model_selection(self, kfold_folder: str, outer_k: int, debug: bool):
        r"""
        Performs model selection.

        Args:
            kfold_folder: The root folder for model selection
            outer_k: the current outer fold to consider
            debug: if ``True``, sequential execution is performed and logs are
                printed to screen
        """
        model_selection_folder = osp.join(kfold_folder, self._SELECTION_FOLDER)

        # Create the dataset provider
        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(
            self.model_configs.data_root,
            self.splits_filepath,
            s2c(self.model_configs.dataset_class),
            self.model_configs.dataset_name,
            s2c(self.model_configs.data_loader_class),
            self.model_configs.data_loader_args,
            self.outer_folds,
            self.inner_folds,
        )

        # Tell the data provider to take data relative
        # to a specific OUTER split
        dataset_getter.set_outer_k(outer_k)

        if not osp.exists(model_selection_folder):
            os.makedirs(model_selection_folder)

        # if the # of configs to try is 1, simply skip model selection
        if len(self.model_configs) > 1:

            # Launch one job for each inner_fold for each configuration
            for config_id, config in enumerate(self.model_configs):
                # Create a separate folder for each configuration
                config_folder = osp.join(
                    model_selection_folder,
                    self._CONFIG_BASE + str(config_id + 1),
                )
                if not osp.exists(config_folder):
                    os.makedirs(config_folder)

                for k in range(self.inner_folds):
                    # Create a separate folder for each fold for each config.
                    fold_exp_folder = osp.join(
                        config_folder, self._INNER_FOLD_BASE + str(k + 1)
                    )
                    fold_results_torch_path = osp.join(
                        fold_exp_folder, f"fold_{str(k + 1)}_results.torch"
                    )

                    # Use pre-computed random seed for the experiment
                    exp_seed = self.model_selection_seeds[outer_k][config_id][
                        k
                    ]
                    dataset_getter.set_exp_seed(exp_seed)

                    # Tell the data provider to take data relative
                    # to a specific INNER split
                    dataset_getter.set_inner_k(k)

                    logger = Logger(
                        osp.join(fold_exp_folder, "experiment.log"),
                        mode="a",
                        debug=debug,
                    )
                    logger.log(
                        json.dumps(
                            dict(
                                outer_k=dataset_getter.outer_k,
                                inner_k=dataset_getter.inner_k,
                                exp_seed=exp_seed,
                                **config,
                            ),
                            sort_keys=False,
                            indent=4,
                        )
                    )
                    if not debug:
                        # Launch the job and append to list of outer jobs
                        future = run_valid.remote(
                            self.experiment_class,
                            dataset_getter,
                            config,
                            config_id,
                            fold_exp_folder,
                            fold_results_torch_path,
                            exp_seed,
                            logger,
                        )
                        self.outer_folds_job_list.append(future)
                    else:  # debug mode
                        if not osp.exists(fold_results_torch_path):
                            start = time.time()
                            experiment = self.experiment_class(
                                config, fold_exp_folder, exp_seed
                            )
                            (
                                training_score,
                                validation_score,
                            ) = experiment.run_valid(dataset_getter, logger)
                            elapsed = time.time() - start
                            torch.save(
                                (training_score, validation_score, elapsed),
                                fold_results_torch_path,
                            )
                        # else:
                        #     res = torch.load(fold_results_torch_path)
                        #     elapsed = res[-1]

                if debug:
                    self.process_config(config_folder, deepcopy(config))
            if debug:
                self.process_inner_results(
                    model_selection_folder, outer_k, len(self.model_configs)
                )
        else:
            # Performing model selection for a single configuration is useless
            with open(
                osp.join(model_selection_folder, self._WINNER_CONFIG), "w"
            ) as fp:
                json.dump(
                    dict(best_config_id=0, config=self.model_configs[0]),
                    fp,
                    sort_keys=False,
                    indent=4,
                )

    def run_final_model(self, outer_k: int, debug: bool):
        r"""
        Performs the final runs once the best model for outer fold ``outer_k``
        has been chosen.

        Args:
            outer_k (int): the current outer fold to consider
            debug (bool): if ``True``, sequential execution is performed and
                logs are printed to screen
        """
        outer_folder = osp.join(
            self._ASSESSMENT_FOLDER, self._OUTER_FOLD_BASE + str(outer_k + 1)
        )
        config_fname = osp.join(
            outer_folder, self._SELECTION_FOLDER, self._WINNER_CONFIG
        )

        with open(config_fname, "r") as f:
            best_config = json.load(f)

        dataset_getter_class = s2c(self.model_configs.dataset_getter)
        dataset_getter = dataset_getter_class(
            self.model_configs.data_root,
            self.splits_filepath,
            s2c(self.model_configs.dataset_class),
            self.model_configs.dataset_name,
            s2c(self.model_configs.data_loader_class),
            self.model_configs.data_loader_args,
            self.outer_folds,
            self.inner_folds,
        )
        # Tell the data provider to take data relative
        # to a specific OUTER split
        dataset_getter.set_outer_k(outer_k)
        dataset_getter.set_inner_k(None)

        # Mitigate bad random initializations with more runs
        for i in range(self.final_training_runs):

            final_run_exp_path = osp.join(outer_folder, f"final_run{i + 1}")
            final_run_torch_path = osp.join(
                final_run_exp_path, f"run_{i + 1}_results.torch"
            )

            # Use pre-computed random seed for the experiment
            exp_seed = self.final_runs_seeds[outer_k][i]
            dataset_getter.set_exp_seed(exp_seed)

            # Retrain with the best configuration and test
            # Set up a log file for this experiment (run in a separate process)
            logger = Logger(
                osp.join(final_run_exp_path, "experiment.log"),
                mode="a",
                debug=debug,
            )
            logger.log(
                json.dumps(
                    dict(
                        outer_k=dataset_getter.outer_k,
                        inner_k=dataset_getter.inner_k,
                        exp_seed=exp_seed,
                        **best_config,
                    ),
                    sort_keys=False,
                    indent=4,
                )
            )

            if not debug:
                # Launch the job and append to list of final runs jobs
                future = run_test.remote(
                    self.experiment_class,
                    dataset_getter,
                    best_config,
                    outer_k,
                    i,
                    final_run_exp_path,
                    final_run_torch_path,
                    exp_seed,
                    logger,
                )
                self.final_runs_job_list.append(future)
            else:
                if not osp.exists(final_run_torch_path):
                    start = time.time()
                    experiment = self.experiment_class(
                        best_config[CONFIG], final_run_exp_path, exp_seed
                    )
                    res = experiment.run_test(dataset_getter, logger)
                    elapsed = time.time() - start

                    training_res, val_res, test_res = res
                    torch.save(
                        (training_res, val_res, test_res, elapsed),
                        final_run_torch_path,
                    )
                # else:
                #     res = torch.load(final_run_torch_path)
                #     elapsed = res[-1]
        if debug:
            self.process_final_runs(outer_k)

    def process_config(self, config_folder: str, config: Config):
        r"""
        Computes the best configuration for each external fold and stores
        it into a file.

        Args:
            config_folder (str):
            config (:class:`~pydgn.evaluation.config.Config`): the
                configuration object
        """
        config_filename = osp.join(config_folder, self._CONFIG_RESULTS)
        k_fold_dict = {
            CONFIG: config,
            FOLDS: [{} for _ in range(self.inner_folds)],
        }

        assert not self.inner_folds <= 0
        for k in range(self.inner_folds):
            # Set up a log file for this experiment (run in a separate process)
            fold_exp_folder = osp.join(
                config_folder, self._INNER_FOLD_BASE + str(k + 1)
            )
            fold_results_torch_path = osp.join(
                fold_exp_folder, f"fold_{str(k + 1)}_results.torch"
            )

            training_res, validation_res, _ = torch.load(
                fold_results_torch_path
            )

            training_loss, validation_loss = (
                training_res[LOSS],
                validation_res[LOSS],
            )
            training_score, validation_score = (
                training_res[SCORE],
                validation_res[SCORE],
            )

            for res_type, mode, res, main_res_type in [
                (LOSS, TRAINING, training_loss, MAIN_LOSS),
                (LOSS, VALIDATION, validation_loss, MAIN_LOSS),
                (SCORE, TRAINING, training_score, MAIN_SCORE),
                (SCORE, VALIDATION, validation_score, MAIN_SCORE),
            ]:
                for key in res.keys():
                    if main_res_type in key:
                        continue
                    k_fold_dict[FOLDS][k][f"{mode}_{key}_{res_type}"] = float(
                        res[key]
                    )

            # Rename main loss key for aesthetic
            k_fold_dict[FOLDS][k][TR_LOSS] = float(training_loss[MAIN_LOSS])
            k_fold_dict[FOLDS][k][VL_LOSS] = float(validation_loss[MAIN_LOSS])
            k_fold_dict[FOLDS][k][TR_SCORE] = float(training_score[MAIN_SCORE])
            k_fold_dict[FOLDS][k][VL_SCORE] = float(
                validation_score[MAIN_SCORE]
            )
            del training_loss[MAIN_LOSS]
            del validation_loss[MAIN_LOSS]
            del training_score[MAIN_SCORE]
            del validation_score[MAIN_SCORE]

        # Note that training/validation loss/score will be used only to extract
        # the proper keys
        for key_dict, set_type, res_type in [
            (training_loss, TRAINING, LOSS),
            (validation_loss, VALIDATION, LOSS),
            (training_score, TRAINING, SCORE),
            (validation_score, VALIDATION, SCORE),
        ]:
            for key in list(key_dict.keys()) + [res_type]:
                suffix = f"_{res_type}" if key != res_type else ""
                set_res = np.array(
                    [
                        k_fold_dict[FOLDS][i][f"{set_type}_{key}{suffix}"]
                        for i in range(self.inner_folds)
                    ]
                )
                k_fold_dict[f"{AVG}_{set_type}_{key}{suffix}"] = float(
                    set_res.mean()
                )
                k_fold_dict[f"{STD}_{set_type}_{key}{suffix}"] = float(
                    set_res.std()
                )

        with open(config_filename, "w") as fp:
            json.dump(k_fold_dict, fp, sort_keys=False, indent=4)

    def process_inner_results(
        self, folder: str, outer_k: int, no_configurations: int
    ):
        r"""
        Chooses the best hyper-parameters configuration using the HIGHEST
        validation mean score.

        Args:
            folder (str): a folder which holds all configurations results
                after K INNER folds
            outer_k (int): the current outer fold to consider
            no_configurations (int): number of possible configurations
        """
        best_avg_vl = -float("inf") if self.higher_is_better else float("inf")
        best_std_vl = float("inf")

        for i in range(1, no_configurations + 1):
            config_filename = osp.join(
                folder, self._CONFIG_BASE + str(i), self._CONFIG_RESULTS
            )

            with open(config_filename, "r") as fp:
                config_dict = json.load(fp)

                avg_vl = config_dict[f"{AVG}_{VALIDATION}_{SCORE}"]
                std_vl = config_dict[f"{STD}_{VALIDATION}_{SCORE}"]

                if self.operator(avg_vl, best_avg_vl) or (
                    best_avg_vl == avg_vl and best_std_vl > std_vl
                ):
                    best_i = i
                    best_avg_vl = avg_vl
                    best_std_vl = std_vl
                    best_config = config_dict

        # Send telegram update
        if (
            self.model_configs.telegram_config is not None
            and self.log_model_selection
        ):
            exp_name = os.path.basename(self.exp_path)
            telegram_msg = (
                f"Exp *{exp_name}* \n"
                f"Model Sel. ended for outer fold *{outer_k + 1}* \n"
                f"Best config id: *{best_i}* \n"
                f"Main score: avg *{best_avg_vl:.4f}* "
                f"/ std *{best_std_vl:.4f}*"
            )
            send_telegram_update(
                self.telegram_bot_token,
                self.telegram_bot_chat_ID,
                telegram_msg,
            )

        with open(osp.join(folder, self._WINNER_CONFIG), "w") as fp:
            json.dump(
                dict(best_config_id=best_i, **best_config),
                fp,
                sort_keys=False,
                indent=4,
            )

    def process_final_runs(self, outer_k: int):
        r"""
        Computes the average scores for the final runs of a specific outer fold

        Args:
            outer_k (int): id of the outer fold from 0 to K-1
        """
        outer_folder = osp.join(
            self._ASSESSMENT_FOLDER, self._OUTER_FOLD_BASE + str(outer_k + 1)
        )
        config_fname = osp.join(
            outer_folder, self._SELECTION_FOLDER, self._WINNER_CONFIG
        )

        with open(config_fname, "r") as f:
            best_config = json.load(f)

            training_losses, validation_losses, test_losses = [], [], []
            training_scores, validation_scores, test_scores = [], [], []
            for i in range(self.final_training_runs):

                final_run_exp_path = osp.join(
                    outer_folder, f"final_run{i + 1}"
                )
                final_run_torch_path = osp.join(
                    final_run_exp_path, f"run_{i + 1}_results.torch"
                )
                res = torch.load(final_run_torch_path)

                tr_res, vl_res, te_res = {}, {}, {}

                training_res, validation_res, test_res, _ = res
                training_loss, validation_loss, test_loss = (
                    training_res[LOSS],
                    validation_res[LOSS],
                    test_res[LOSS],
                )
                training_score, validation_score, test_score = (
                    training_res[SCORE],
                    validation_res[SCORE],
                    test_res[SCORE],
                )

                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                test_losses.append(test_loss)
                training_scores.append(training_score)
                validation_scores.append(validation_score)
                test_scores.append(test_score)

                # this block may be unindented, *_score used only to retrieve
                # keys
                scores = [
                    (training_score, tr_res, training_scores),
                    (validation_score, vl_res, validation_scores),
                    (test_score, te_res, test_scores),
                ]
                losses = [
                    (training_loss, tr_res, training_losses),
                    (validation_loss, vl_res, validation_losses),
                    (test_loss, te_res, test_losses),
                ]

                # this block may be unindented, set_score used only to retrieve
                # keys
                for res_type, res in [(LOSS, losses), (SCORE, scores)]:
                    for set_res_type, set_dict, set_results in res:
                        for key in set_res_type.keys():
                            suffix = (
                                f"_{res_type}"
                                if (key != MAIN_LOSS and key != MAIN_SCORE)
                                else ""
                            )
                            set_dict[key + suffix] = np.mean(
                                [
                                    float(set_run[key])
                                    for set_run in set_results
                                ]
                            )
                            set_dict[key + f"{suffix}_{STD}"] = np.std(
                                [
                                    float(set_run[key])
                                    for set_run in set_results
                                ]
                            )

        # Send telegram update
        if (
            self.model_configs.telegram_config is not None
            and self.log_final_runs
        ):
            exp_name = os.path.basename(self.exp_path)
            telegram_msg = (
                f"Exp *{exp_name}* \n"
                f"Final runs ended for outer fold *{outer_k + 1}* \n"
                f"Main test score: avg *{scores[2][1][MAIN_SCORE]:.4f}* "
                f'/ std *{scores[2][1][f"{MAIN_SCORE}_{STD}"]:.4f}*'
            )
            send_telegram_update(
                self.telegram_bot_token,
                self.telegram_bot_chat_ID,
                telegram_msg,
            )

        with open(
            osp.join(outer_folder, self._OUTER_RESULTS_FILENAME), "w"
        ) as fp:

            json.dump(
                {
                    BEST_CONFIG: best_config,
                    OUTER_TRAIN: tr_res,
                    OUTER_VALIDATION: vl_res,
                    OUTER_TEST: te_res,
                },
                fp,
                sort_keys=False,
                indent=4,
            )

    def process_outer_results(self):
        r"""
        Aggregates Outer Folds results and compute Training and Test mean/std
        """
        outer_tr_results = []
        outer_vl_results = []
        outer_ts_results = []
        assessment_results = {}

        for i in range(1, self.outer_folds + 1):
            config_filename = osp.join(
                self._ASSESSMENT_FOLDER,
                self._OUTER_FOLD_BASE + str(i),
                self._OUTER_RESULTS_FILENAME,
            )

            with open(config_filename, "r") as fp:
                outer_fold_results = json.load(fp)
                outer_tr_results.append(outer_fold_results[OUTER_TRAIN])
                outer_vl_results.append(outer_fold_results[OUTER_VALIDATION])
                outer_ts_results.append(outer_fold_results[OUTER_TEST])

                for k in outer_fold_results[
                    OUTER_TRAIN
                ].keys():  # train keys are the same as valid and test keys
                    # Do not want to average std of different final runs in
                    # different outer folds
                    if k[-3:] == STD:
                        continue

                    # there may be different optimal losses for each outer
                    # fold, so we cannot always compute the average over
                    # K outer folds of the same loss this is not so
                    # problematic as one can always recover the average and
                    # standard loss values across outer folds when we
                    # have the same loss for all outer folds using
                    # a jupyter notebook
                    if "_loss" in k:
                        continue

                    outer_results = [
                        (outer_tr_results, TRAINING),
                        (outer_vl_results, VALIDATION),
                        (outer_ts_results, TEST),
                    ]

                    for results, set in outer_results:
                        set_results = np.array([res[k] for res in results])
                        assessment_results[
                            f"{AVG}_{set}_{k}"
                        ] = set_results.mean()
                        assessment_results[
                            f"{STD}_{set}_{k}"
                        ] = set_results.std()

        # Send telegram update
        if self.model_configs.telegram_config is not None:
            exp_name = os.path.basename(self.exp_path)
            telegram_msg = (
                f"Exp *{exp_name}* \n"
                f"Experiment has finished \n"
                f"Test score: avg "
                f'*{assessment_results[f"{AVG}_{TEST}_{MAIN_SCORE}"]:.4f}* '
                f"/ std"
                f' *{assessment_results[f"{STD}_{TEST}_{MAIN_SCORE}"]:.4f}*'
            )
            send_telegram_update(
                self.telegram_bot_token,
                self.telegram_bot_chat_ID,
                telegram_msg,
            )

        with open(
            osp.join(self._ASSESSMENT_FOLDER, self._ASSESSMENT_FILENAME), "w"
        ) as fp:
            json.dump(assessment_results, fp, sort_keys=False, indent=4)
