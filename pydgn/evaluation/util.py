import datetime
import json
import math
import os
import random
from typing import Tuple, Callable, List

import torch
import tqdm
from pydgn.data.provider import DataProvider

from pydgn.data.dataset import DatasetInterface
from pydgn.experiment.util import s2c
from pydgn.model.interface import ModelInterface
from pydgn.static import *


def return_class_and_args(
    config: dict, key: str, return_class_name: bool = False
) -> Tuple[Callable[..., object], dict]:
    r"""
    Returns the class and arguments associated to a specific key in the
    configuration file.

    Args:
        config (dict): the configuration dictionary
        key (str): a string representing a particular class in the
            configuration dictionary
        return_class_name (bool): if ``True``, returns the class name as a
            string rather than the class object

    Returns:
        a tuple (class, dict of arguments), or (None, None) if the key
        is not present in the config dictionary
    """
    if key not in config or config[key] is None:
        return None, None
    elif isinstance(config[key], str):
        return s2c(config[key]), {}
    elif isinstance(config[key], dict):
        return (
            s2c(config[key]["class_name"])
            if not return_class_name
            else config[key]["class_name"],
            config[key]["args"] if "args" in config[key] else {},
        )
    else:
        raise NotImplementedError(
            f"Parameter {key} " f"has not been formatted properly"
        )


def clear_screen():
    """
    Clears the CLI interface.
    """
    try:
        os.system("clear")
    except Exception as e:
        try:
            os.system("cls")
        except Exception:
            pass


class ProgressManager:
    r"""
    Class that is responsible for drawing progress bars.

    Args:
        outer_folds (int): number of external folds for model assessment
        inner_folds (int): number of internal folds for model selection
        no_configs (int): number of possible configurations in model selection
        final_runs (int): number of final runs per outer fold once the
            best model has been selected
        show (bool): whether to show the progress bar or not.
            Default is ``True``
    """

    # Possible vars of ``bar_format``:
    #       * ``l_bar, bar, r_bar``,
    #       * ``n, n_fmt, total, total_fmt``,
    #       * ``percentage, elapsed, elapsed_s``,
    #       * ``ncols, nrows, desc, unit``,
    #       * ``rate, rate_fmt, rate_noinv``,
    #       * ``rate_noinv_fmt, rate_inv, rate_inv_fmt``,
    #       * ``postfix, unit_divisor, remaining, remaining_s``

    def __init__(
        self, outer_folds, inner_folds, no_configs, final_runs, show=True
    ):
        self.ncols = 100
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.no_configs = no_configs
        self.final_runs = final_runs
        self.pbars = []
        self.show = show

        if not self.show:
            return

        clear_screen()
        self.show_header()
        for i in range(self.outer_folds):
            for j in range(self.inner_folds):
                self.pbars.append(self._init_selection_pbar(i, j))

        for i in range(self.outer_folds):
            self.pbars.append(self._init_assessment_pbar(i))

        self.show_footer()

        self.times = [{} for _ in range(len(self.pbars))]

    def _init_selection_pbar(self, i: int, j: int):
        """
        Initializes the progress bar for model selection

        Args:
            i (int): the id of the outer fold (from 0 to outer folds - 1)
            j (int): the id of the inner fold (from 0 to inner folds - 1)
        """
        position = i * self.inner_folds + j
        pbar = tqdm.tqdm(
            total=self.no_configs,
            ncols=self.ncols,
            ascii=True,
            position=position,
            unit="config",
            bar_format=" {desc} {percentage:3.0f}%|"
            "{bar}|{n_fmt}/{total_fmt}{postfix}",
        )
        pbar.set_description(f"Out_{i + 1}/Inn_{j + 1}")
        mean = str(datetime.timedelta(seconds=0))
        pbar.set_postfix_str(f"(1 cfg every {mean})")
        return pbar

    def _init_assessment_pbar(self, i: int):
        """
        Initializes the progress bar for risk assessment

        Args:
            i (int): the id of the outer fold (from 0 to outer folds - 1)
        """
        position = self.outer_folds * self.inner_folds + i
        pbar = tqdm.tqdm(
            total=self.final_runs,
            ncols=self.ncols,
            ascii=True,
            position=position,
            unit="config",
            bar_format=" {desc} {percentage:3.0f}%|"
            "{bar}|{n_fmt}/{total_fmt}{postfix}",
        )
        pbar.set_description(f"Final run {i + 1}")
        mean = str(datetime.timedelta(seconds=0))
        pbar.set_postfix_str(f"(1 run every {mean})")
        return pbar

    def show_header(self):
        """
        Prints the header of the progress bar
        """
        """
        \033[F --> move cursor to the beginning of the previous line
        \033[A --> move cursor up one line
        \033[<N>A --> move cursor up N lines
        """
        print(
            f'\033[F\033[A{"*" * ((self.ncols - 21) // 2 + 1)} '
            f'Experiment Progress {"*" * ((self.ncols - 21) // 2)}\n'
        )

    def show_footer(self):
        """
        Prints the footer of the progress bar
        """
        pass  # need to work how how to print after tqdm

    def refresh(self):
        """
        Refreshes the progress bar
        """

        self.show_header()
        for i, pbar in enumerate(self.pbars):

            # When resuming, do not consider completed exp. (delta approx. < 1)
            completion_times = [
                delta
                for k, (delta, completed) in self.times[i].items()
                if completed and delta > 1
            ]

            if len(completion_times) > 0:
                min_seconds = min(completion_times)
                max_seconds = max(completion_times)
                mean_seconds = sum(completion_times) / len(completion_times)
            else:
                min_seconds = 0
                max_seconds = 0
                mean_seconds = 0

            mean_time = str(datetime.timedelta(seconds=mean_seconds)).split(
                "."
            )[0]
            min_time = str(datetime.timedelta(seconds=min_seconds)).split(".")[
                0
            ]
            max_time = str(datetime.timedelta(seconds=max_seconds)).split(".")[
                0
            ]

            pbar.set_postfix_str(
                f"min:{min_time}|avg:{mean_time}|max:{max_time}"
            )

            pbar.refresh()
        self.show_footer()

    def update_state(self, msg: dict):
        """
        Updates the state of the progress bar (different from showing it
        on screen, see :func:`refresh`) once a message is received

        Args:
            msg (dict): message with updates to be parsed
        """
        if not self.show:
            return

        try:
            type = msg.get("type")

            if type == END_CONFIG:
                outer_fold = msg.get(OUTER_FOLD)
                inner_fold = msg.get(INNER_FOLD)
                config_id = msg.get(CONFIG_ID)
                position = outer_fold * self.inner_folds + inner_fold
                elapsed = msg.get(ELAPSED)
                configs_times = self.times[position]
                # Compute delta t for a specific config
                configs_times[config_id] = (
                    elapsed,
                    True,
                )  # (time.time() - configs_times[config_id][0], True)
                # Update progress bar
                self.pbars[position].update()
                self.refresh()
            elif type == END_FINAL_RUN:
                outer_fold = msg.get(OUTER_FOLD)
                run_id = msg.get(RUN_ID)
                position = self.outer_folds * self.inner_folds + outer_fold
                elapsed = msg.get(ELAPSED)
                configs_times = self.times[position]
                # Compute delta t for a specific config
                configs_times[run_id] = (
                    elapsed,
                    True,
                )  # (time.time() - configs_times[run_id][0], True)
                # Update progress bar
                self.pbars[position].update()
                self.refresh()
            else:
                raise Exception(
                    f"Cannot parse type of message {type}, fix this."
                )

        except Exception as e:
            print(e)
            return

    def __enter__(self):
        """
        Needed when Progress Manager is used as context manager.
        Does nothing besides returning self.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Needed when Progress Manager is used as context manager.
        Closes the progress bar.
        """
        for pbar in self.pbars:
            pbar.close()


"""
Various options for random search model selection
"""


def choice(*args):
    """
    Implements a random choice between a list of values
    """
    return random.choice(args)


def uniform(*args):
    """
    Implements a uniform sampling given an interval
    """
    return random.uniform(*args)


def normal(*args):
    """
    Implements a univariate normal sampling given its parameters
    """
    return random.normalvariate(*args)


def randint(*args):
    """
    Implements a random integer sampling in an interval
    """
    return random.randint(*args)


def loguniform(*args):
    r"""
    Performs a log-uniform random selection.

    Args:
        *args: a tuple of (log min, log max, [base]) to use. Base 10 is used
            if the third argument is not available.

    Returns:
        a randomly chosen value
    """
    log_min, log_max, *base = args
    base = base[0] if len(base) > 0 else 10

    log_min = math.log(log_min) / math.log(base)
    log_max = math.log(log_max) / math.log(base)

    return base ** (random.uniform(log_min, log_max))


def retrieve_experiments(model_selection_folder,
                         skip_results_not_found : bool =False) -> List[dict]:
    """
    Once the experiments are done, retrieves the config_results.json files of
    all configurations in a specific model selection folder, and returns them
    as a list of dictionaries

    :param model_selection_folder: path to the folder of a model selection,
        that is, your_results_path/..../MODEL_SELECTION/
    :param skip_results_not_found: whether to skip an experiment if a
        `config_results.json` file has not been produced yet. Useful when
        analyzing experiments while others still run.
    :return: a list of dictionaries, one per configuration, each with an extra
        key "exp_folder" which identifies the config folder.
    """
    config_directory = os.path.join(model_selection_folder)

    if not os.path.exists(config_directory):
        raise FileNotFoundError(f"Directory not found: {config_directory}")

    folder_names = []
    for _, dirs, _ in os.walk(config_directory):
        for d in dirs:
            if "config" in d:
                folder_names.append(os.path.join(config_directory, d))
        break  # do not recursively explore subfolders

    configs = []
    for cf in folder_names:
        config_results_path = os.path.join(cf, "config_results.json")
        if not os.path.exists(config_results_path) and skip_results_not_found:
            continue

        exp_info = json.load(
            open(config_results_path, "rb")
        )
        exp_config = exp_info

        exp_config["exp_folder"] = cf
        configs.append(exp_config)

    return configs


def filter_experiments(
    config_list: List[dict], logic: bool = "AND", parameters: dict = {}
):
    """
    Filters the list of configurations returned by the method ``retrieve_experiments`` according to a dictionary.
    The dictionary contains the keys and values of the configuration files you are looking for.

    If you specify more then one key/value pair to look for, then the `logic` parameter specifies whether you want to filter
    using the AND or OR rule.

    For a key, you can specify more than one possible value you are interested in by passing a list as the value, for instance
    {'device': 'cpu', 'lr': [0.1, 0.01]}

    Args:
        config_list: The list of configuration files
        logic: if ``AND``, a configuration is selected iff all conditions are satisfied. If ``OR``, a config is selected when at least
            one of the criteria is met.
        parameters: dictionary with parameters used to filter the configurations

    Returns:
        a list of filtered configurations like the one in input
    """

    def _finditem(obj, key):
        if key in obj:
            return obj[key], True

        for k, v in obj.items():
            if isinstance(v, dict):
                item, found = _finditem(v, key)
                if found:
                    return item, True

        return None, False

    assert logic in ["AND", "OR"], "logic can only be AND/OR case sensitive"

    filtered_config_list = []

    for config in config_list:
        keep = True if logic == "AND" else False

        for k, v in parameters.items():

            cf_v, found = _finditem(config, k)
            assert found, (
                f"Key {k} not found in the " f"configuration, check your input"
            )

            if type(v) == list:
                assert len(v) > 0, (
                    f'the list of values for key "{k}" cannot be'
                    f" empty, consider removing this key"
                )

                # the user specified a list of acceptable values
                # it is sufficient that one of them is present to return True
                if cf_v in v and logic == "OR":
                    keep = True
                    break

                if cf_v not in v and logic == "AND":
                    keep = False
                    break

            else:

                if v == cf_v and logic == "OR":
                    keep = True
                    break

                if v != cf_v and logic == "AND":
                    keep = False
                    break

        if keep:
            filtered_config_list.append(config)

    return filtered_config_list


def retrieve_best_configuration(model_selection_folder) -> dict:
    """
    Once the experiments are done, retrieves the winning configuration from
    a specific model selection folder, and returns it as a dictionaries

    :param model_selection_folder: path to the folder of a model selection,
        that is, your_results_path/..../MODEL_SELECTION/
    :return: a dictionary with info about the best configuration
    """
    config_directory = os.path.join(model_selection_folder)

    if not os.path.exists(config_directory):
        raise FileNotFoundError(f"Directory not found: {config_directory}")

    best_config = json.load(
        open(os.path.join(config_directory, "winner_config.json"), "rb")
    )
    return best_config


def instantiate_dataset_from_config(config: dict) -> DatasetInterface:
    """
    Instantiate a dataset from a configuration file.
    :param config (dict): the configuration file
    :return: an instance of DatasetInterface, i.e., the dataset
    """
    data_root = config[CONFIG][DATA_ROOT]
    dataset_name = config[CONFIG][DATASET]
    dataset_class = s2c(config[CONFIG][DATASET_CLASS])

    return dataset_class(data_root, dataset_name)


def instantiate_data_provider_from_config(config: dict,
                                          splits_filepath: str,
                                          n_outer_folds: int,
                                          n_inner_folds: int) -> DataProvider:
    """
    Instantiate a data provider from a configuration file.
    :param config (dict): the configuration file
    :param splits_filepath (str): the path to data splits file
    :param n_outer_folds (int): the number of outer folds
    :param n_inner_folds (int): the number of inner folds
    :return: an instance of DataProvider, i.e., the data provider
    """
    data_root = config[CONFIG][DATA_ROOT]
    dataset_name = config[CONFIG][DATASET]
    dataset_class = s2c(config[CONFIG][DATASET_CLASS])
    dataset_getter = s2c(config[CONFIG][DATASET_GETTER])
    dl_class, dl_args = return_class_and_args(config[CONFIG], DATA_LOADER)

    return dataset_getter(
        data_root=data_root,
        splits_filepath=splits_filepath,
        dataset_class=dataset_class,
        dataset_name=dataset_name,
        data_loader_class=dl_class,
        data_loader_args=dl_args,
        outer_folds=n_outer_folds,
        inner_folds=n_inner_folds
    )


def instantiate_model_from_config(config: dict,
                                  dataset: DatasetInterface,
                                  config_type: str = "supervised_config") -> ModelInterface:
    """
    Instantiate a model from a configuration file.
    :param config (dict): the configuration file
    :param dataset (DatasetInterface): the dataset used in the experiment
    :param config_type (str): the type of model in ["supervised_config", "unsupervised_config"],
        as written on the YAML experiment configuration file. Defaults to "supervised_config"
    :return: an instance of ModelInterface, i.e., the model
    """
    config_ = config[CONFIG][config_type]

    readout_str = config_.get(READOUT)
    readout_class = s2c(readout_str) if readout_str is not None else None

    model_class = s2c(config_[MODEL])
    model = model_class(dataset.dim_node_features,
                        dataset.dim_edge_features,
                        dataset.dim_target,
                        readout_class,
                        config=config_)

    return model


def load_checkpoint(checkpoint_path: str, model: ModelInterface,
                    device: torch.device):
    """
    Load a checkpoint from a checkpoint file into a model.
    :param checkpoint_path: the checkpoint file path
    :param model (ModelInterface): the model
    :param device (torch.device): the device, e.g, "cpu" or "cuda"
    """
    ckpt_dict = torch.load(
        checkpoint_path, map_location="cpu" if device == "cpu" else None
    )
    model_state = ckpt_dict[MODEL_STATE]

    # Needed only when moving from cpu to cuda (due to changes in config
    # file). Move all parameters to cuda.
    for param in model_state.keys():
        model_state[param] = model_state[param].to(device)

    model.load_state_dict(model_state)