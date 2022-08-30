from copy import deepcopy
from typing import List

from pydgn.evaluation.util import return_class_and_args
from pydgn.static import *


class Grid:
    r"""
    Class that implements grid-search. It computes all possible configurations
    starting from a suitable config file.

    Args:
        configs_dict (dict): the configuration dictionary specifying the
            different configurations to try
    """
    __search_type__ = GRID_SEARCH

    def __init__(self, configs_dict: dict):

        self.configs_dict = configs_dict
        self.seed = self.configs_dict.get(SEED, None)
        self._exp_name = self.configs_dict.get(EXP_NAME)
        self.data_root = self.configs_dict[DATA_ROOT]
        self.dataset_class = self.configs_dict[DATASET_CLASS]
        self.dataset_name = self.configs_dict[DATASET_NAME]
        self.data_loader_class, self.data_loader_args = return_class_and_args(
            self.configs_dict, DATA_LOADER, return_class_name=True
        )
        self.experiment = self.configs_dict[EXPERIMENT]
        self.higher_results_are_better = self.configs_dict[
            HIGHER_RESULTS_ARE_BETTER
        ]
        self.evaluate_every = self.configs_dict[evaluate_every]
        self.device = self.configs_dict[DEVICE]
        self.dataset_getter = self.configs_dict[DATASET_GETTER]

        # This MUST be called at the END of the init method!
        self.hparams = self._gen_configs()

    def _gen_configs(self) -> List[dict]:
        r"""
        Takes a dictionary of key:list pairs and computes all possible
        combinations.

        Returns:
            A list of al possible configurations in the form of dictionaries
        """
        configs = [
            cfg
            for cfg in self._gen_helper(
                self.configs_dict[self.__search_type__]
            )
        ]
        for cfg in configs:
            cfg.update(
                {
                    DATASET: self.dataset_name,
                    DATASET_GETTER: self.dataset_getter,
                    DATA_LOADER: self.data_loader_class,
                    DATA_LOADER_ARGS: self.data_loader_args,
                    DATASET_CLASS: self.dataset_class,
                    DATA_ROOT: self.data_root,
                    DEVICE: self.device,
                    EXPERIMENT: self.experiment,
                    HIGHER_RESULTS_ARE_BETTER: self.higher_results_are_better,
                    evaluate_every: self.evaluate_every,
                }
            )
        return configs

    def _gen_helper(self, cfgs_dict: dict) -> dict:
        """
        Helper generator that yields one possible configuration at a time.
        """
        keys = cfgs_dict.keys()
        result = {}

        if cfgs_dict == {}:
            yield {}
        else:
            configs_copy = deepcopy(cfgs_dict)  # create a copy to remove keys

            # process the "first" key
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = cfgs_dict[param]

            # BASE CASE: key is associated to an atomic value
            if (
                isinstance(first_key_values, str)
                or isinstance(first_key_values, int)
                or isinstance(first_key_values, float)
                or isinstance(first_key_values, bool)
                or first_key_values is None
            ):
                # print(f'FIRST loop {first_key_values}')
                result[param] = first_key_values

                # Can be {}, hence no loop
                if configs_copy == {}:
                    yield deepcopy(result)
                else:
                    # recursive call on the other keys
                    for nested_config in self._gen_helper(configs_copy):
                        result.update(nested_config)
                        yield deepcopy(result)

            # LIST CASE: you should call _list_helper recursively on each
            # element
            elif isinstance(first_key_values, list):

                for sub_config in self._list_helper(first_key_values):
                    result[param] = sub_config

                    # Can be {}, hence no loop
                    if configs_copy == {}:
                        yield deepcopy(result)
                    else:
                        # recursive call on the other keys
                        for nested_config in self._gen_helper(configs_copy):
                            result.update(nested_config)
                            yield deepcopy(result)

            # DICT CASE: you should recursively call _grid _gen_helper on this
            # dict
            elif isinstance(first_key_values, dict):

                for value_config in self._gen_helper(first_key_values):
                    result[param] = value_config

                    # Can be {}, hence no loop
                    if configs_copy == {}:
                        yield deepcopy(result)
                    else:
                        # recursive call on the other keys
                        for nested_config in self._gen_helper(configs_copy):
                            result.update(nested_config)
                            yield deepcopy(result)

    def _list_helper(self, values: object) -> object:
        """
        Recursively parses lists of possible options for a given
        hyper-parameter.
        """
        for value in values:
            if (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, bool)
                or value is None
            ):
                yield value
            elif isinstance(value, dict):
                for cfg in self._gen_helper(value):
                    yield cfg
            elif isinstance(value, list):
                for cfg in self._list_helper(value):
                    yield cfg

    def __iter__(self):
        """
        Iterates over all hyper-parameter configurations
        """
        return iter(self.hparams)

    def __len__(self):
        """
        Computes the number of hyper-parameter configurations to try
        """
        return len(self.hparams)

    def __getitem__(self, index):
        """
        Gets a specific configuration indexed by an id
        """
        return self.hparams[index]

    @property
    def exp_name(self) -> str:
        r"""
        Computes the name of the root folder

        Returns:
             the name of the root folder as made of ``EXP-NAME_DATASET-NAME``
        """
        return f"{self._exp_name}_{self.dataset_name}"

    @property
    def num_configs(self) -> int:
        r"""
        Computes the number of configurations to try during model selection

        Returns:
             the number of configurations
        """
        return len(self)
