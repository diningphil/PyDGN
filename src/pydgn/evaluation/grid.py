from copy import deepcopy
from typing import Callable, List

from pydgn.data.dataset import DatasetInterface
from pydgn.static import *


class Grid:
    r"""
    Class that implements grid-search. It computes all possible configurations starting from a suitable config file.

    Args:
        data_root (str): the root directory in which the dataset is stored
        dataset_class (Callable[..., :class:`~pydgn.data.dataset.DatasetInterface`]): class of the dataset to use
        dataset_name (str): the name of the dataset
        configs_dict (dict): the configuration dictionary specifying the different configurations to try
    """
    def __init__(self,
                 data_root: str,
                 dataset_class: Callable[...,DatasetInterface],
                 dataset_name: str,
                 **configs_dict: dict):

        self.configs_dict = configs_dict
        self.seed = self.configs_dict.get(SEED, None)
        self.data_root = data_root
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name
        self.data_loader_class = self.configs_dict[DATA_LOADER]
        self.experiment = self.configs_dict[EXPERIMENT]
        self.higher_results_are_better = self.configs_dict[HIGHER_RESULTS_ARE_BETTER]
        self.log_every = self.configs_dict[LOG_EVERY]
        self.device = self.configs_dict[DEVICE]
        self.num_dataloader_workers = self.configs_dict[NUM_DATALOADER_WORKERS]
        self.pin_memory = self.configs_dict[PIN_MEMORY]
        self.model = self.configs_dict[MODEL]
        self.dataset_getter = self.configs_dict[DATASET_GETTER]

        # This MUST be called at the END of the init method!
        self.hparams = self._gen_configs()

    def _gen_configs(self) -> List[dict]:
        r"""
        Takes a dictionary of key:list pairs and computes all possible combinations.

        Returns:
            A list of al possible configurations in the form of dictionaries
        """
        configs = [cfg for cfg in self._gen_helper(self.configs_dict[GRID_SEARCH])]
        for cfg in configs:
            cfg.update({DATASET: self.dataset_name,
                        DATASET_GETTER: self.dataset_getter,
                        DATA_LOADER: self.data_loader_class,
                        DATASET_CLASS: self.dataset_class,
                        DATA_ROOT: self.data_root,
                        MODEL: self.model,
                        DEVICE: self.device,
                        NUM_DATALOADER_WORKERS: self.num_dataloader_workers,
                        PIN_MEMORY: self.pin_memory,
                        EXPERIMENT: self.experiment,
                        HIGHER_RESULTS_ARE_BETTER: self.higher_results_are_better,
                        LOG_EVERY: self.log_every})
        return configs

    def _gen_helper(self, cfgs_dict: dict) -> dict:
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
            if type(first_key_values) == str or type(first_key_values) == int or type(
                    first_key_values) == float or type(first_key_values) == bool or first_key_values is None:
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

            # LIST CASE: you should call _list_helper recursively on each element
            elif type(first_key_values) == list:

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

            # DICT CASE: you should recursively call _grid _gen_helper on this dict
            elif type(first_key_values) == dict:

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
        for value in values:
            if type(value) == str or type(value) == int or type(value) == float or type(value) == bool or value is None:
                yield value
            elif type(value) == dict:
                for cfg in self._gen_helper(value):
                    yield cfg
            elif type(value) == list:
                for cfg in self._list_helper(value):
                    yield cfg

    def __iter__(self):
        return iter(self.hparams)

    def __len__(self):
        return len(self.hparams)

    def __getitem__(self, index):
        return self.hparams[index]

    @property
    def exp_name(self) -> str:
        r"""
        Computes the name of the root folder

        Returns:
             the name of the root folder as made of MODEL-NAME_DATASET-NAME
        """
        return f"{self.model.split('.')[-1]}_{self.dataset_name}"

    @property
    def num_configs(self) -> int:
        r"""
        Computes the number of configurations to try during model selection

        Returns:
             the number of configurations
        """
        return len(self)
