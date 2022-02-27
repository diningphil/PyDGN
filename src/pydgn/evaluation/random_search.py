from copy import deepcopy
from typing import Callable, List

from pydgn.data.dataset import DatasetInterface
from pydgn.experiment.util import s2c
from pydgn.static import *


class RandomSearch:
    r"""
    Class that implements random-search. It computes all possible configurations starting from a suitable config file.

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
        self.num_samples = self.configs_dict[NUM_SAMPLES]

        # Generation has to be moved to first usage because of reproducibility (seed is set inside RiskAssesser)
        self.hparams = None

    def _gen_configs(self) -> List[dict]:
        r"""
        Takes a dictionary of key:list pairs and computes all possible combinations.

        Returns:
            A list of al possible configurations in the form of dictionaries
        """
        configs = [cfg for cfg in self._gen_helper(self.configs_dict[RANDOM_SEARCH])]
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

    def _gen_helper(self, cfgs_dict):
        keys = cfgs_dict.keys()
        param = list(keys)[0]

        for _ in range(self.num_samples):
            result = {}
            for key, values in cfgs_dict.items():
                # BASE CASE: key is associated to an atomic value
                if type(values) in [str, int, float, bool, None]:
                    result[key] = values
                # DICT CASE: call _dict_helper on this dict
                elif type(values) == dict:
                    result[key] = self._dict_helper(deepcopy(values))

            yield deepcopy(result)

    def _dict_helper(self, configs):
        if SAMPLE_METHOD in configs:
            return self._sampler_helper(configs)

        for key, values in configs.items():
            if type(values) == dict:
                configs[key] = self._dict_helper(configs[key])

        return configs

    def _sampler_helper(self, configs):
        method, args = configs[SAMPLE_METHOD], configs[ARGS]
        sampler = s2c(method)
        sample = sampler(*args)

        if type(sample) == dict:
            return self._dict_helper(sample)

        return sample

    def __iter__(self):
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return iter(self.hparams)

    def __len__(self):
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return len(self.hparams)

    def __getitem__(self, index):
        if self.hparams is None:
            self.hparams = self._gen_configs()
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
