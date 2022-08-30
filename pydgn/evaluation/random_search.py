from copy import deepcopy

from pydgn.evaluation.grid import Grid
from pydgn.experiment.util import s2c
from pydgn.static import *


class RandomSearch(Grid):
    r"""
    Class that implements random-search. It computes all possible
    configurations starting from a suitable config file.

    Args:
        configs_dict (dict): the configuration dictionary specifying the
            different configurations to try
    """
    __search_type__ = RANDOM_SEARCH

    def __init__(self, configs_dict: dict):
        self.num_samples = configs_dict[NUM_SAMPLES]
        super().__init__(configs_dict)

    def _gen_helper(self, cfgs_dict):
        r"""
        Takes a dictionary of key:list pairs and computes all possible
        combinations.

        Returns:
            A list of al possible configurations in the form of dictionaries
        """
        keys = cfgs_dict.keys()
        param = list(keys)[0]

        for _ in range(self.num_samples):
            result = {}
            for key, values in cfgs_dict.items():
                # BASE CASE: key is associated to an atomic value
                if type(values) in [str, int, float, bool, None]:
                    result[key] = values
                # DICT CASE: call _dict_helper on this dict
                elif isinstance(values, dict):
                    result[key] = self._dict_helper(deepcopy(values))

            yield deepcopy(result)

    def _dict_helper(self, configs):
        r"""
        Recursively parses a dictionary

        Returns:
            A dictionary
        """
        if SAMPLE_METHOD in configs:
            return self._sampler_helper(configs)

        for key, values in configs.items():
            if isinstance(values, dict):
                configs[key] = self._dict_helper(configs[key])

        return configs

    def _sampler_helper(self, configs):
        r"""
        Samples possible hyperparameter(s) and returns it
        (them, in this case as a dict)

         Returns:
             A dictionary
        """
        method, args = configs[SAMPLE_METHOD], configs[ARGS]
        sampler = s2c(method)
        sample = sampler(*args)

        if isinstance(sample, dict):
            return self._dict_helper(sample)

        return sample

    def __iter__(self):
        """
        Iterates over all hyper-parameter configurations (generated just once)
        """
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return iter(self.hparams)

    def __len__(self):
        """
        Computes the number of hyper-parameter configurations to try
        """
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return len(self.hparams)

    def __getitem__(self, index):
        """
        Gets a specific configuration indexed by an id
        """
        if self.hparams is None:
            self.hparams = self._gen_configs()
        return self.hparams[index]
