from copy import deepcopy

from pydgn.evaluation.grid import Grid
from pydgn.experiment.util import s2c
from pydgn.static import *


class RandomSearch(Grid):
    r"""
    Class that implements random-search. It computes all possible configurations starting from a suitable config file.

    Args:
        configs_dict (dict): the configuration dictionary specifying the different configurations to try
    """
    __search_type__ = RANDOM_SEARCH

    def __init__(self, configs_dict: dict):
        self.num_samples = configs_dict[NUM_SAMPLES]
        super().__init__(configs_dict)

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
