from copy import deepcopy

from pydgn.experiment.util import s2c
from pydgn.static import *


class RandomSearch:
    """ This class performs a random search for hyper-parameters optimisation over the search spaces defined in the configuration file """

    def __init__(self, data_root, dataset_class, dataset_name, **configs_dict):
        """
        Initializes the RandomSearch object by looking for specific keys in the dictionary, namely 'experiment', 'device',
        'model', 'dataset-getter' and 'higher_results_are_better'. The configuration dictionary should have
        a field named 'random' in which all possible hyper-parameters are listed (see examples)
        :param data_root: the root directory in which the dataset is stored
        :param dataset_class: one of the classes in datasets.datasets that specifies how to process the data
        :param dataset_name: the name of the dataset
        :param configs_dict: the configuration dictionary
        """
        self.configs_dict = configs_dict
        self.seed = self.configs_dict.get(SEED, None)
        self.data_root = data_root
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name
        self.experiment = self.configs_dict[EXPERIMENT]
        self.higher_results_are_better = self.configs_dict[HIGHER_RESULTS_ARE_BETTER]
        self.log_every = self.configs_dict[LOG_EVERY]
        self.device = self.configs_dict[DEVICE]
        self.num_dataloader_workers = self.configs_dict[NUM_DATALOADER_WORKERS]
        self.pin_memory = self.configs_dict[PIN_MEMORY]
        self.model = self.configs_dict[MODEL]
        self.dataset_getter = self.configs_dict[DATASET_GETTER]
        self.num_samples = self.configs_dict[NUM_SAMPLES]

        # For continual learning tasks
        # - Reharsal
        self.n_tasks = self.configs_dict.get(NUM_TASKS, None)
        self.n_rehearsal_patterns_per_task = self.configs_dict.get(NUM_REHEARSAL_PATTERNS_PER_TASK, None)

        # Generation has to be moved to first usage because of reproducibility (seed is set inside RiskAssesser)
        self.hparams = None

    def _gen_configs(self):
        '''
        Takes a dictionary of key:list pairs and computes possible hyper-parameter configurations.
        :return: A list of possible configurations
        '''
        configs = [cfg for cfg in self._gen_helper(self.configs_dict[RANDOM_SEARCH])]
        for cfg in configs:
            cfg.update({DATASET: self.dataset_name,
                        DATASET_GETTER: self.dataset_getter,
                        DATASET_CLASS: self.dataset_class,
                        DATA_ROOT: self.data_root,
                        MODEL: self.model,
                        DEVICE: self.device,
                        NUM_DATALOADER_WORKERS: self.num_dataloader_workers,
                        PIN_MEMORY: self.pin_memory,
                        EXPERIMENT: self.experiment,
                        HIGHER_RESULTS_ARE_BETTER: self.higher_results_are_better,
                        LOG_EVERY: self.log_every,
                        NUM_TASKS: self.n_tasks,
                        NUM_REHEARSAL_PATTERNS_PER_TASK: self.n_rehearsal_patterns_per_task})
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
    def exp_name(self):
        return f"{self.model.split('.')[-1]}_{self.dataset_name}"

    @property
    def num_configs(self):
        return len(self)
