from copy import deepcopy

from pydgn.static import *


class Grid:
    """ This class computes all possible hyper-parameters configurations based on the configuration file """

    def __init__(self, data_root, dataset_class, dataset_name, **configs_dict):
        """
        Initializes the Grid object by looking for specific keys in the dictionary, namely 'experiment', 'device',
        'model', 'dataset-getter' and 'higher_results_are_better'. The configuration dictionary should have
        a field named 'grid' in which all possible hyper-parameters are listed (see examples)
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

        # For continual learning tasks
        # - Rehearsal
        self.n_tasks = self.configs_dict.get(NUM_TASKS, None)
        self.n_rehearsal_patterns_per_task = self.configs_dict.get(NUM_REHEARSAL_PATTERNS_PER_TASK, None)

        # This MUST be called at the END of the init method!
        self.hparams = self._gen_configs()

    def _gen_configs(self):
        '''
        Takes a dictionary of key:list pairs and computes all possible permutations.
        :return: A list of al possible configurations
        '''
        configs = [cfg for cfg in self._gen_helper(self.configs_dict[GRID_SEARCH])]
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

    def _list_helper(self, values):
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
    def exp_name(self):
        return f"{self.model.split('.')[-1]}_{self.dataset_name}"

    @property
    def num_configs(self):
        return len(self)
