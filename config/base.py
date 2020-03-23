from copy import deepcopy
from utils.serialization import load_yaml, save_yaml


class Grid:
    """ This class computes all possible hyper-parameters configurations based on the configuration file """

    @classmethod
    def from_file(cls, path, data_root, dataset_class, dataset_name):
        configs_dict = load_yaml(path)
        return cls(data_root, dataset_class, dataset_name, **configs_dict)

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
        self.data_root = data_root
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name
        self.experiment = self.configs_dict['experiment']
        self.higher_results_are_better = self.configs_dict['higher_results_are_better']
        self.device = self.configs_dict['device']
        self.model = self.configs_dict['model']
        self.dataset_getter = self.configs_dict['dataset-getter']
        self.hparams = self._gen_configs()

    def _gen_configs(self):
        '''
        Takes a dictionary of key:list pairs and computes all possible permutations.
        :return: A list of al possible configurations
        '''
        configs = [cfg for cfg in self._gen_helper(self.configs_dict['grid'])]
        for cfg in configs:
            cfg.update({"dataset": self.dataset_name,
                        "dataset_getter": self.dataset_getter,
                        "dataset_class": self.dataset_class,
                        "data_root": self.data_root,
                        "model": self.model,
                        "device": self.device,
                        "experiment": self.experiment,
                        "higher_results_are_better": self.higher_results_are_better})

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

    def save(self, path):
        save_yaml(self._grid, path)

    @property
    def exp_name(self):
        return f"{self.model.split('.')[-1]}_{self.dataset_name.split('.')[-1]}"

    @property
    def num_configs(self):
        return len(self)


class Config:
    """ Simple class to manage the configuration dictionary """

    def __init__(self, config_dict):
        self.config_dict = config_dict

    def __getattr__(self, attr):
        return self.config_dict[attr]

    def __getitem__(self, item):
        return self.config_dict[item]

    def __contains__(self, item):
        return item in self.config_dict

    def __len__(self):
        return len(self.config_dict)

    def __iter__(self):
        return iter(self.config_dict)

    def keys(self):
        return self.config_dict.keys()

    def items(self):
        return self.config_dict.items()

    def save(self, path):
        save_yaml(self.config_dict, path)
