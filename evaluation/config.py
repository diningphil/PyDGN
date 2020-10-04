import json


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

    def __str__(self):
        return json.dumps(self.config_dict, sort_keys=True, indent=4)
