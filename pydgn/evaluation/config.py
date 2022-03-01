import json
from collections import KeysView, ItemsView


class Config:
    r"""
    Simple class to manage the configuration dictionary as a Python object with fields.

    Args:
        config_dict (dict): the configuration dictionary
    """
    def __init__(self, config_dict: dict):
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

    def keys(self) -> KeysView:
        r"""
        Invokes the `keys()` method of the configuration dictionary

        Returns:
            the set of keys in the dictionary
        """
        return self.config_dict.keys()

    def items(self) -> ItemsView:
        r"""
        Invokes the `items()` method of the configuration dictionary

        Returns:
            a list of (key, value) pairs
        """
        return self.config_dict.items()

    def __str__(self):
        return json.dumps(self.config_dict, sort_keys=True, indent=4)
