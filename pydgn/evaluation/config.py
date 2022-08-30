import json
from collections import KeysView, ItemsView


class Config:
    r"""
    Simple class to manage the configuration dictionary as a Python object
    with fields.

    Args:
        config_dict (dict): the configuration dictionary
    """

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict

    def __getattr__(self, attr):
        """
        Returns the item associated with the key in the dictionary
        """
        return self.config_dict[attr]

    def __getitem__(self, item):
        """
        Returns the item associated with the key in the dictionary
        """
        return self.config_dict[item]

    def __contains__(self, item):
        """
        Returns true if the dictionary contains a key, false otherwise
        """
        return item in self.config_dict

    def __len__(self):
        """
        Returns the number of keys in the dictionary
        """
        return len(self.config_dict)

    def __iter__(self):
        """
        Generates an iterable object from the dictionary
        """
        return iter(self.config_dict)

    def get(self, key: str, default: object):
        """
        Returns the key from the dictionary if present, otherwise the default
        value specified

        Args:
            key (str): the key to look up in the dictionary
            default (`object`): the default object

        Returns:
            a value from the dictionaryu
        """
        return self.config_dict.get(key, default)

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
        """
        Computes an indented json representation of the dictionary
        """
        return json.dumps(self.config_dict, sort_keys=True, indent=4)
