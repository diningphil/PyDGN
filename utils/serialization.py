import json
import yaml
import numpy as np
from pathlib import Path


def load_yaml(path):
    """ Loads a yaml file stored in path """
    path = Path(path)
    return yaml.load(open(path, "r"), Loader=yaml.SafeLoader)


def save_yaml(obj, path):
    """ Saves a dictionary to a yaml file """
    path = Path(path)
    return yaml.dump(obj, open(path, "w"))


def load_numpy(path):
    """ Loads a numpy file stored in path """
    path = Path(path)
    return np.loadtxt(path, ndmin=2)


def save_numpy(matrix, path):
    """ Saves a dictionary to a numpy file """
    path = Path(path)
    np.savetxt(path, matrix)


def load_json(path):
    """ Loads a json file stored in path """
    path = Path(path)
    return json.load(open(path, "r"))


def save_json(obj, path):
    """ Saves a dictionary to a json file """
    path = Path(path)
    json.dump(obj, open(path, "w"))