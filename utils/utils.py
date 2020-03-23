import os
import inspect
from pathlib import Path


def get_or_create_dir(path):
    """ Creates directories associated to the specified path if they are missing, and it returns the Path object """
    path = Path(path)
    if not path.exists():
        os.makedirs(path)
    return path


def check_argument(cls, arg_name):
    """ Checks whether arg_name is in the signature of a method or class """
    sign = inspect.signature(cls)
    return arg_name in sign.parameters.keys()
