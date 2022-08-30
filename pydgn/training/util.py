import os

import torch

from pydgn.static import *


def atomic_save(data: dict, filepath: str):
    r"""
    Atomically stores a dictionary that can be serialized by
    :func:`torch.save`, exploiting the atomic :func:`os.replace`.

    Args:
        data (dict): the dictionary to be stored
        filepath (str): the absolute filepath where to store the dictionary
    """
    try:
        tmp_path = str(filepath) + ATOMIC_SAVE_EXTENSION
        torch.save(data, tmp_path)
        os.replace(tmp_path, filepath)
    except Exception as e:
        os.remove(tmp_path)
        raise e
