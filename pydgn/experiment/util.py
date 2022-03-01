from pydoc import locate
from typing import Callable


def s2c(class_name: str) -> Callable[..., object]:
    r"""
    Converts a dotted path to the corresponding class

    Args:
         class_name (str): dotted path to class name

    Returns:
        the class to be used
    """
    result = locate(class_name)
    if result is None:
        raise ImportError(f"The (dotted) path '{class_name}' is unknown. Check your configuration.")
    return result
