from pydoc import locate


def s2c(class_name):
    """ Converts a dotted path to the corresponding class """
    result = locate(class_name)
    if result is None:
        raise ImportError(f"The (dotted) path '{class_name}' is unknown. Check your configuration.")
    return result
