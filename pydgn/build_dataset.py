import argparse
import os
import sys

import yaml

from pydgn.data.util import preprocess_data
from pydgn.static import CONFIG_FILE_CLI_ARGUMENT, CONFIG_FILE


def get_args_dict() -> dict:
    """
    Processes CLI arguments (i.e., the config file location) and returns
    a dictionary.

    Returns:
        a dictionary with the name of the configuration file in the
        :obj:`pydgn.static.CONFIG_FILE` field.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        CONFIG_FILE_CLI_ARGUMENT,
        dest=CONFIG_FILE,
        help="config file to parse the data",
    )
    return vars(parser.parse_args())


def main():
    """
    Launches the data preparation pipeline.
    """
    # Necessary to locate dotted paths in projects that use PyDGN
    sys.path.append(os.getcwd())

    args = get_args_dict()
    options = yaml.load(open(args[CONFIG_FILE], "r"), Loader=yaml.FullLoader)
    preprocess_data(options)
