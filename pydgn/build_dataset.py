import os
import sys
import argparse

import yaml

from pydgn.data.util import preprocess_data
from pydgn.static import CONFIG_FILE_CLI_ARGUMENT, CONFIG_FILE


def get_args_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_FILE_CLI_ARGUMENT, dest=CONFIG_FILE, help='config file to parse the data')
    return vars(parser.parse_args())


def main():
    # Necessary to locate dotted paths in projects that use PyDGN
    sys.path.append(os.getcwd())

    args = get_args_dict()
    options = yaml.load(open(args[CONFIG_FILE], "r"), Loader=yaml.FullLoader)
    preprocess_data(options)
