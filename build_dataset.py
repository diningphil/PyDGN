import argparse

import yaml
from src.pydgn.data import preprocess_data


def get_args_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_FILE_CLI_ARGUMENT, dest=CONFIG_FILE, help='config file to parse the data')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = get_args_dict()
    options = yaml.load(open(args[CONFIG_FILE], "r"), Loader=yaml.FullLoader)
    preprocess_data(options)
