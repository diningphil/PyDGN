import argparse
from config.utils import s2c
from datasets.splitter import Splitter
from datasets.utils import preprocess_data
from utils.serialization import load_yaml


def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', dest="config_file",
                        help='config file to parse the data')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = get_args_dict()
    options = load_yaml(args["config_file"])
    preprocess_data(options)
