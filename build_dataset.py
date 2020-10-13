import yaml
import argparse
from data.util import preprocess_data


def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', dest="config_file",
                        help='config file to parse the data')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = get_args_dict()
    options = yaml.load(open(args['config_file'], "r"), Loader=yaml.FullLoader)
    preprocess_data(options)
