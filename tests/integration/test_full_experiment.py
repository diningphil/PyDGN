from shutil import rmtree

import pytest
import yaml

from pydgn.data.util import preprocess_data
from pydgn.launch_experiment import evaluation
from pydgn.static import DEBUG, DEVICE, CONFIG_FILE


def test_dataset_creation():
    config = yaml.load(
        open("tests/integration/dataset_config.yml", "r"),
        Loader=yaml.FullLoader,
    )
    preprocess_data(config)


@pytest.mark.dependency(depends=["test_dataset_creation"])
def test_experiment_completion():
    config = {}
    config[CONFIG_FILE] = "tests/integration/exp_config.yml"
    config[DEBUG] = True

    class MockConfig:
        def __init__(self, d):
            for key in d.keys():
                setattr(self, key, d[key])

    config = MockConfig(config)
    evaluation(config)
    rmtree("tests/integration/debug")
