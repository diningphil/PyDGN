from shutil import rmtree

import pytest
import yaml

from pydgn.data.util import preprocess_data
from pydgn.launch_experiment import evaluation
from pydgn.static import DEBUG, DEVICE, CONFIG_FILE


def test_dataset_creation():
    config = yaml.load(
        open("tests/integration/graph_class_dataset_config.yml", "r"),
        Loader=yaml.FullLoader,
    )
    preprocess_data(config)


@pytest.mark.dependency(depends=["test_dataset_creation"])
def test_supervised_experiment():
    class MockConfig:
        def __init__(self, d):
            for key in d.keys():
                setattr(self, key, d[key])

    config = {}
    config[CONFIG_FILE] = "tests/integration/sup_exp_config.yml"
    config[DEBUG] = True
    config = MockConfig(config)
    evaluation(config)


@pytest.mark.dependency(depends=["test_dataset_creation"])
def test_semi_supervised_experiment():
    class MockConfig:
        def __init__(self, d):
            for key in d.keys():
                setattr(self, key, d[key])

    config = {}
    config[CONFIG_FILE] = "tests/integration/semi_sup_exp_config.yml"
    config[DEBUG] = True
    config = MockConfig(config)
    evaluation(config)


@pytest.mark.dependency(
    depends=["test_supervised_experiment", "test_semi_supervised_experiment"]
)
def test_cleanup():
    rmtree("tests/integration/debug")
