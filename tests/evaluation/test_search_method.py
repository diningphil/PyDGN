import pytest
import yaml

from pydgn.evaluation.grid import Grid
from pydgn.evaluation.random_search import RandomSearch


@pytest.fixture
def search_method_config_length():
    return [
        (Grid, "tests/evaluation/grid_search.yml", 6),
        (RandomSearch, "tests/evaluation/random_search.yml", 10),
    ]


def test_search_method(search_method_config_length):
    for search_method, filepath, num_of_configs in search_method_config_length:
        search = search_method(
            yaml.load(open(filepath, "r"), Loader=yaml.FullLoader)
        )
        # Check the amount of configurations expected and those produced
        # are the same
        assert len(search) == num_of_configs

        # No two configurations should be equal
        # (unless it's intended from the config file)
        for i in range(len(search)):
            for j in range(i + 1, len(search)):
                assert search[i] != search[j]
