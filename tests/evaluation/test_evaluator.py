import json
from shutil import rmtree

import numpy as np
import yaml

from pydgn.evaluation.evaluator import RiskAssesser
from pydgn.evaluation.grid import Grid
from pydgn.experiment.experiment import Experiment
from pydgn.static import DATA_SPLITS_FILE, LOSS, SCORE, MAIN_LOSS, MAIN_SCORE


class FakeTask(Experiment):
    def run_valid(self, dataset_getter, logger):
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k

        train_loss = {MAIN_LOSS: outer_k + inner_k}
        train_score = {MAIN_SCORE: outer_k + inner_k}
        val_loss = {MAIN_LOSS: outer_k + inner_k + 1}
        val_score = {MAIN_SCORE: outer_k + inner_k + 1}

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}

        return train_res, val_res

    def run_test(self, dataset_getter, logger):
        outer_k = dataset_getter.outer_k

        train_loss = {MAIN_LOSS: outer_k}
        train_score = {MAIN_SCORE: outer_k}
        val_loss = {MAIN_LOSS: outer_k + 1}
        val_score = {MAIN_SCORE: outer_k + 1}
        test_loss = {MAIN_LOSS: outer_k + 2}
        test_score = {MAIN_SCORE: outer_k + 2}

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}

        return train_res, val_res, test_res


# This test activates most of the library's main routines.
def test_evaluator():
    results_folder = "tests/evaluation/debug_evaluator/"
    search = Grid(
        yaml.load(
            open("tests/evaluation/grid_search.yml", "r"),
            Loader=yaml.FullLoader,
        )
    )
    search.telegram_config = None
    splits_filepath = search.configs_dict.get(DATA_SPLITS_FILE)

    evaluator = RiskAssesser(
        10,
        10,
        FakeTask,
        results_folder,
        splits_filepath,
        search,
        10,
        True,
        0,
        base_seed=42,
    )
    # the goal is to evaluate the code that computes results, to ensure
    # it is correct. For this, we don't need debug=False, ow Ray should be
    # started.
    evaluator.risk_assessment(debug=True)

    for outer_k in range(10):
        inner_train_results = np.array([float(i + outer_k) for i in range(10)])
        inner_val_results = np.array(
            [float(i + outer_k) + 1 for i in range(10)]
        )

        ms_results = json.load(
            open(
                "tests/evaluation/debug_evaluator/"
                + f"MODEL_ASSESSMENT/OUTER_FOLD_{outer_k+1}/"
                + "MODEL_SELECTION/winner_config.json",
                "r",
            )
        )

        assert ms_results["avg_training_loss"] == inner_train_results.mean()
        assert ms_results["avg_validation_loss"] == inner_val_results.mean()

    outer_train_results = np.array([float(i) for i in range(10)])
    outer_val_results = np.array([float(i) + 1 for i in range(10)])
    outer_test_results = np.array([float(i) + 2 for i in range(10)])

    ass_results = json.load(
        open(
            "tests/evaluation/debug_evaluator/"
            + "MODEL_ASSESSMENT/assessment_results.json",
            "r",
        )
    )

    assert ass_results["avg_training_main_score"] == outer_train_results.mean()
    assert ass_results["avg_validation_main_score"] == outer_val_results.mean()
    assert ass_results["avg_test_main_score"] == outer_test_results.mean()

    assert ass_results["std_training_main_score"] == outer_train_results.std()
    assert ass_results["std_validation_main_score"] == outer_val_results.std()
    assert ass_results["std_test_main_score"] == outer_test_results.std()

    rmtree(results_folder)
