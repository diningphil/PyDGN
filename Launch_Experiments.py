import os
import torch

from config.base import Grid

from datasets.utils import DATA_DIR

from evaluation.model_selection.hold_out_selection import HoldOutSelection
from evaluation.model_selection.k_fold_selection import KFoldSelection
from evaluation.risk_assessment.hold_out_assessment import HoldOutAssessment
from evaluation.risk_assessment.k_fold_assessment import KFoldAssessment

import argparse

from experiments.incremental_task import IncrementalTask
from experiments.supervised_task import SupervisedTask
from experiments.semi_supervised_task import SemiSupervisedTask


def evaluation(config_file,
               data_root,
               dataset_class,
               dataset_name,
               outer_folds,
               outer_processes,
               inner_folds,
               inner_processes,
               final_training_runs,
               result_folder,
               debug):

    # Needed to avoid thread spawning, conflicts with multi-processing. You may set a number > 1 but take into account
    # the number of processes on the machine
    torch.set_num_threads(1)

    grid = Grid.from_file(config_file, data_root, dataset_class, dataset_name)
    experiment = grid.experiment

    if experiment == 'supervised':
        experiment_class = SupervisedTask
    elif experiment == 'semi-supervised':
        experiment_class = SemiSupervisedTask
    elif experiment == 'incremental':
        experiment_class = IncrementalTask
    else:
        raise NotImplementedError(f'{experiment} experiment not implemented yet.')

    exp_path = os.path.join(result_folder, f'{grid.exp_name}_assessment')
    if inner_folds > 1:
        model_selector = KFoldSelection(inner_folds, max_processes=inner_processes,
                                        higher_is_better=grid.higher_results_are_better)
    else:
        model_selector = HoldOutSelection(max_processes=inner_processes,
                                          higher_is_better=grid.higher_results_are_better)

    if outer_folds > 1:
        risk_assesser = KFoldAssessment(outer_folds, model_selector, exp_path, grid,
                                        outer_processes=outer_processes, final_training_runs=final_training_runs)
    else:
        risk_assesser = HoldOutAssessment(model_selector, exp_path, grid,
                                          final_training_runs=final_training_runs)

    risk_assesser.risk_assessment(experiment_class, debug=debug)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', dest='dataset_name')
    parser.add_argument('--dataset-class', dest='dataset_class', default='torch_geometric.datasets.TUDataset')
    parser.add_argument('--data-root', dest='data_root', default=DATA_DIR.as_posix())
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--outer-folds', dest='outer_folds', default=10, type=int)
    parser.add_argument('--outer-processes', dest='outer_processes', default=2, type=int)
    parser.add_argument('--inner-folds', dest='inner_folds', default=1, type=int)
    parser.add_argument('--inner-processes', dest='inner_processes', default=1, type=int)
    parser.add_argument('--final-training-runs', dest='final_training_runs', default=3, type=int)
    parser.add_argument('--debug', action="store_true", dest='debug', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    try:
        evaluation(
            args.config_file,
            args.data_root,
            args.dataset_class,
            args.dataset_name,
            outer_folds=args.outer_folds,
            outer_processes=args.outer_processes,
            inner_folds=args.inner_folds,
            inner_processes=args.inner_processes,
            final_training_runs=args.final_training_runs,
            result_folder=args.result_folder,
            debug=args.debug)
    except Exception as e:
        raise e