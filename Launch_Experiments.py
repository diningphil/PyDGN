import os
os.environ["OMP_NUM_THREADS"] = "1"  # This is CRUCIAL to avoid bottlenecks when running experiments in parallel. DO NOT REMOVE IT
import sys
import torch
import logging
import argparse
from pathlib import Path
# Disable info logging from rdflib and dgl
logging.getLogger("rdflib").setLevel(logging.WARNING)
logging.getLogger("dgl").setLevel(logging.ERROR)

# Ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from datasets.splitter import Splitter
from evaluation.grid import Grid
from evaluation.k_fold_selection import KFoldSelection
from evaluation.k_fold_assessment import KFoldAssessment
from experiments.supervised_task import SupervisedTask
from experiments.semi_supervised_task import SemiSupervisedTask
from experiments.incremental_task import IncrementalTask


def evaluation(config_file,
               data_root,
               dataset_class,
               dataset_name,
               data_splits_file,
               outer_processes,
               inner_processes,
               final_training_runs,
               result_folder,
               splits_folder,
               debug):

    # Needed to avoid thread spawning, conflicts with multi-processing. You may set a number > 1 but take into account
    # the number of processes on the machine
    torch.set_num_threads(1)

    grid = Grid.from_file(config_file, data_root, dataset_class, dataset_name)
    experiment = grid.experiment
    use_cuda = 'cuda' in grid.device
    exp_path = os.path.join(result_folder, f'{grid.exp_name}_{experiment}_experiment')

    if experiment == 'supervised':
        experiment_class = SupervisedTask
    elif experiment == 'semi-supervised':
        experiment_class = SemiSupervisedTask
    elif experiment == 'incremental':
        experiment_class = IncrementalTask
    else:
        raise NotImplementedError(f'{experiment} experiment not implemented yet.')

    splitter = Splitter.load(data_splits_file)

    inner_folds, outer_folds = splitter.n_inner_folds, splitter.n_outer_folds
    print(f'Data splits loaded, outer folds are {outer_folds} and inner folds are {inner_folds}')

    model_selector = KFoldSelection(inner_folds, max_processes=inner_processes,
                                        higher_is_better=grid.higher_results_are_better)

    risk_assesser = KFoldAssessment(outer_folds, model_selector, exp_path, splits_folder, grid,
                                        outer_processes=outer_processes, final_training_runs=final_training_runs)

    risk_assesser.risk_assessment(experiment_class, debug=debug, use_cuda=use_cuda)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', dest='dataset_name')
    parser.add_argument('--dataset-class', dest='dataset_class', default='datasets.datasets.TUDatasetInterface')
    parser.add_argument('--data-root', dest='data_root', default='DATA')
    parser.add_argument('--data-splits', dest='data_splits_file')
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--splits-folder', dest='splits_folder', default='SPLITS')
    parser.add_argument('--outer-processes', dest='outer_processes', default=1, type=int)
    parser.add_argument('--inner-processes', dest='inner_processes', default=1, type=int)
    parser.add_argument('--final-training-runs', dest='final_training_runs', default=1, type=int)
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
            args.data_splits_file,
            outer_processes=args.outer_processes,
            inner_processes=args.inner_processes,
            final_training_runs=args.final_training_runs,
            result_folder=args.result_folder,
            splits_folder=args.splits_folder,
            debug=args.debug)
    except Exception as e:
        raise e
