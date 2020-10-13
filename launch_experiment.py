import os
os.environ["OMP_NUM_THREADS"] = "1"  # This is CRUCIAL to avoid bottlenecks when running experiments in parallel. DO NOT REMOVE IT
import sys
import yaml
import torch
# Needed to avoid thread spawning, conflicts with multi-processing.
# You may set a number > 1 but take into account
# the number of processes on the machine
torch.set_num_threads(1)
import logging
import argparse

import ray

# Disable info logging from rdflib and dgl
logging.getLogger("rdflib").setLevel(logging.WARNING)
logging.getLogger("dgl").setLevel(logging.ERROR)

# Ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from data.splitter import Splitter
from evaluation.grid import Grid
from evaluation.evaluator import RiskAssesser
from evaluation.util import set_gpus
from experiment.util import s2c


def get_key(key, priority_dict, config_dict):
    v = priority_dict.get(key, None)
    if v is None:
        assert key in config_dict, f"{key.replace('_','-')} not specified. You can do this via the command line (priority!) or through the config file."
        return config_dict[key]
    return v


def evaluation(args):

    kwargs = vars(args)

    configs_dict = yaml.load(open(kwargs['config_file'], "r"),
                             Loader=yaml.FullLoader)

    # The following variables are needed to start the evaluation protocol,
    # but they can be also put at the beginning of the configuration file
    data_root = get_key('data_root', kwargs, configs_dict)
    data_splits_file = get_key('data_splits_file', kwargs, configs_dict)
    dataset_class = get_key('dataset_class', kwargs, configs_dict)
    dataset_name = get_key('dataset_name', kwargs, configs_dict)
    dataset_name = get_key('dataset_name', kwargs, configs_dict)
    debug = get_key('debug', kwargs, configs_dict)
    final_training_runs = get_key('final_training_runs', kwargs, configs_dict)
    max_cpus = get_key('max_cpus', kwargs, configs_dict)
    max_gpus = get_key('max_gpus', kwargs, configs_dict)
    gpus_per_task = get_key('gpus_per_task', kwargs, configs_dict)
    splits_folder = get_key('splits_folder', kwargs, configs_dict)
    result_folder = get_key('result_folder', kwargs, configs_dict)

    # Overwrite configs file options with command line ones, which have priority
    for k,v in kwargs.items():
        if k not in ['data_root', 'dataset_class', 'dataset_name']:
            # if the argument was indeed passed with a value to overwrite
            if v is not None:
                configs_dict[k] = v

    grid = Grid(data_root, dataset_class, dataset_name, **configs_dict)

    experiment = grid.experiment
    experiment_class = s2c(experiment)
    use_cuda = 'cuda' in grid.device
    exp_path = os.path.join(result_folder, f"{grid.exp_name}_{experiment.split('.')[-1]}")

    # Ensure a generic "cuda" device is set when using more than 1 GPU
    # We will choose the GPU with least ratio of memory usage
    if use_cuda:
        grid.device = "cuda"

    # Prepare the GPUs, in case they are needed
    set_gpus(max_gpus)

    if not use_cuda:
        # Users should not change default GPU argument values when using CPU devices.
        # It is useless and probably a mistake
        assert max_gpus == 1 and gpus_per_task == 1
        max_gpus = 0
        gpus_per_task = 0

    if os.environ.get('ip_head') is not None:
        # TODO this should handle multi-GPU automatically. Must be tested.
        assert os.environ.get('redis_password') is not None
        ray.init(address=os.environ.get('ip_head'), _redis_password=os.environ.get('redis_password'))
        print("Connected to Ray cluster.")
        print(f"Available nodes: {ray.nodes()}")
    else:
        ray.init(num_cpus=max_cpus, num_gpus=max_gpus)
        print(f"Started local ray instance.")

    splitter = Splitter.load(data_splits_file)

    inner_folds, outer_folds = splitter.n_inner_folds, splitter.n_outer_folds
    print(f'Data splits loaded, outer folds are {outer_folds} and inner folds are {inner_folds}')

    risk_assesser = RiskAssesser(outer_folds, inner_folds, experiment_class, exp_path, splits_folder, grid,
                                 final_training_runs=final_training_runs,
                                 higher_is_better=grid.higher_results_are_better, gpus_per_task=gpus_per_task)
    risk_assesser.risk_assessment(debug=debug)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--data-root', dest='data_root', default=None)
    parser.add_argument('--data-splits', dest='data_splits_file', default=None)
    parser.add_argument('--dataset-class', dest='dataset_class', default=None)
    parser.add_argument('--dataset-getter', dest='dataset_getter', default=None)
    parser.add_argument('--dataset-name', dest='dataset_name', default=None)
    parser.add_argument('--debug', action="store_true", dest='debug', default=False)
    parser.add_argument('--device', dest='device', default=None)
    parser.add_argument('--experiment', dest='experiment', default=None)
    parser.add_argument('--final-training-runs', dest='final_training_runs',
                        default=1, type=int)
    parser.add_argument('--gpus-per-task', dest='gpus_per_task', default=1,
                        type=float)
    parser.add_argument('--higher-results-are-better', action="store_true",
                        dest='higher_results_are_better', default=None)
    parser.add_argument('--log-every', dest='log_every', default=1, type=int)
    parser.add_argument('--max-cpus', dest='max_cpus', default=1, type=int)
    parser.add_argument('--max-gpus', dest='max_gpus', default=1, type=int)
    parser.add_argument('--model', dest='model', default=None)
    parser.add_argument('--num-dataloader-workers',
                        dest='num_dataloader_workers', default=None, type=int)
    parser.add_argument('--pin-memory', dest='pin_memory', action="store_true",
                        default=None)
    parser.add_argument('--result-folder', dest='result_folder', default=None)
    parser.add_argument('--splits-folder', dest='splits_folder', default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        evaluation(args)
    except Exception as e:
        raise e
