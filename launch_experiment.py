import os

from pydgn.experiment.util import s2c
from pydgn.static import *

os.environ[OMP_NUM_THREADS] = "1"  # This is CRUCIAL to avoid bottlenecks when running experiments in parallel. DO NOT REMOVE IT

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

# Disable info logging from rdflib
logging.getLogger("rdflib").setLevel(logging.WARNING)

# Ignore warnings
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import warnings
warnings.simplefilter("error", UserWarning) 

from pydgn.data.splitter import Splitter
from pydgn.evaluation.grid import Grid
from pydgn.evaluation.random_search import RandomSearch
from pydgn.evaluation.util import set_gpus


def get_key(key, priority_dict, config_dict):
    v = priority_dict.get(key, None)
    if v is None:
        assert key in config_dict, f"{key.replace('_', '-')} not specified. You can do this via the command line (priority!) or through the config file."
        return config_dict[key]
    return v


def evaluation(args):
    kwargs = vars(args)

    configs_dict = yaml.load(open(kwargs[CONFIG_FILE], "r"),
                             Loader=yaml.FullLoader)

    # The following variables are needed to start the evaluation protocol,
    # but they can be also put at the beginning of the configuration file
    data_root = get_key(DATA_ROOT, kwargs, configs_dict)
    data_splits_filepath = get_key(DATA_SPLITS_FILE, kwargs, configs_dict)
    dataset_class = get_key(DATASET_CLASS, kwargs, configs_dict)
    data_loader_class = get_key(DATA_LOADER, kwargs, configs_dict)
    dataset_name = get_key(DATASET_NAME, kwargs, configs_dict)
    debug = get_key(DEBUG, kwargs, configs_dict)
    final_training_runs = get_key(FINAL_TRAINING_RUNS, kwargs, configs_dict)
    max_cpus = get_key(MAX_CPUS, kwargs, configs_dict)
    max_gpus = get_key(MAX_GPUS, kwargs, configs_dict)
    gpus_per_task = get_key(GPUS_PER_TASK, kwargs, configs_dict)
    splits_folder = get_key(SPLITS_FOLDER, kwargs, configs_dict)
    result_folder = get_key(RESULT_FOLDER, kwargs, configs_dict)

    # Overwrite configs file options with command line ones, which have priority
    for k, v in kwargs.items():
        if k not in [DATA_ROOT, DATASET_CLASS, DATASET_NAME]:
            # if the argument was indeed passed with a value to overwrite
            if v is not None:
                configs_dict[k] = v

    assert GRID_SEARCH in configs_dict or RANDOM_SEARCH in configs_dict

    search_class = (Grid if GRID_SEARCH in configs_dict else RandomSearch)
    search = search_class(data_root, dataset_class, dataset_name, **configs_dict)

    # Set the random seed
    seed = search.seed if search.seed is not None else 42
    print(f'Base seed set to {seed}.')
    experiment = search.experiment
    experiment_class = s2c(experiment)
    use_cuda = CUDA in search.device
    exp_path = os.path.join(result_folder, f"{search.exp_name}")

    # Ensure a generic "cuda" device is set when using more than 1 GPU
    # We will choose the GPU with least ratio of memory usage
    if use_cuda:
        search.device = CUDA

    # Prepare the GPUs, in case they are needed
    set_gpus(max_gpus)

    if not use_cuda:
        max_gpus = 0
        gpus_per_task = 0

    os.environ[PYDGN_RAY_NUM_GPUS_PER_TASK] = str(int(gpus_per_task))

    if os.environ.get('ip_head') is not None:
        assert os.environ.get('redis_password') is not None
        ray.init(address=os.environ.get('ip_head'), _redis_password=os.environ.get('redis_password'))
        print("Connected to Ray cluster.")
        print(f"Available nodes: {ray.nodes()}")
    else:
        ray.init(num_cpus=max_cpus, num_gpus=max_gpus)
        print(f"Started local ray instance.")

    splitter = Splitter.load(data_splits_filepath)
    inner_folds, outer_folds = splitter.n_inner_folds, splitter.n_outer_folds
    print(f'Data splits loaded, outer folds are {outer_folds} and inner folds are {inner_folds}')

    # WARNING: leave the import here, it reads env variables set before
    from pydgn.evaluation.evaluator import RiskAssesser
    risk_assesser = RiskAssesser(outer_folds, inner_folds, experiment_class, exp_path, splits_folder,
                                 data_splits_filepath, search,
                                 final_training_runs=final_training_runs,
                                 higher_is_better=search.higher_results_are_better, gpus_per_task=gpus_per_task,
                                 base_seed=seed)
    risk_assesser.risk_assessment(debug=debug)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_FILE_CLI_ARGUMENT, dest=CONFIG_FILE)
    parser.add_argument(DATA_ROOT_CLI_ARGUMENT, dest=DATA_ROOT, default=None)
    parser.add_argument(DATA_SPLITS_FILE_CLI_ARGUMENT, dest=DATA_SPLITS_FILE, default=None)
    parser.add_argument(DATASET_CLASS_CLI_ARGUMENT, dest=DATASET_CLASS, default=None)
    parser.add_argument(DATA_LOADER_CLASS_CLI_ARGUMENT, dest=DATA_LOADER, default=None)
    parser.add_argument(DATASET_GETTER_CLI_ARGUMENT, dest=DATASET_GETTER, default=None)
    parser.add_argument(DATASET_NAME_CLI_ARGUMENT, dest=DATASET_NAME, default=None)
    parser.add_argument(DEBUG_CLI_ARGUMENT, action="store_true", dest=DEBUG, default=False)
    parser.add_argument(DEVICE_CLI_ARGUMENT, dest=DEVICE, default=None)
    parser.add_argument(EXPERIMENT_CLI_ARGUMENT, dest=EXPERIMENT, default=None)
    parser.add_argument(FINAL_TRAINING_RUNS_CLI_ARGUMENT, dest=FINAL_TRAINING_RUNS,
                        default=1, type=int)
    parser.add_argument(GPUS_PER_TASK_CLI_ARGUMENT, dest=GPUS_PER_TASK, default=1,
                        type=float)
    parser.add_argument(HIGHER_RESULTS_ARE_BETTER_CLI_ARGUMENT, action="store_true",
                        dest=HIGHER_RESULTS_ARE_BETTER, default=None)
    parser.add_argument(LOG_EVERY_CLI_ARGUMENT, dest=LOG_EVERY, default=1, type=int)
    parser.add_argument(MAX_CPUS_CLI_ARGUMENT, dest=MAX_CPUS, default=1, type=int)
    parser.add_argument(MAX_GPUS_CLI_ARGUMENT, dest=MAX_GPUS, default=1, type=int)
    parser.add_argument(MODEL_CLI_ARGUMENT, dest=MODEL, default=None)
    parser.add_argument(RESULT_FOLDER_CLI_ARGUMENT, dest=RESULT_FOLDER, default=None)
    parser.add_argument(SPLITS_FOLDER_CLI_ARGUMENT, dest=SPLITS_FOLDER, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        evaluation(args)
    except Exception as e:
        raise e
