import argparse
import os
import sys
from typing import Optional

import gpustat
import yaml

from pydgn.static import *


def set_gpus(num_gpus: int, gpus_subset: Optional[str] = None):
    """
    Sets the visible GPUS for the experiments according to the availability
    in terms of memory. Prioritize GPUs with less memory usage.
    Sets the ``CUDA_DEVICE_ORDER`` env variable to ``PCI_BUS_ID``
    and ``CUDA_VISIBLE_DEVICES`` to the ordered list of GPU indices.

    Args:
        num_gpus (int): maximum number of GPUs to use when
            launching experiments in parallel
        gpus_subset (str): specifies a subset of GPU indices
            that the library should use as a string of comma-separated indices.
            Useful if one wants to specify specific GPUs on which to run
            the experiments, regardless of memory constraints
    """
    if (gpus_subset is not None) and (num_gpus > len(gpus_subset)):
        raise Exception(
            f"The user asked to use {num_gpus} GPUs but only a valid subset of"
            f" {len(gpus_subset)} GPUs are provided. "
            f"Please adjust the {str(MAX_GPUS)} and the "
            f"{str(GPUS_SUBSET)} parameters in the configuration file."
        )

    if gpus_subset is not None:
        gpus_subset = gpus_subset.split(",")

    try:
        selected = []

        stats = gpustat.GPUStatCollection.new_query()

        for i in range(num_gpus):

            ids_mem = [
                res
                for res in map(
                    lambda gpu: (
                        int(gpu.entry["index"]),
                        float(gpu.entry["memory.used"])
                        / float(gpu.entry["memory.total"]),
                    ),
                    stats,
                )
                if (str(res[0]) not in selected)
                and (
                    str(res[0]) in gpus_subset
                    if gpus_subset is not None
                    else True
                )
            ]

            if len(ids_mem) == 0:
                # No more gpus available
                break

            best = min(ids_mem, key=lambda x: x[1])
            bestGPU, bestMem = best[0], best[1]
            # print(f"{i}-th best is {bestGPU} with mem {bestMem}")
            selected.append(str(bestGPU))

        if gpus_subset is not None:
            print(
                f"The user specified the following "
                f"GPUs to use: {gpus_subset}"
            )
        print("Setting GPUs to: {}".format(",".join(selected)))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    except BaseException as e:
        print("GPU not available: " + str(e))


def evaluation(options: argparse.Namespace):
    """
    Takes the CLI arguments and launches the evaluation procedure.
    The method takes care of instantiating the Ray process and setting up
    the amount of resources available. Then it instantiates a
    :obj:`RiskAssesser` object to carry out the experiments.

    Args:
        :param options (argparse.Namespace):
    """
    kwargs = vars(options)
    debug = kwargs[DEBUG]
    configs_dict = yaml.load(
        open(kwargs[CONFIG_FILE], "r"), Loader=yaml.FullLoader
    )

    # Telegram bot settings
    telegram_config_file = configs_dict.get(TELEGRAM_CONFIG_FILE, None)
    if telegram_config_file is not None:
        telegram_config = yaml.load(
            open(telegram_config_file, "r"), Loader=yaml.FullLoader
        )
    else:
        telegram_config = None

    # Hardware initialization
    max_cpus = configs_dict[MAX_CPUS]

    device = configs_dict[DEVICE]
    use_cuda = CUDA in device

    if not use_cuda:
        max_gpus = 0
        gpus_per_task = 0
    else:
        # We will choose the GPU with least ratio of memory usage
        max_gpus = configs_dict[MAX_GPUS]
        # Optionally, use a specific subset of gpus
        gpus_subset = configs_dict.get(GPUS_SUBSET, None)
        # Choose which GPUs to use
        try:
            set_gpus(max_gpus, gpus_subset)
        except Exception as e:
            print("An exception occurred while setting the GPUs:")
            print(e)

        gpus_per_task = configs_dict[GPUS_PER_TASK]

    # we probably don't need this anymore, but keep it commented in case
    # OMP_NUM_THREADS = 'OMP_NUM_THREADS'
    # os.environ[OMP_NUM_THREADS] = "1"  # This is CRUCIAL to avoid
    # bottlenecks when running experiments in parallel. DO NOT REMOVE IT

    #
    # Once CUDA_VISIBLE_DEVICES has been set, we can start importing
    # all the necessary modules
    #
    import ray

    # we probably don't need this anymore, but keep it commented in case
    # Needed to avoid thread spawning, conflicts with multi-processing.
    # You may set a number > 1 but take into account the
    # number of processes on the machine
    # torch.set_num_threads(1)

    from pydgn.experiment.util import s2c
    from pydgn.data.splitter import Splitter
    from pydgn.evaluation.grid import Grid
    from pydgn.evaluation.random_search import RandomSearch

    assert GRID_SEARCH in configs_dict or RANDOM_SEARCH in configs_dict
    search_class = Grid if GRID_SEARCH in configs_dict else RandomSearch
    search = search_class(configs_dict)

    if use_cuda:
        # Ensure a generic "cuda" device is set when using more than 1 GPU
        search.device = CUDA

    # Set the bot token, chat_id configuration
    search.telegram_config = telegram_config

    # Set the random seed
    seed = search.seed if search.seed is not None else 42
    print(f"Base seed set to {seed}.")
    experiment = search.experiment
    experiment_class = s2c(experiment)
    exp_path = os.path.join(configs_dict[RESULT_FOLDER], f"{search.exp_name}")

    os.environ[PYDGN_RAY_NUM_GPUS_PER_TASK] = str(float(gpus_per_task))

    # You can make PyDGN work on a cluster of machines!
    if os.environ.get("ip_head") is not None:
        assert os.environ.get("redis_password") is not None
        ray.init(
            address=os.environ.get("ip_head"),
            _redis_password=os.environ.get("redis_password"),
        )
        print("Connected to Ray cluster.")
        print(f"Available nodes: {ray.nodes()}")
    # Or you can work on your single server
    else:
        ray.init(num_cpus=max_cpus, num_gpus=max_gpus)
        print(f"Started local ray instance.")

    data_splits_file = configs_dict[DATA_SPLITS_FILE]

    splitter = Splitter.load(data_splits_file)
    inner_folds, outer_folds = splitter.n_inner_folds, splitter.n_outer_folds
    print(
        f"Data splits loaded, outer folds are {outer_folds} "
        f"and inner folds are {inner_folds}"
    )

    # WARNING: leave the import here, it reads env variables set before
    from pydgn.evaluation.evaluator import RiskAssesser

    risk_assesser = RiskAssesser(
        outer_folds,
        inner_folds,
        experiment_class,
        exp_path,
        data_splits_file,
        search,
        final_training_runs=configs_dict[FINAL_TRAINING_RUNS],
        higher_is_better=search.higher_results_are_better,
        gpus_per_task=gpus_per_task,
        base_seed=seed,
    )

    risk_assesser.risk_assessment(debug=debug)
    ray.shutdown()


def get_args() -> argparse.Namespace:
    """
    Processes CLI arguments (i.e., the config file location and debug option)
    and returns a namespace.

    Returns:
        a namespace with the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_FILE_CLI_ARGUMENT, dest=CONFIG_FILE)
    parser.add_argument(
        DEBUG_CLI_ARGUMENT, action="store_true", dest=DEBUG, default=False
    )
    return parser.parse_args()


def main():
    """
    Launches the experiment.
    """
    # Necessary to locate dotted paths in projects that use PyDGN
    sys.path.append(os.getcwd())

    options = get_args()
    try:
        evaluation(options)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
