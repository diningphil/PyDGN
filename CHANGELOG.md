# Changelog

## [1.3.1] Weighted Loss implementation

### Added

- You can specify weights for loss in `AdditiveLoss` by passing a dictionary of (loss name, loss weight) entries as an argument.
  See the documentation of `AdditiveLoss` for more info or the example in `examples/MODEL_CONFIGS/config_SupToyDGN.yml`.

### Fixed

- Better handling of `len()` in `TUDatasetInterface`

### Changed

- Package requirements now specify an upper bound on some packages, including Pytorch and PyG to ensure compatibility. Better be safe than sorry :)


## [1.3.0] Support for Pytorch 1.13.0, CUDA 11.6, CUDA 11.7, PyG 2.1.0, Ray 2.1.0, support + minor fixes

### Changed

- Updates to tests to make the fake datasets compatible with PyG 2.1.0

### Fixed

- IDLE ray workers not deallocating GPUs

- Now we sort the data list returned by training engine as if samples were not shuffled.

  Meaning the returned data list is consistent with the original ordering of the dataset.

#### Comments

We tried to provide support for creating an environment with PyG 2.2.0, but importing the library seems to cause
`segmentation fault` in certain cases. Therefore, we will wait until the issue is fixed and then update the script.


## [1.2.6] Minor changes

### Added

- You can now specify a specific subset of gpus to use in the configuration file.
  
  Just add the optional field `gpus_subset: 1,2,3` if you want to only use GPUs with index 1,2 and 3.


## [1.2.5] Reverting to previous Ray version

### Changed

- Ray 2.0.0 seems to have a problem with killing `IDLE` processes and releasing their resources, i.e. OOM on GPU. 
  We are reverting to a version that we were using before and did not have this problem.


## [1.2.4] Minor fixes + tests for main functionalities

### Fixed

- Minor check in splitter
- Minor fix in link prediction splitter, one evaluation link was being left out
- Minor fix in early stopper, `epoch_results` dict was overwritten after applying early stopping. Does not affect performances since the field is re-initialized the subsequent epoch by the training engine.
- Removed setting random seed for map-style dataset. It was not useful (see Torch doc on reproducibility) and could cause transforms based on random sampling (e.g. negative sampling) to behave always in the same way

### Changed

- Changed semantics of gradient clipper, as there are not many alternatives out there

## [1.2.3] Added support for single graph tasks

At the moment, the entire graph must fit in CPU/GPU memory. `DataProvider` extensions to partition the graph using PyG should not be difficult.

### Added

- New splitter, `SingleGraphSplitter`, which randomly splits nodes in a single graph (with optional stratification)
- New provider, `SingleGraphDataProvider`, which adds mask fields to the single DataBatch object (representing the graph)

### Changed

- renamed method `get_graph_targets` of `Splitter` to `get_targets`, and modified it to make it more general


## [1.2.2] Telegram Bot Support!

### Added

- Telegram bot support. Just specify a telegram configuration in a YAML file and let the framework do the rest! Just remember not to push your telegram config file in your repo!

## [1.2.1] Splitter Fix after 1.2.0 changes, deprecating 1.2.0

### Fixed

- the changed introduced in splitter causes seed to be resetted when splits were loaded during each experiment. Now it has been fixed by setting the seed only when split() is called.
- minor in num_features of OGBGDatasetInterface

## [1.2.0] Simplified Metric usage

### Changed

- Simplified metrics to either take the mean score over batches or to compute epoch-wise scores (default behavior).
  In the former case, the result may be affected by batch size, especially in cases like micro-AP and similar scores. 
  Use it only in case it is too expensive (RAM/GPU memory) to compute the scores in a single shot.

### Fixed

- Bug in splitter, the seed was not set properly and different executions led to different results. This is not a problem whenever the splits are publicly released after the experiments (which is always the case).
- Minor in data loader workers for iterable datasets

## [1.1.0] Temporal PyDGN for single graph sequences

### Added

- Temporal learning routines (with documentation), works with single graphs sequences
- Template to show how we can use PyDGN on a cluster (see `cluster_slurm_example.sh`) - launch using `sbatch cluster_slurm_example.sh`. **Disclaimer**: you must have experience with slurm, the script is not working out of the box and settings must be adjusted to your system.

### Fixed

- removed method from `OGBGDatasetInterface` that broke the data split generation phase.
- added `**kwargs` to all datasets

### Changed

- Extended behavior of ``TrainingEngine`` to allow for null target values and some temporal bookkeeping (allows a lot of code reuse). 
- Now ``batch_loss`` and ``batch_score`` in the ``State`` object are initialized to ``None`` before training/evaluation of a new batch starts. This could have been a problem in the temporal setting, where we want to accumulate results for different snapshots.

## [1.0.9] Iterable Dataset implementation for large datasets stored on disk in chunks of files

We provide an implementation of iterable-style datasets, where the dataset usually doesn't fit into main memory and
it is stored into different files on disk. If you don't overwrite the ``__iter__`` function, we assume to perform data splitting at
file level, rather than sample level. Each file can in fact contain a list of ``Data`` objects, which will be streamed
sequentially. Variations are possible, depending on your application, but you can use this new dataset class as a good starting point.
If you do, be careful to test it together with the iterable versions of the data provider, engine, and engine callback.

### Added

- Implemented an Iterable Dataset inspired by the [WebDataset](https://github.com/webdataset/webdataset) interface
- Similarly, added ``DataProvider``, ``Engine`` and ``EngineCallback`` classes for the Iterable-style datasets.

### Changed

- Now we can pass additional arguments at runtime to the dataset

## [1.0.8] Minor changes

### Changed

- (Needs simple test) Setting ``CUDA_VISIBLE_DEVICES`` variable before cuda is initialized, so that in ``--debug`` mode we can use the GPU with the least amount of used memory.
- Commented a couple of lines which forces OMP_NUM_THREADS to 1 and Pytorch threads to 1 as well. It seems we don't need them anymore.

## [1.0.7] Minor fix

### Fixed

- to comply with `TUDataset`, we do not override the method `__len__` anymore

## [1.0.5] Minor fix

### Fixed

- `load_dataset` does not assume anymore that there exists a `processed` data folder, but it is backward-compatible with previous versions.
- fixed an indexing bug on target data for node classification experiments (caused program to crash)
- Metric: renamed `_handle_reduction` to `_expand_reduction` and created a new helper routine `_update_num_samples` to allow a user to decide how to compute and average scores.

## [1.0.4] Minor fix

### Fixed

- use of fractions of GPUs for a single task
- changed signature in forward to allow a dictionary (for MultiScore) or a value (for basic metrics)
- added squeeze in MulticlassAccuracy when target tensor has shape (?, 1)

## [1.0.3] Minor fix

### Fixed

- Same bug as before but for `pydgn-dataset` =).

## [1.0.2] Minor fix

### Fixed

- Bug that prevented locating classes via dotted paths in external projects.

## [1.0.0] PyDGN with Documentation

### Added

- A documentation (it was about time!!!)
- Possibility of specifying inner and outer validation ratio
- We can now use a specific data loader and specify its arguments in the configuration file
- We can now force a metric to compute node-based or graph-based metrics, rather than looking at the ground truth's shape.
- Possibility of evaluating on validation (and test) every `n` epochs
- Use entrypoints to simplify usage of the library
- All arguments must be now specified in the config file. There is a template one can use in the doc.

### Changed (IMPORTANT!)

- Removed any backward compatibilities with very old versions (<=0.6.2)
- Substituted Loss and Score classes with a single Metric class, to avoid code redundancy
- Pre-computed random outer validation splits (extracted from 10% of outer training set) for data splits from "A Fair Comparison on Graph Neural Networks for Graph Classification". Note that this does not impact the test splits.

## [0.7.3]

### Changed

- The setup installation files now work with Pytorch 1.10 and Pytorch Geometric 2.0.3, and the library assumes Python >= 3.8

## [0.7.2] 


### Fixed
- Fixed minor bug in experiment. The function create_unsupervised_model looked for supervised_config, rather than unsupervised_config, when looking for the readout
- Feature request: loss, score, and additiveloss now take a parameter `use_nodes_batch_size` to force computation w.r.t. input nodes rather than target dimension (the default)

## [0.7.1] 

### Added

- Minor refactoring of the engines to avoid redundant flow of information

### Fixed
- Fixed a bug in EventHandler. If one extends EventHandler with new events, which are triggered by a training engine, make sure that callbacks that implement the EventHandler interface do not break when the new events are triggered.
- Refactored Profiler to abstract from the EventHandler. This created problems when a callback implmenented an interface that extends EventHandler. If the callback does not implement a particular method, nothing happens and the dispatcher moves on.

## [0.7.0] - PyDGN temporal (with [Alessio Gravina](http://pages.di.unipi.it/gravina/) based on [Pytorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html)) + minor fixes

### Added

- PyDGN Temporal: Support for `single graph sequence` tasks, the most common use case at the moment
  (tested for supervised experiments only)

### Fixed

- Minor in `cgmm_incremental` experiment
- loss/score now considers the case of reduction=mean (default) and sum when computing the epoch's loss/score

## [0.6.1] - Stop and Resume from different devices

### Fixed

- When using checkpoints, we can now switch devices without getting a deserialization error

## [0.6.0] - Evaluator's Refactoring

### Added

- Evaluator now stores and displays values for the loss used. Also, validation of final runs is kept.

### Modified

- Heavy refactoring of the evaluator


### Fixed

- Removed old code in `TUDatasetInterface`
- Epochs' log starts from 1 :D

## [0.5.1] - Code for E-CGMM + Minor fixes

### Added

- Code for E-CGMM ("Modeling Edge Features with Deep Bayesian Graph Networks", IJCNN 2021)
- ConstantEdgeIfEmpty transform (used by e.g., E-CGMM)

### Fixed

- Changed name from `transforms` to `transform` for data preprocessing config files (backward compatible)
- Minor fix when handling edge data with incremental models like E-CGMM
- Fix in graph readout: forgot to pass arguments to super when inheriting from `GraphReadout`

## [0.5.0] - Pytorch 1.8.1, random search, and many fixes

### Added

- Data splits from "A fair comparison of graph neural networks for graph classification", ICLR 2021.

- Replaced strings with macros, to improve maintainability

- Support to replicability with seeds. Debug CPU mode and parallel CPU mode can reproduce the same results. With CUDA
  things change due to DataLoader implementation. Even by running the whole exp again, some runs in GPU differ slightly.

- Added current_set to Score and MultiScore, to keep track of whether the set under consideration is TRAINING,
  VALIDATION or TEST

- Random Search support: specify a `num_samples` in the config file with the number of random trials, replace `grid`
  with `random`, and specify a sampling method for each hyper-parameter. We provide different sampling methods:
    - choice --> pick at random from a list of arguments
    - uniform --> pick uniformly from min and max arguments
    - normal --> sample from normal distribution with mean and std
    - randint --> pick at random from min and max
    - loguniform --> pick following the recprocal distribution from log_min, log_max, with a specified base

- Implemented a 2-way training scheme for CGMM and variants, which allows to first compute and store the graph
  embeddings, and then load them from disk to solve classification tasks. Very fast and lightweight, allowing to easily
  try 1K configuration for each outer fold.

- Early stopping can now work with loss functions rather than scores (but a score must be provided nonetheless)

### Changed

- Minor improvements in result files

- Debug mode now prints output to the console

- Refactored engine for Link Prediction, by subclassing the TrainingEngine class

- Added chance to mini-batch edges (but not nodes) in single graph link prediction, to reduce the computational burden

- Compute statistics in iocgmm.py: removed 1 - from bottom computation, because it assigned 1 to nodes with degree 0

### Fixed:

- ProgressManager can load the elapsed time of finished experiments, in case you stop and resume the entire process

- Fix for semi-supervised graph regression in `training/util.py` (added an `.unsqueeze(0)` on the `y` variable for graph
  prediction tasks)

- Last config of model selection was not checked in debug mode

- Moving model to device inside experiments, before the model is passed to the optimizer. This solves optimizer
  initialization problems e.g., with Adagrad

- Minor fix in AdditiveLoss

- Minor fix in Progress Manager due to Ray upgrade in PyDGN 0.4.0

- Refactored both CGMM and its incremental training strategy

- Improved evaluator code to define just once remote ray functions

- Jupyter notebook for backward compatibility with our ICLR 2020 data splits

- Backward compatibility in Splitter that handles missing validation fields in old splits.

- Using data root provided through cli rather than the value stored in dataset_kwargs.pt by data preprocessing
  operations. This is because data location may have changed.

- Removed load_splitter from utils, which assumed a certain shape of the splits filename. Now we pass the filepath to
  the data provider.

- minor fix in AdditiveLoss and MultiScore

- MultiScore now does not make any assumption on the underlying scores. Plus, it is easier to maintain.

- CGMM (and variants) mini batch computation (with full batch training) now produces the same loss/scores as full batch
  computation

- minor fix in evaluator.py

- memory leak when not releasing output embeddings from gpu in `engine.py`

- releasing score output from gpu in `score.py`

## [0.4.0] - 2020-10-13

#### Ray support to distributed computation!

### Added:

- [Ray](https://github.com/ray-project/ray) transparently replaces multiprocessing. It will help with future extensions
  to multi-GPU computing

- Support for parallel executions on potentially different GPUs
  (with Ray we can now allocate a predefined portion of a GPU for a task)

- Dataset and Splitter for Open Graph Benchmark graph classification datasets

### Changed:

- Modified dataset/utils.py removing the need for Path objects

- Refactored evaluation: risk assessment and model selection logic is now greatly simplified. Code is more robust and
  maintainable.

- Print indented config

- Moved s2c to evaluation/utils.py

- Improved LaunchExperiment

- Config files should now specify the complete path to a specific experiment class

- Renamed files and folders to follow Python convention

### Fixed:

- bug fix when extending the list of final run jobs. We need to add to waiting variable the lastly scheduled jobs only.

- bug fix in evaluator when using ray

- Engine now saves (EngineCallback) and restores `stop_training` in checkpoint

## [0.3.2] - 2020-10-4

### Added

- Implemented a Multi Loss to combine different loss functions while plotting the individual components as well

- Added standard deviation when multiple final runs are used

### Changed

- Removed redundant files and refactored a bit to simplify folder structures

- Removed utils folder, moved the few methods in the other folders where they were needed

- Skipping those final runs that are already completed!

- For each config in k-fold model selection, skipping the experiment of a specific configuration that has produced a
  result on a specific fold

- K-Fold Assesser and K-Fold Selector can handle hold-out strategies as well. This simplifies maintenance

- The wrapper can now be customized by passing a specific engine_callback EventHandler class in the config file

- No need to specify `--debug` when using a GPU. Experiments that need to run in GPU will automatically trigger
  sequential execution

- Model selection is now skipped when a single configuration is given

- Added the type of experiment to the experiments folder name

- `OMP_NUM_THREADS=1` is now dynamically set inside `Launch_Experiment.py` rather by modifying the `.bashrc` file

- Simplified installation. README updated

- Final runs' outputs are now stored in different folders

- Improved splitter to allow for a simple train val test split with no shuffling. Useful to reuse holdout splits that
  have been already provided.

- Improved provider to use outer validation splits in run_test, if provided by the splitter (backward compatibility)

- Made Scheduler an abstract class. EpochScheduler now uses the epoch to call the step() method

### Fixed

- Removed ProgressManager refresh timer that caused non-termination of the program

- Continual experiments: `intermediate_results.csv` and `training_results.csv` are now deleted when restarting/resuming
  an experiment.

- Minor fix in engine about the `batch_input` field of a `State` variable

- Error when dumping a configuration file that contains a scheduler

- Error when validation was provided but early stopper was not. removed if that prevented validation and test scores
  from being computed

## [0.2.0] - 2020-09-10

A series of improvements and bug fixes. We can now run link prediction experiments on a single graph. Data splits
generated are incompatible with those of version 0.1.0.

### Added

- Link prediction data splitter (for single graph)
- Link prediction data provider (for single graph)

### Changed

- Improvements on progress bar manager
- Minor improvements on plotter

### Fixed

- Nothing relevant

### Additional Notes

The library now creates new processes using the `spawn` method. Spawning rather than forking prevents Pytorch from
complaining (see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
and https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods). Also, there is a warning on a
leaked semaphore which we will ignore for now. Finally, `spawn` will be useful to implement CUDA
multiprocessing https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note. However, the
Pytorch DataLoader in a child process breaks if `num_workers > 0`. Waiting for Pytorch to address this issue.

## [0.1.0] - 2020-07-14

We use PyDGN on a daily basis for our internal projects. In this first release there are some major additions to
previous and unreleased version that greatly improve the user experience.

### Added

- Progress bars show the status of the experiments (with average completion time) for outer and inner cross validation (
  CV).
  ![](https://github.com/diningphil/PyDGN/blob/master/images/progress.png)
- Tensorboard visualization is activated by default when using a Plotter.
- A new profiler keeps track of the time spent on each event (see Event engine).
  ![](https://github.com/diningphil/PyDGN/blob/master/images/profiler.png)
- All experiments can be interrupted at any time and resumed gracefully (the engine looks for the last checkpoint).

### Changed

- Removed all models that are not necessary to try and test the library.

### Fixed

- Various multiprocessing issues caused by using the `fork` method with Pytorch.

### Additional Notes

The library now creates new processes using the `spawn` method. Spawning rather than forking prevents Pytorch from
complaining (see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
and https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods). Also, there is a warning on a
leaked semaphore which we will ignore for now. Finally, `spawn` will be useful to implement CUDA
multiprocessing https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note. However, the
Pytorch DataLoader in a child process breaks if `num_workers > 0`. Waiting for Pytorch to address this issue.
