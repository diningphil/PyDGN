# Changelog

## [1.0.0] PyDGN with Documentation

### TODO (decreasing priority):

- Add unit test
- Add integration tests
- Add again temporal PyDGN (with proper documentation)
- Add shared dataset functionality to save main memory when parallelizing

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

## [0.7.3]

### Changed

- The setup installation files now work with Pytorch 1.10 and Pytorch Geometric 2.0.3, and the library assumes Python >= 3.8

## [0.7.2] 


### Fixed
- Fixed minor bug in experiment. The function create_unsupervised_model looked for supervised_config, rather than unsupervised_config, when looking for the predictor
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
- Fix in graph predictor: forgot to pass arguments to super when inheriting from `GraphPredictor`

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
