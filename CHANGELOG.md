
# Changelog

## [0.4.1] - TBD

### Fixed:

- memory leak when not releasing output embeddings from gpu in `engine.py`

- releasing score output from gpu in `score.py`

## [0.4.0] - 2020-10-13

#### Ray support to distributed computation!

### Added:

- [Ray](https://github.com/ray-project/ray) transparently replaces multiprocessing.
  It will help with future extensions to multi-GPU computing

- Support for parallel executions on potentially different GPUs
  (with Ray we can now allocate a predefined portion of a GPU for a task)

- Dataset and Splitter for Open Graph Benchmark graph classification datasets

### Changed:

- Modified dataset/utils.py removing the need for Path objects

- Refactored evaluation: risk assessment and model selection logic is now greatly simplified. Code is more robust and maintainable.

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

- For each config in k-fold model selection, skipping the experiment of a specific configuration that has produced a result on a specific fold

- K-Fold Assesser and K-Fold Selector can handle hold-out strategies as well. This simplifies maintenance

- The wrapper can now be customized by passing a specific engine_callback EventHandler class in the config file

- No need to specify `--debug` when using a GPU. Experiments that need to run in GPU will automatically trigger sequential execution

- Model selection is now skipped when a single configuration is given

- Added the type of experiment to the experiments folder name

- `OMP_NUM_THREADS=1` is now dynamically set inside `Launch_Experiment.py` rather by modifying the `.bashrc` file

- Simplified installation. README updated

- Final runs' outputs are now stored in different folders

- Improved splitter to allow for a simple train val test split with no shuffling. Useful to reuse holdout splits that have been already provided.

- Improved provider to use outer validation splits in run_test, if provided by the splitter (backward compatibility)

- Made Scheduler an abstract class. EpochScheduler now uses the epoch to call the step() method

### Fixed

- Removed ProgressManager refresh timer that caused non-termination of the program

- Continual experiments: `intermediate_results.csv` and `training_results.csv` are now deleted when restarting/resuming an experiment.

- Minor fix in engine about the `batch_input` field of a `State` variable

- Error when dumping a configuration file that contains a scheduler

- Error when validation was provided but early stopper was not. removed if that prevented validation and test scores from being computed

### TODOs

- Add [Ray](https://github.com/ray-project/ray) support, replacing multiprocessing. Subject to usage of Pipes as in multiprocessing.

## [0.2.0] - 2020-09-10

A series of improvements and bug fixes. We can now run link prediction experiments on a single graph. Data splits generated are incompatible with those of version 0.1.0.

### Added

- Link prediction data splitter (for single graph)
- Link prediction data provider (for single graph)

### Changed

- Improvements on progress bar manager
- Minor improvements on plotter

### Fixed

- Nothing relevant

### Additional Notes

The library now creates new processes using the `spawn` method. Spawning rather than forking prevents Pytorch from complaining (see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork and https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods). Also, there is a warning on a leaked semaphore which we will ignore for now. Finally, `spawn` will be useful to implement CUDA multiprocessing https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note. However, the Pytorch DataLoader in a child process breaks if `num_workers > 0`. Waiting for Pytorch to address this issue.



## [0.1.0] - 2020-07-14

We use PyDGN on a daily basis for our internal projects. In this first release there are some major additions to previous and unreleased version that greatly improve the user experience.

### Added
- Progress bars show the status of the experiments (with average completion time) for outer and inner cross validation (CV).
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

The library now creates new processes using the `spawn` method. Spawning rather than forking prevents Pytorch from complaining (see https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork and https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods). Also, there is a warning on a leaked semaphore which we will ignore for now. Finally, `spawn` will be useful to implement CUDA multiprocessing https://pytorch.org/docs/stable/notes/multiprocessing.html#multiprocessing-cuda-note. However, the Pytorch DataLoader in a child process breaks if `num_workers > 0`. Waiting for Pytorch to address this issue.
