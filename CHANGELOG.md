
# Changelog

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
