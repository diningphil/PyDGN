# PyDGN

## [Wiki](https://github.com/diningphil/PyDGN/wiki)

## Description
![](https://github.com/diningphil/PyDGN/blob/master/images/pydgn-logo.png)
This is a Python library to easily experiment with [Deep Graph Networks](https://arxiv.org/abs/1912.12693) (DGNs). It provides automatic management of data splitting, loading and the most common experimental settings. It also handles both model selection and risk assessment procedures, by trying many different configurations in parallel (CPU).
This repository is built upon the [Pytorch Geometric Library](https://pytorch-geometric.readthedocs.io/en/latest/), which provides support for data management.

**If you happen to use or modify this code, please remember to cite our tutorial paper**:

[Bacciu Davide, Errica Federico, Micheli Alessio, Podda Marco: *A Gentle Introduction to Deep Learning for Graphs*](https://arxiv.org/abs/1912.12693), Neural Networks, 2020. DOI: `10.1016/j.neunet.2020.06.006`.

If you are interested in a rigorous evaluation of Deep Graph Networks, check this out:

[Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio: *A Fair Comparison of Graph Neural Networks for Graph Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).* [Code](https://github.com/diningphil/gnn-comparison)

*Missing features*
- Support to multiprocessing in GPU is not provided yet, but single GPU support is enabled.

## Installation:
(We assume **git** and **Miniconda/Anaconda** are installed)

#### PyTorch (CPU version) 

    source setup/install_cpu.sh

#### PyTorch (CUDA version 10.1) 

    source setup/install_cuda.sh
     
Remember that [PyTorch MacOS Binaries dont support CUDA, install from source if CUDA is needed](https://pytorch.org/get-started/locally/)

## Usage:

### Preprocess your dataset (see also Wiki)
    python PrepareDatasets.py --config-file [your config file]

### Launch an experiment in debug mode (see also Wiki)
    python Launch_Experiments.py --config-file [your config file] --data-splits [the splits file] --data-root [root folder of your data] --dataset-name [name of the dataset] --dataset-class [class that handles the dataset] --outer-processes [number of outer folds to process in parallel] --inner-processes [number of configurations to run in parallel for each outer fold] --final-training-runs [how many final runs when evaluating on test. Results are averaged] --result-folder [folder where to store results]
    
To debug your code it is useful to add `--debug` to the command above. Notice, however, that the CLI will not work as expected here, as code will be executed sequentially. After debugging, if you need sequential execution (e.g. because you run the experiments on single GPU) use `--inner-processes 1 --outer-processes 1` without the `--debug` option.  

## Credits:
This is a joint project with **Marco Podda** ([Github](https://github.com/marcopodda)/[Homepage](https://sites.google.com/view/marcopodda/home)), whom I thank for his relentless dedication.

## Contributing
**This research software is provided as-is**. We are working on this library in our spare time. 

If you find a bug, please open an issue to report it, and we will do our best to solve it. For generic/technical questions, please email us rather than opening an issue.

## License:
PyDGN is GPL 3.0 licensed, as written in the LICENSE file.

## Troubleshooting

If you get errors like ``/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found``:
* make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``
* ``echo $LD_LIBRARY_PATH`` should contain ``:/home/[your user name]/miniconda3/lib``
* after checking the above points, you can reinstall everything with pip using the ``--no-cache-dir`` option
