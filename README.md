# PyDGN

## [Wiki](https://github.com/diningphil/PyDGN/wiki)

## Description
![](https://github.com/diningphil/PyDGN/blob/master/images/pydgn-logo.png)
This is a Python library to easily experiment with [Deep Graph Networks](https://arxiv.org/abs/1912.12693) (DGNs). It provides automatic management of data splitting, loading and the most common experimental settings. It also handles both model selection and risk assessment procedures, by trying many different configurations in parallel (CPU).
This repository is built upon the [Pytorch Geometric Library](https://pytorch-geometric.readthedocs.io/en/latest/), which provides support for data management.

If you happen to use or modify this code, please remember to cite our tutorial paper:

[Bacciu Davide, Errica Federico, Micheli Alessio, Podda Marco: *A Gentle Introduction to Deep Learning for Graphs*](https://arxiv.org/abs/1912.12693), Neural Networks, 2020. DOI: `10.1016/j.neunet.2020.06.006`.

If you are interested in a rigorous evaluation of Deep Graph Networks, check this out:

[Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio: *A Fair Comparison of Graph Neural Networks for Graph Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).* [Code](https://github.com/diningphil/gnn-comparison)

*New features*
- Support to multiprocessing in GPU is now provided via Ray (see v0.4.0)!

## Installation:
(We assume **git** and **Miniconda/Anaconda** are installed)

First, make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``. Then, ``echo $LD_LIBRARY_PATH`` should always contain ``:/home/[your user name]/miniconda3/lib``. Then run from your terminal the following command:

    source install.sh [<your_cuda_version>]

Where `<your_cuda_version>` is an optional argument that can be either `cpu`, `cu92`, `cu101`, `cu102` or `cu110` for Pytorch 1.7.0. If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment named `pydgn`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!

Remember that [PyTorch MacOS Binaries dont support CUDA, install from source if CUDA is needed](https://pytorch.org/get-started/locally/)

## Usage:

### Preprocess your dataset (see also Wiki)
    python build_dataset.py --config-file [your data config file]

### Launch an experiment in debug mode (see also Wiki)
    python launch_experiment.py --config-file [your exp. config file] --splits-folder [the splits MAIN folder] --data-splits [the splits file] --data-root [root folder of your data] --dataset-name [name of the dataset] --dataset-class [class that handles the dataset] --max-cpus [max cpu parallelism] --max-gpus [max gpu parallelism] --gpus-per-task [how many gpus to allocate for each job] --final-training-runs [how many final runs when evaluating on test. Results are averaged] --result-folder [folder where to store results]

To debug your code it is useful to add `--debug` to the command above. Notice, however, that the CLI will not work as expected here, as code will be executed sequentially. After debugging, if you need sequential execution, you can use `--max-cpus 1 --max-gpus 1 --gpus-per-task [0/1]` without the `--debug` option.  

## Credits:
This is a joint project with **Marco Podda** ([Github](https://github.com/marcopodda)/[Homepage](https://sites.google.com/view/marcopodda/home)), whom I thank for his relentless dedication.

Many thanks to **Antonio Carta** ([Github](https://github.com/AntonioCarta)/[Homepage](http://pages.di.unipi.it/carta)) for incorporating the Ray library (see v0.4.0) into PyDGN! This will be of tremendous help.

## Contributing
**This research software is provided as-is**. We are working on this library in our spare time.

If you find a bug, please open an issue to report it, and we will do our best to solve it. For generic/technical questions, please email us rather than opening an issue.

## License:
PyDGN is GPL 3.0 licensed, as written in the LICENSE file.

## Troubleshooting

If you get errors like ``/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found``:
* make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``
* ``echo $LD_LIBRARY_PATH`` should contain ``:/home/[your user name]/[your anaconda or miniconda folder name]/lib``
* after checking the above points, you can reinstall everything with pip using the ``--no-cache-dir`` option
