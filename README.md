# PyDGN

## [Wiki](https://github.com/diningphil/PyDGN/wiki)

## Description
![](https://github.com/diningphil/PyDGN/blob/master/images/pydgn-logo.png)
This is a Python library to easily experiment with [Deep Graph Networks](https://www.sciencedirect.com/science/article/pii/S0893608020302197) (DGNs). It provides automatic management of data splitting, loading and the most common experimental settings. It also handles both model selection and risk assessment procedures, by trying many different configurations in parallel (CPU).
This repository is built upon the [Pytorch Geometric Library](https://pytorch-geometric.readthedocs.io/en/latest/), which provides support for data management.

If you happen to use or modify this code, please remember to cite our tutorial paper:

[Bacciu Davide, Errica Federico, Micheli Alessio, Podda Marco: *A Gentle Introduction to Deep Learning for Graphs*](https://www.sciencedirect.com/science/article/pii/S0893608020302197), Neural Networks, 2020. DOI: `10.1016/j.neunet.2020.06.006`.

If you are interested in a rigorous evaluation of Deep Graph Networks, check this out:

[Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio: *A Fair Comparison of Graph Neural Networks for Graph Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).* [Code](https://github.com/diningphil/gnn-comparison)

*New features*
- Support to multiprocessing in GPU is now provided via Ray (see v0.4.0)!

## Installation:
(We assume **git** and **Miniconda/Anaconda** are installed)

First, make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``. Then, ``echo $LD_LIBRARY_PATH`` should always contain ``:/home/[your user name]/miniconda3/lib``. Then run from your terminal the following command:

    source setup/install.sh [<your_cuda_version>]
    pip install pydgn

Where `<your_cuda_version>` is an optional argument that can be either `cpu`, `cu102` or `cu111` for Pytorch 1.9.0. If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment named `pydgn`, with all the required packages needed to run our code. **Important:** do NOT run this command using `bash` instead of `source`!

Remember that [PyTorch MacOS Binaries dont support CUDA, install from source if CUDA is needed](https://pytorch.org/get-started/locally/)

## Usage:

### Preprocess your dataset (see also Wiki)
    python build_dataset.py --config-file [your data config file]

#### Exampla

    python build_dataset.py --config-file DATA_CONFIGS/config_PROTEINS.yml 

### Launch an experiment in debug mode (see also Wiki)
    python launch_experiment.py --config-file [your exp. config file] --splits-folder [the splits MAIN folder] --data-splits [the splits file] --data-root [root folder of your data] --dataset-name [name of the dataset] --dataset-class [class that handles the dataset] --max-cpus [max cpu parallelism] --max-gpus [max gpu parallelism] --gpus-per-task [how many gpus to allocate for each job] --final-training-runs [how many final runs when evaluating on test. Results are averaged] --result-folder [folder where to store results]
    
#### Example (GPU required)

    python launch_experiment.py --config-file MODEL_CONFIGS/config_SupToyDGN_RandomSearch.yml --splits-folder DATA_SPLITS/CHEMICAL/ --data-splits DATA_SPLITS/CHEMICAL/PROTEINS/PROTEINS_outer10_inner1.splits --data-root DATA --dataset-name PROTEINS --dataset-class pydgn.data.dataset.TUDatasetInterface --max-cpus 1 --max-gpus 1 --final-training-runs 1 --result-folder RESULTS/DEBUG


To debug your code it is useful to add `--debug` to the command above. Notice, however, that the CLI will not work as expected here, as code will be executed sequentially. After debugging, if you need sequential execution, you can use `--max-cpus 1 --max-gpus 1 --gpus-per-task [0/1]` without the `--debug` option.  

#### Grid Search 101
Have a look at one of the config files.

#### Random Search 101
Specify a `num_samples` in the config file with the number of random trials, replace `grid`
  with `random`, and specify a sampling method for each hyper-parameter. We provide different sampling methods:
 - choice --> pick at random from a list of arguments
 - uniform --> pick uniformly from min and max arguments
 - normal --> sample from normal distribution with mean and std
 - randint --> pick at random from min and max
 - loguniform --> pick following the recprocal distribution from log_min, log_max, with a specified base

There is one config file, namely `config_SupToyDGN_RandomSearch.yml`, which you can check to see an example.

## Data Splits
We provide the data splits taken from

[Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio: *A Fair Comparison of Graph Neural Networks for Graph Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).* [Code](https://github.com/diningphil/gnn-comparison)

in the `DATA_SPLITS` folder.

## Credits:
This is a joint project with **Marco Podda** ([Github](https://github.com/marcopodda )/[Homepage](https://sites.google.com/view/marcopodda/home)), whom I thank for his relentless dedication.

Many thanks to **Antonio Carta** ([Github](https://github.com/AntonioCarta )/[Homepage](http://pages.di.unipi.it/carta)) for incorporating the Ray library (see v0.4.0) into PyDGN! This will be of tremendous help.

Many thanks to **Danilo Numeroso** ([Github](https://github.com/danilonumeroso )/[Homepage](https://pages.di.unipi.it/numeroso/)) for implementing a very flexible random search! This is a very convenient alternative to grid search.

## Contributing
**This research software is provided as-is**. We are working on this library in our spare time.

If you find a bug, please open an issue to report it, and we will do our best to solve it. For generic/technical questions, please email us rather than opening an issue.

## License:
PyDGN is GPL 3.0 licensed, as written in the LICENSE file.

## Troubleshooting
As of 15th of August 2021, there is an [issue](https://discuss.pytorch.org/t/warning-leaking-caffe2-thread-pool-after-fork-function-pthreadpool/127559/2) with Pytorch 1.9.0 which impacts the CLI.
This is why the setup script installs Pytorch 1.8.1 in the `pydgn` conda environment until Pytorch 1.10 is released (known to solve the issue).

--

If you get errors like ``/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found``:
* make sure gcc 5.2.0 is installed: ``conda install -c anaconda libgcc=5.2.0``
* ``echo $LD_LIBRARY_PATH`` should contain ``:/home/[your user name]/[your anaconda or miniconda folder name]/lib``
* after checking the above points, you can reinstall everything with pip using the ``--no-cache-dir`` option
