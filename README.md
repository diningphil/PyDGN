<p align="center">
  <img src="https://github.com/diningphil/PyDGN/blob/master/docs/_static/pydgn-logo.png"  width="300"/>
</p>

# PyDGN: a research library for Deep Graph Networks 
[![Documentation Status](https://readthedocs.org/projects/pydgn/badge/?version=latest)](https://pydgn.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/diningphil/PyDGN/badge.svg?branch=master)](https://coveralls.io/github/diningphil/PyDGN?branch=master)

#### Read the [Documentation](https://pydgn.readthedocs.io/en/latest/index.html)

This is a Python library to easily experiment
with [Deep Graph Networks](https://www.sciencedirect.com/science/article/pii/S0893608020302197) (DGNs). It provides
automatic management of data splitting, loading and common experimental settings. It also handles both model
selection and risk assessment procedures, by trying many different configurations in parallel (CPU or GPU).

If you happen to use or modify this code, please remember to cite our tutorial paper:

[Bacciu Davide, Errica Federico, Micheli Alessio, Podda Marco: *A Gentle Introduction to Deep Learning for
Graphs*](https://www.sciencedirect.com/science/article/pii/S0893608020302197), Neural Networks, 2020.
DOI: `10.1016/j.neunet.2020.06.006`.

If you are interested in a rigorous evaluation of Deep Graph Networks, which kick-started this library, check this out:

[Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio: *A Fair Comparison of Graph Neural Networks for Graph
Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning
Representations (ICLR 2020).* [Code](https://github.com/diningphil/gnn-comparison)

## Installation:

We assume **git** and **Miniconda/Anaconda** are installed. Then you can use the script below to install `pydgn` in a controlled and separate environment (this is up to you):

    source setup/install.sh [<your_cuda_version>]
    pip install pydgn

Where `<your_cuda_version>` is an optional argument that (as of 2/3/22) can be either `cpu`, `cu102` or `cu113` for Pytorch >= 1.10.2
If you do not provide a cuda version, the script will default to `cpu`. The script will create a virtual environment
named `pydgn`, with all the required packages needed to run our code. **Important:** do NOT run this command
using `bash` instead of `source`!

## Quickstart:

#### Build dataset and data splits

    pydgn-dataset --config-file examples/DATA_CONFIGS/config_PROTEINS.yml

#### Train

    pydgn-train  --config-file examples/MODEL_CONFIGS/config_SupToyDGN.yml 

And we are up and running!

<p align="center">
  <img src="https://github.com/diningphil/PyDGN/blob/master/docs/_static/exp_gui.png"  width="600"/>
</p>

To debug your code you can add `--debug` to the command above, but the "GUI" will be disabled.

To stop the computation, use ``CTRL-C`` to send a ``SIGINT`` signal, and consider using the command ``ray stop`` to stop
all Ray processes. **Warning:** ``ray stop`` stops **all** ray processes you have launched, including those of other
experiments in progress, if any.

## Projects using PyDGN

- [Graph Mixture Density Networks](https://github.com/diningphil/graph-mixture-density-networks)
- [Contextual Graph Markov Model](https://github.com/diningphil/CGMM)
- [Extended Contextual Graph Markov Model](https://github.com/diningphil/E-CGMM)
- [Continual Learning Benchmark for Graphs](https://github.com/diningphil/continual_learning_for_graphs)


## Data Splits

We provide the data splits taken from

[Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio: *A Fair Comparison of Graph Neural Networks for Graph
Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *8th International Conference on Learning
Representations (ICLR 2020).* [Code](https://github.com/diningphil/gnn-comparison)

in the `examples/DATA_SPLITS` folder.

## License:

PyDGN >= 1.0.0 is `BSD 3-Clause` licensed, as written in the `LICENSE` file.
