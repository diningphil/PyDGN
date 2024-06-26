<p align="center">
  <img src="https://github.com/diningphil/PyDGN/blob/main/docs/_static/pydgn-logo.png"  width="300"/>
</p>

# PyDGN: a research library for Deep Graph Networks 
[![License](https://img.shields.io/badge/License-BSD_3--Clause-gray.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/pydgn/badge/?version=latest)](https://pydgn.readthedocs.io/en/latest/?badge=latest)
[![Python Package](https://github.com/diningphil/PyDGN/actions/workflows/python-publish.yml/badge.svg)](https://github.com/diningphil/PyDGN/actions/workflows/python-publish.yml)
[![Downloads](https://static.pepy.tech/personalized-badge/pydgn?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/pydgn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Interrogate](https://github.com/diningphil/PyDGN/blob/main/.badges/interrogate_badge.svg)](https://interrogate.readthedocs.io/en/latest/)
[![Coverage](https://github.com/diningphil/PyDGN/blob/main/.badges/coverage_badge.svg)]()
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05713/status.svg)](https://doi.org/10.21105/joss.05713)

## [Documentation](https://pydgn.readthedocs.io/en/latest/index.html)

This is a Python library to easily experiment
with [Deep Graph Networks](https://www.sciencedirect.com/science/article/pii/S0893608020302197) (DGNs). It provides
automatic management of data splitting, loading and common experimental settings. It also handles both model
selection and risk assessment procedures, by trying many different configurations in parallel (CPU or GPU).

## Citing this work

If you used this library for your project, please consider citing us:

    @article{pydgn,
      author = {Errica, Federico and Bacciu, Davide and Micheli, Alessio},
      doi = {10.21105/joss.05713},
      journal = {Journal of Open Source Software},
      month = oct,
      number = {90},
      pages = {5713},
      title = {{PyDGN: a Python Library for Flexible and Reproducible Research on Deep Learning for Graphs}},
      url = {https://joss.theoj.org/papers/10.21105/joss.05713},
      volume = {8},
      year = {2023}
    }

## Installation:

Automated tests passing on Windows, Linux, and MacOS. Requires at least Python 3.8.
Simply run
    
    pip install pydgn

## Quickstart:

#### Build dataset and data splits

    pydgn-dataset --config-file examples/DATA_CONFIGS/config_NCI1.yml

#### Train

    pydgn-train  --config-file examples/MODEL_CONFIGS/config_SupToyDGN.yml 

And we are up and running!

<p align="center">
  <img src="https://github.com/diningphil/PyDGN/blob/main/docs/_static/exp_gui.png"  width="600"/>
</p>

To debug your code you can add `--debug` to the command above, but the "GUI" will be disabled.

To stop the computation, use ``CTRL-C`` to send a ``SIGINT`` signal, and consider using the command ``ray stop`` to stop
all Ray processes. **Warning:** ``ray stop`` stops **all** ray processes you have launched, including those of other
experiments in progress, if any.

### Using the Trained Models

It's very easy to load the model from the experiments (see also the [Tutorial](https://pydgn.readthedocs.io/en/latest/tutorial.html)):

    from pydgn.evaluation.util import *

    config = retrieve_best_configuration('RESULTS/supervised_grid_search_toy_NCI1/MODEL_ASSESSMENT/OUTER_FOLD_1/MODEL_SELECTION/')
    splits_filepath = 'examples/DATA_SPLITS/CHEMICAL/NCI1/NCI1_outer10_inner1.splits'
    device = 'cpu'

    # instantiate dataset
    dataset = instantiate_dataset_from_config(config)

    # instantiate model
    model = instantiate_model_from_config(config, dataset, config_type="supervised_config")

    # load model's checkpoint, assuming the best configuration has been loaded
    checkpoint_location = 'RESULTS/supervised_grid_search_toy_NCI1/MODEL_ASSESSMENT/OUTER_FOLD_1/final_run1/best_checkpoint.pth'
    load_checkpoint(checkpoint_location, model, device=device)

    # you can now call the forward method of your model
    y, embeddings = model(dataset[0])


## Projects using PyDGN

- [Infinite Contextual Graph Markov Model (ICML 2022)](https://github.com/diningphil/iCGMM)
- [Graph Mixture Density Networks (ICML 2021)](https://github.com/diningphil/graph-mixture-density-networks)
- [Contextual Graph Markov Model (ICML 2018, JMLR 2020)](https://github.com/diningphil/CGMM)
- [Extended Contextual Graph Markov Model (IJCNN 2021)](https://github.com/diningphil/E-CGMM)
- [Continual Learning Benchmark for Graphs (WWW Workshop 2021, spotlight)](https://github.com/diningphil/continual_learning_for_graphs)


## Data Splits

We provide the data splits taken from

[Errica Federico, Podda Marco, Bacciu Davide, Micheli Alessio: *A Fair Comparison of Graph Neural Networks for Graph
Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *8th International Conference on Learning
Representations (ICLR 2020).* [Code](https://github.com/diningphil/gnn-comparison)

in the `examples/DATA_SPLITS` folder.

## License:

PyDGN >= 1.0.0 is `BSD 3-Clause` licensed, as written in the `LICENSE` file.
