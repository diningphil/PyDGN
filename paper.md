---
title: 'PyDGN: a Python Library for Flexible and Reproducible Research on Deep Learning for Graphs'
tags:
  - Python
  - Machine Learning
  - Graph Networks
  - Deep Learning for Graphs
authors:
  - name: Federico Errica
    orcid: 0000-0001-5181-2904
    corresponding: true
    affiliation: 1
  - name: Davide Bacciu
    orcid: 0000-0001-5213-2468
    affiliation: 2
  - name: Alessio Micheli
    orcid: 0000-0001-5764-5238
    affiliation: 2
affiliations:
 - name: NEC Laboratories Europe, Germany
   index: 1
 - name: University of Pisa, Italy
   index: 2
date: 25 June 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The disregard for standardized evaluation procedures is a longstanding issue in the Machine Learning (ML) community that severely hampers real progress. This is especially true for fast-growing research areas, where a substantial amount of literature relentlessly appears every day. In the graph machine learning field, these issues have already been brought to light and partially addressed, but a general-purpose library for rigorous evaluations and reproducible experiments is lacking in the graph machine learning landscape. We therefore introduce a new Python library, called `PyDGN`, to provide users with a system that lets them focus on models' development while ensuring empirical rigor and reproducibility of their results.

# Statement of need

To date, the graph ML community [@sperduti_supervised_1997; @scarselli_graph_2009; @micheli_neural_2009; @bronstein_geometric_2017; @hamilton_representation_2017; @wu_comprehensive_2020] has already developed benchmarking software to re-evaluate existing models on a fixed set of datasets [@shchur_pitfalls_2018; @errica_fair_2020; @hu_open_2020; @liu_dig_2021]. In addition, existing libraries such as Pytorch Geometric (PyG) [@fey_fast_2019], Deep Graph Library (DGL) [@wang_dgl_2019], and Spektral [@grattarola_graph_2021] provide the building blocks of Deep Graph Networks (DGNs) [@bacciu_gentle_2020], also known as message-passing architectures [@gilmer_neural_2017], effectively acting as the backbone of most graph ML software packages.

At the same time, the community lacks a software library that is specifically dedicated to ensuring reproducibility and replicability of experiments without compromising the flexibility required by our everyday research. To fill this gap, we have developed `PyDGN`, a Python library whose aim is not to cover all possible implementations of Deep Graph Networks (DGNs) but rather to enable rigorous and reproducible evaluations without compromising the users' flexibility in developing research prototypes. `PyDGN` builds upon `Pytorch` [@paszke_pytorch_2019] and `Pytorch Geometric` (PyG) [@fey_fast_2019] to handle graph-structured data and reuse efficient implementations of machine learning models. It exploits [Ray](https://www.ray.io/) to run experiments in parallel (also on clusters of machines) and it supports [GPU](https://developer.nvidia.com/cuda-toolkit) computation for faster executions. Our goal is to help practitioners and researchers to easily focus on the development of their models and to effortlessly evaluate them under fair, robust, and reproducible experimental conditions, thus mitigating empirical malpractices that often occur in the Machine Learning (ML) community [@lipton_troubling_2018; @shchur_pitfalls_2018; @errica_fair_2020]. `PyDGN` has already been used in a number of research projects that have been published at top-tier venues, as listed in the [official GitHub repository](https://github.com/diningphil/PyDGN).

![PyDGN is logically organized into different modules that cover specific aspects of the entire evaluation's pipeline, from data creation to a model's risk assessment.\label{fig:pydgn-structure}](paper.png){ width=100% }

We refer the reader to Figure \autoref{fig:pydgn-structure} for a visual depiction of the main components. We remark that all modules, with the exception of the one responsible for evaluation (due to its standard behavior), are readily extensible and promote rapid prototyping through code reuse.

## How to use it
Users can easily prepare and launch their evaluations through *configuration files*, one for the data preparation and the second for the actual experiment. In the former, the user specifies: **i)** how to split the data; **ii)** the dataset to use; and **iii)** optional data transformations. In the second file, the user indicates: **i)** data and splits; **ii)** hardware devices and parallelism; **iii)** hyper-parameter configurations for model selection; **iv)** training-specific details like metrics and optimizer. Dedicated scripts prepare the data, its splits, launch the experiments and compute results. The [PyDGN documentation](https://pydgn.readthedocs.io/) helps the user understand the main mechanisms through tutorials and examples.

### Data preparation
A recurrent issue in the evaluation of ML models is that they are compared using different data splits. The first step to reproducibility is thus the creation and retention of such splits: we provide code that partitions the data depending on the required node/graph/link prediction scenarios. Since splitting depends on the type of evaluation procedure, we cover \textit{hold-out}, \textit{k-fold}, and \textit{nested/double} \textit{k-fold} cross validation, which are the most common evaluation scenarios in the ML literature.

To create and use a dataset, we provide an interface that allows users to easily specify pre-processing as well as runtime processing of graphs. We extend the available dataset classes in `PyG` to achieve this goal. In addition, a (extensible) data provider automatically retrieves the correct subset of data (training/validation/test) during an experiment run, making sure that the user does not involuntarily leak test data for training/validation.

### Evaluation procedures
`PyDGN` is equipped with a series of routines that take off the burden of performing model selection and risk assessment from the users. This reduces chances of empirical flaws and favors fair comparisons. After model selection,  the best configuration, obtained with respect to the validation set, is re-trained and evaluated on the test set. In addition, a start-and-stop mechanism can resume execution of unfinished experiments when the whole evaluation is interrupted; if model checkpoints are enabled, `PyDGN` resumes every individual run from the very last training epoch.

### Experiment templates
Each experiment has to implement a specific interface consisting of two methods. The first is called during the model selection, and the second is called during risk assessment of the model. Our library ships with two standard experiments, an end-to-end training on a single task and two-step training where first we compute unsupervised node/graph embeddings and then apply a supervised predictor on top of them to solve a downstream task. These two implementations cover most use cases and ensure that the different data splits are used in the correct way.

### Implementing models
To implement a DGN in our library, it is sufficient to adhere to a very simple interface that specifies initialization arguments and the type of output for the prediction step. It is enough to wrap the interface around a `PyG` model to have it immediately working. The library provides the current hyper-parameter configuration to be evaluated to the model in the form of a dictionary, generated by looking at the configuration file.

### Training via publish-subscribe
The training engine implements all boilerplate code regarding the training loop. We apply the publish-subscribe design pattern to trigger the execution of callbacks at specific points in the training procedure. Every metric, early stopping, scheduler, gradient clipper, optimizer, data fetcher, and stats plotter implements some of these callbacks; when a callback is triggered, a shared state object is passed as argument to allow communication between the different components of the training process.

# Acknowledgements

We acknowledge contributions from Antonio Carta, Danilo Numeroso, Daniele Castellana, Alessio Gravina, Francesco Landolfi, and support from Marco Podda during the genesis of this project.

# References
