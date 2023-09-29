Introduction
============

**PyDGN** (*Python for Deep Graph Networks*) is a framework that you can use to easily train and evaluate `Deep Graph Networks <https://www.sciencedirect.com/science/article/abs/pii/S0893608020302197>`_.

The concept was born in 2019, and it had the following goals in mind:
 * automatize **model selection** and **model assessment** procedures (see `our ICLR 2020 paper <https://arxiv.org/abs/1912.09893>`_ for an introductory explanation of these terms),
 * foster reproducibility and robustness of results,
 * reduce the amount of boilerplate code to write,
 * make it flexible enough to encompass a wide range of use cases.
 * support a number of different hardware set ups, including a cluster of nodes (using `Ray <https://docs.ray.io/en/latest/>`_),

To run an experiment, you usually rely on 2 **YAML configuration files**:
  * one to pre-process the dataset and create the data splits,
  * another with information about the experiment itself and the hyper-parameters to try.

We already provide support to easily run some of the most common experiments:
  * Supervised tasks, like node and graph predictions
  * Semi-supervised tasks
  * Link prediction tasks

Generally speaking, we rely on the data format defined in `PyG <https://pytorch-geometric.readthedocs.io/en/latest/>`_ to process the graphs.

To this day, we have already used older **PyDGN** versions far beyond the simple examples that are shown here. One can devise `incremental training <https://github.com/diningphil/CGMM>`_ and `continual learning <https://github.com/diningphil/continual_learning_for_graphs>`_  experiments, and we are planning to properly extend the library to
**temporal** graph learning as well.

**DISCLAIMER:** while we are trying hard to make this library as user-friendly as possible, in the end it is a library for *research*.
Depending on what strange experiments you will need to perform, it is possible that you will have to subclass and implement your own modules.
For instance, subclassing an :class:`~pydgn.training.event.handler.EventHandler` requires you to know a bit about how the :class:`~pydgn.training.engine.TrainingEngine` works and when specific events are triggered.
However, we believe reading boilerplate code is far easier than writing your own, and it reduces the risks of doing something wrong (and if you find a bug please tell us!)


Installation (Linux only):
*******************

The recommended way to install the library is to follow the steps to install ``torch`` and ``torch_geometric`` prior to installing PyDGN.

Then simply run

.. code-block:: python

    pip install pydgn
