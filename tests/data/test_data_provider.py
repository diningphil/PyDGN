from copy import deepcopy
from shutil import rmtree
from unittest.mock import patch

import pytest
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import sort_edge_index

from pydgn.data.dataset import DatasetInterface, ToyIterableDataset
from pydgn.data.provider import (
    DataProvider,
    SingleGraphDataProvider,
    LinkPredictionSingleGraphDataProvider,
    IterableDataProvider,
)
from pydgn.data.splitter import (
    Splitter,
    SingleGraphSplitter,
    LinkPredictionSingleGraphSplitter,
)
from tests.data.test_data_splitter import (
    link_prediction_dataset,
    graph_classification_dataset,
    node_classification_dataset,
)


def mock_get_dataset(cls, **kwargs):
    """
    Returns the dataset stored in the object (see main test)
    """
    return deepcopy(cls.dataset)


def mock_get_splitter(cls, **kwargs):
    """
    Instantiates a splitter and generates random splits
    """
    if cls.splitter is None:
        splitter = Splitter(
            n_outer_folds=cls.outer_folds,
            n_inner_folds=cls.inner_folds,
            seed=0,
            stratify=True,
            shuffle=True,
            outer_val_ratio=0.1,
        )
        dataset = cls._get_dataset()
        splitter.split(dataset, splitter.get_targets(dataset)[1])
        cls.splitter = splitter
        return cls.splitter

    return cls.splitter


def mock_get_link_pred_splitter(cls, **kwargs):
    """
    Instantiates a splitter and generates random splits
    """
    if cls.splitter is None:

        splitter = LinkPredictionSingleGraphSplitter(
            n_outer_folds=cls.outer_folds,
            n_inner_folds=cls.inner_folds,
            seed=0,
            stratify=True,
            shuffle=True,
            outer_val_ratio=0.1,
        )
        dataset = cls._get_dataset()
        splitter.split(dataset, None)
        cls.splitter = splitter
        return cls.splitter

    return cls.splitter


def mock_get_singlegraphsplitter(cls, **kwargs):
    """
    Instantiates a splitter and generates random splits
    """
    splitter = SingleGraphSplitter(
        n_outer_folds=cls.outer_folds,
        n_inner_folds=cls.inner_folds,
        seed=0,
        stratify=True,
        shuffle=True,
        outer_val_ratio=0.1,
    )
    dataset = cls._get_dataset()
    splitter.split(dataset, splitter.get_targets(dataset)[1])

    return splitter


@patch.object(DataProvider, "_get_splitter", mock_get_splitter)
@patch.object(DataProvider, "_get_dataset", mock_get_dataset)
def test_DataProvider(graph_classification_dataset):
    """
    Check that the data provider returns the correct data associated
    with different data splits
    """
    batch_size = 32
    for outer_folds in [1, 10]:
        for inner_folds in [1, 10]:
            for shuffle in [False, True]:
                provider = DataProvider(
                    None,
                    None,
                    DatasetInterface,
                    "",
                    DataLoader,
                    {},
                    outer_folds=outer_folds,
                    inner_folds=inner_folds,
                )
                provider.dataset = graph_classification_dataset
                provider.set_exp_seed(0)

                for o in range(outer_folds):
                    provider.set_outer_k(o)
                    for i in range(inner_folds):
                        provider.set_inner_k(i)

                        inner_train_loader = provider.get_inner_train(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(inner_train_loader.dataset.indices) == set(
                            provider._get_splitter()
                            .inner_folds[o][i]
                            .train_idxs
                        )

                        inner_val_loader = provider.get_inner_val(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(inner_val_loader.dataset.indices) == set(
                            provider._get_splitter().inner_folds[o][i].val_idxs
                        )

                    provider.set_inner_k(None)

                    outer_train_loader = provider.get_outer_train(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert set(outer_train_loader.dataset.indices) == set(
                        provider._get_splitter().outer_folds[o].train_idxs
                    )

                    outer_val_loader = provider.get_outer_val(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert set(outer_val_loader.dataset.indices) == set(
                        provider._get_splitter().outer_folds[o].val_idxs
                    )

                    outer_test_loader = provider.get_outer_test(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert set(outer_test_loader.dataset.indices) == set(
                        provider._get_splitter().outer_folds[o].test_idxs
                    )


@patch.object(
    SingleGraphDataProvider, "_get_splitter", mock_get_singlegraphsplitter
)
@patch.object(SingleGraphDataProvider, "_get_dataset", mock_get_dataset)
def test_SingleGraphDataProvider(node_classification_dataset):
    """
    Check that the data provider returns the correct data associated
    with different data splits
    """
    batch_size = 32
    for outer_folds in [1, 10]:
        for inner_folds in [1, 10]:
            for shuffle in [False, True]:
                provider = SingleGraphDataProvider(
                    None,
                    None,
                    DatasetInterface,
                    "",
                    DataLoader,
                    {},
                    outer_folds=outer_folds,
                    inner_folds=inner_folds,
                )
                provider.dataset = node_classification_dataset
                provider.set_exp_seed(0)

                for o in range(outer_folds):
                    provider.set_outer_k(o)
                    for i in range(inner_folds):
                        provider.set_inner_k(i)

                        inner_train_loader = provider.get_inner_train(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(
                            inner_train_loader.dataset.data.training_indices.tolist()
                        ) == set(
                            provider._get_splitter()
                            .inner_folds[o][i]
                            .train_idxs
                        )

                        assert set(
                            inner_train_loader.dataset.data.eval_indices.tolist()
                        ) == set(
                            provider._get_splitter()
                            .inner_folds[o][i]
                            .train_idxs
                        )

                        inner_val_loader = provider.get_inner_val(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert set(
                            inner_val_loader.dataset.data.training_indices.tolist()
                        ) == set(
                            provider._get_splitter()
                            .inner_folds[o][i]
                            .train_idxs
                        )

                        assert set(
                            inner_val_loader.dataset.data.eval_indices.tolist()
                        ) == set(
                            provider._get_splitter().inner_folds[o][i].val_idxs
                        )

                    provider.set_inner_k(None)

                    outer_train_loader = provider.get_outer_train(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert set(
                        outer_train_loader.dataset.data.training_indices.tolist()
                    ) == set(
                        provider._get_splitter().outer_folds[o].train_idxs
                    )

                    assert set(
                        outer_train_loader.dataset.data.eval_indices.tolist()
                    ) == set(
                        provider._get_splitter().outer_folds[o].train_idxs
                    )

                    outer_val_loader = provider.get_outer_val(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert set(
                        outer_val_loader.dataset.data.training_indices.tolist()
                    ) == set(
                        provider._get_splitter().outer_folds[o].train_idxs
                    )

                    assert set(
                        outer_val_loader.dataset.data.eval_indices.tolist()
                    ) == set(provider._get_splitter().outer_folds[o].val_idxs)

                    outer_test_loader = provider.get_outer_test(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert set(
                        outer_test_loader.dataset.data.training_indices.tolist()
                    ) == set(
                        provider._get_splitter().outer_folds[o].train_idxs
                    )

                    assert set(
                        outer_test_loader.dataset.data.eval_indices.tolist()
                    ) == set(provider._get_splitter().outer_folds[o].test_idxs)


@patch.object(
    LinkPredictionSingleGraphDataProvider,
    "_get_splitter",
    mock_get_link_pred_splitter,
)
@patch.object(
    LinkPredictionSingleGraphDataProvider, "_get_dataset", mock_get_dataset
)
def test_LinkPredictionSingleGraphDataProvider(link_prediction_dataset):
    """
    Check that the data provider returns the correct data associated
    with different data splits
    """
    for batch_size in [0, 32]:
        for outer_folds in [1, 10]:
            for inner_folds in [1, 10]:
                for shuffle in [False, True]:
                    provider = LinkPredictionSingleGraphDataProvider(
                        None,
                        None,
                        DatasetInterface,
                        "",
                        DataLoader,
                        {},
                        outer_folds=outer_folds,
                        inner_folds=inner_folds,
                    )
                    provider.set_exp_seed(0)

                    for o in range(outer_folds):
                        provider.set_outer_k(o)
                        for i in range(inner_folds):

                            # The graph gets modified at each call
                            provider.dataset = link_prediction_dataset
                            provider.set_inner_k(i)

                            inner_train_loader = provider.get_inner_train(
                                shuffle=shuffle, batch_size=batch_size
                            )

                            inner_pos_train_edges, _, inner_neg_train_edges = (
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .train_idxs
                            )

                            inner_val_loader = provider.get_inner_val(
                                shuffle=shuffle, batch_size=batch_size
                            )

                            inner_pos_val_edges, _, inner_neg_val_edges = (
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .val_idxs
                            )

                            for loader, pos_list, neg_list in [
                                (
                                    inner_train_loader,
                                    inner_pos_train_edges,
                                    inner_neg_train_edges,
                                ),
                                (
                                    inner_val_loader,
                                    inner_pos_val_edges,
                                    inner_neg_val_edges,
                                ),
                            ]:
                                pos_edge_idx = torch.cat(
                                    [d.y[1] for d in loader.dataset], dim=1
                                )
                                neg_edge_idx = torch.cat(
                                    [d.y[2] for d in loader.dataset], dim=1
                                )

                                pos_list = torch.tensor(pos_list).long()

                                neg_list = torch.tensor(neg_list).long()

                                pos_edge_idx = sort_edge_index(pos_edge_idx)
                                pos_list = sort_edge_index(pos_list)

                                neg_edge_idx = sort_edge_index(neg_edge_idx)
                                neg_list = sort_edge_index(neg_list)

                                assert (
                                    pos_edge_idx.tolist() == pos_list.tolist()
                                )

                                assert (
                                    neg_edge_idx.tolist() == neg_list.tolist()
                                )

                        provider.set_inner_k(None)

                        outer_train_loader = provider.get_outer_train(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        outer_pos_train_edges, _, outer_neg_train_edges = (
                            provider._get_splitter().outer_folds[o].train_idxs
                        )

                        outer_val_loader = provider.get_outer_val(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        outer_pos_val_edges, _, outer_neg_val_edges = (
                            provider._get_splitter().outer_folds[o].val_idxs
                        )

                        outer_test_loader = provider.get_outer_test(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        outer_pos_test_edges, _, outer_neg_test_edges = (
                            provider._get_splitter().outer_folds[o].test_idxs
                        )

                        for loader, pos_list, neg_list in [
                            (
                                outer_train_loader,
                                outer_pos_train_edges,
                                outer_neg_train_edges,
                            ),
                            (
                                outer_val_loader,
                                outer_pos_val_edges,
                                outer_neg_val_edges,
                            ),
                            (
                                outer_test_loader,
                                outer_pos_test_edges,
                                outer_neg_test_edges,
                            ),
                        ]:
                            pos_edge_idx = torch.cat(
                                [d.y[1] for d in loader.dataset], dim=1
                            )
                            neg_edge_idx = torch.cat(
                                [d.y[2] for d in loader.dataset], dim=1
                            )

                            pos_list = torch.tensor(pos_list).long()

                            neg_list = torch.tensor(neg_list).long()

                            pos_edge_idx = sort_edge_index(pos_edge_idx)
                            pos_list = sort_edge_index(pos_list)

                            neg_edge_idx = sort_edge_index(neg_edge_idx)
                            neg_list = sort_edge_index(neg_list)

                            assert pos_edge_idx.tolist() == pos_list.tolist()

                            assert neg_edge_idx.tolist() == neg_list.tolist()


def mock_get_iterable_splitter(cls, **kwargs):
    """
    Instantiates a splitter and generates random splits
    """
    if cls.splitter is None:
        splitter = Splitter(
            n_outer_folds=cls.outer_folds,
            n_inner_folds=cls.inner_folds,
            seed=0,
            stratify=False,
            shuffle=True,
            outer_val_ratio=0.1,
        )
        dataset = cls._get_dataset()
        splitter.split(dataset, None)
        cls.splitter = splitter
        return cls.splitter

    return cls.splitter


def mock_get_iterable_dataset(cls, **kwargs):
    """
    Returns the dataset stored in the object (see main test)
    """
    data_root = "tests/data/debug/DATA"

    if "url_indices" in kwargs:
        indices = kwargs["url_indices"]
        return ToyIterableDataset(data_root, "toy", url_indices=indices)
    return ToyIterableDataset(data_root, "toy")


@patch.object(IterableDataProvider, "_get_dataset", mock_get_iterable_dataset)
@patch.object(
    IterableDataProvider, "_get_splitter", mock_get_iterable_splitter
)
def test_IterableDataProvider():
    """
    Check that the data provider returns the correct data associated
    with different data splits
    """
    data_root = "tests/data/debug/DATA"
    batch_size = 32
    for outer_folds in [1, 3]:
        for inner_folds in [1, 3]:
            for shuffle in [False, True]:
                provider = IterableDataProvider(
                    data_root,
                    None,
                    DatasetInterface,
                    "",
                    DataLoader,
                    {},
                    outer_folds=outer_folds,
                    inner_folds=inner_folds,
                )
                provider.dataset = ToyIterableDataset(data_root, "toy")
                provider.set_exp_seed(0)

                for o in range(outer_folds):
                    provider.set_outer_k(o)
                    for i in range(inner_folds):
                        provider.set_inner_k(i)

                        inner_train_loader = provider.get_inner_train(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert len(inner_train_loader.dataset) == len(
                            set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .train_idxs
                            )
                        )

                        inner_val_loader = provider.get_inner_val(
                            shuffle=shuffle, batch_size=batch_size
                        )

                        assert len(inner_val_loader.dataset) == len(
                            set(
                                provider._get_splitter()
                                .inner_folds[o][i]
                                .val_idxs
                            )
                        )

                    provider.set_inner_k(None)

                    outer_train_loader = provider.get_outer_train(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert len(outer_train_loader.dataset) == len(
                        set(provider._get_splitter().outer_folds[o].train_idxs)
                    )

                    outer_val_loader = provider.get_outer_val(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert len(outer_val_loader.dataset) == len(
                        set(provider._get_splitter().outer_folds[o].val_idxs)
                    )

                    outer_test_loader = provider.get_outer_test(
                        shuffle=shuffle, batch_size=batch_size
                    )

                    assert len(outer_test_loader.dataset) == len(
                        set(provider._get_splitter().outer_folds[o].test_idxs)
                    )
    rmtree("tests/data/debug/")
