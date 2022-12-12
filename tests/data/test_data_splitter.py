import shutil
from typing import Union, List, Tuple

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, coalesce

from pydgn.data.dataset import DatasetInterface, OGBGDatasetInterface
from pydgn.data.splitter import (
    Splitter,
    SingleGraphSplitter,
    LinkPredictionSingleGraphSplitter,
)


class FakeGraphClassificationDataset(DatasetInterface):
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def download(self):
        pass

    def process(self):
        pass

    def __init__(
        self,
        root="tests/tmp/DATA_graph_classification_dataset",
        name="mock_dataset",
    ):
        super().__init__(root, name)
        self.num_samples = 1000
        self.feats = 10
        self.classes = 3
        self.data = []

        for s in range(self.num_samples):
            num_nodes = int(torch.randint(low=2, high=100, size=(1,)))
            x = torch.rand((num_nodes, self.feats))
            y = torch.randint(self.classes, (1,))
            edge_index = torch.randint(num_nodes, (2, num_nodes * 2))
            edge_attr = torch.rand(edge_index.shape[1])

            self.data.append(
                Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            )

    def get(self, idx: int) -> Data:
        return self.data[idx]

    @property
    def dim_node_features(self) -> int:
        return self.data[0].x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        return 1

    @property
    def dim_target(self) -> int:
        return self.classes

    def len(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self.data)


@pytest.fixture
def graph_classification_dataset():
    """
    Builds a random dataset for graph classification
    """
    return FakeGraphClassificationDataset()


@pytest.fixture
def node_classification_dataset():
    """
    Builds a random dataset for node classification (single graph)
    """

    class FakeDataset(DatasetInterface):
        @property
        def raw_file_names(self) -> Union[str, List[str], Tuple]:
            return []

        @property
        def processed_file_names(self) -> Union[str, List[str], Tuple]:
            return []

        def download(self):
            pass

        def process(self):
            pass

        def __init__(self):
            super().__init__(
                "tests/tmp/DATA_node_classification_dataset", None
            )
            self.feats = 10
            self.classes = 3

            num_nodes = int(torch.randint(low=1000, high=2000, size=(1,)))
            x = torch.rand((num_nodes, self.feats))
            y = torch.randint(self.classes, (num_nodes,))
            edge_index = torch.randint(num_nodes, (2, num_nodes * 2))
            edge_attr = torch.rand(edge_index.shape[1])

            self.data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y
            )

        def get(self, idx: int) -> Data:
            return self.data

        @property
        def dim_node_features(self) -> int:
            return self.data[0].x.shape[1]

        @property
        def dim_edge_features(self) -> int:
            return 1

        @property
        def dim_target(self) -> int:
            return self.classes

        def len(self) -> int:
            return len(self)

        def __len__(self) -> int:
            return 1

    return FakeDataset()


@pytest.fixture
def link_prediction_dataset():
    """
    Builds a random dataset for link prediction (single graph)
    """

    class FakeDataset(DatasetInterface):
        @property
        def raw_file_names(self) -> Union[str, List[str], Tuple]:
            return []

        @property
        def processed_file_names(self) -> Union[str, List[str], Tuple]:
            return []

        def download(self):
            pass

        def process(self):
            pass

        def __init__(self):
            super().__init__("tests/tmp/DATA_link_prediction_dataset", None)
            self.feats = 10
            self.classes = 3

            num_nodes = int(torch.randint(low=100, high=200, size=(1,)))
            x = torch.rand((num_nodes, self.feats))
            # add one class per link, just in case (future link class. tasks)
            y = torch.randint(self.classes, (num_nodes * 2,))
            edge_index = torch.randint(num_nodes, (2, num_nodes * 2))
            edge_attr = torch.rand(edge_index.shape[1])

            # Very important line of code: remove duplicates
            edge_index, edge_attr = coalesce(
                edge_index, edge_attr, num_nodes=num_nodes
            )

            self.data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y
            )

        def get(self, idx: int) -> Data:
            return self.data

        @property
        def dim_node_features(self) -> int:
            return self.data[0].x.shape[1]

        @property
        def dim_edge_features(self) -> int:
            return 1

        @property
        def dim_target(self) -> int:
            return self.classes

        def len(self) -> int:
            return len(self)

        def __len__(self) -> int:
            return 1

    return FakeDataset()


# To each task its own splitter
@pytest.fixture
def node_and_graph_task_input(
    node_classification_dataset, graph_classification_dataset
):
    """
    Returns tuples (dataset, splitter) to test for data splits overlap
    """
    return [
        (graph_classification_dataset, Splitter),
        (node_classification_dataset, SingleGraphSplitter),
    ]


def test_node_graph_split_overlap(node_and_graph_task_input):
    """
    Tests data splits overlap for node and graph prediction
    """
    for dataset, splitter_class in node_and_graph_task_input:
        for n_outer_folds in [1, 10]:
            for n_inner_folds in [1, 10]:
                for stratify in [False, True]:

                    splitter = splitter_class(
                        n_outer_folds=n_outer_folds,
                        n_inner_folds=n_inner_folds,
                        seed=0,
                        stratify=stratify,
                        shuffle=True,
                        outer_val_ratio=0.1,
                    )

                    _, targets = splitter.get_targets(dataset)
                    splitter.split(dataset, targets)

                    for outer in range(n_outer_folds):

                        outer_train_idxs = splitter.outer_folds[
                            outer
                        ].train_idxs
                        outer_val_idxs = splitter.outer_folds[outer].val_idxs
                        outer_test_idxs = splitter.outer_folds[outer].test_idxs

                        # False if empty
                        assert not bool(
                            set(outer_train_idxs)
                            & set(outer_val_idxs)
                            & set(outer_test_idxs)
                        )

                        for inner in range(n_inner_folds):

                            inner_train_idxs = splitter.inner_folds[outer][
                                inner
                            ].train_idxs
                            inner_val_idxs = splitter.inner_folds[outer][
                                inner
                            ].val_idxs

                            # False if empty
                            assert not bool(
                                set(inner_train_idxs)
                                & set(inner_val_idxs)
                                & set(outer_test_idxs)
                            )

                            # Check length consistency
                            if len(dataset) == 1:
                                assert (
                                    len(inner_train_idxs)
                                    + len(inner_val_idxs)
                                    + len(outer_test_idxs)
                                    == dataset[0].x.shape[0]
                                )
                            else:
                                assert len(inner_train_idxs) + len(
                                    inner_val_idxs
                                ) + len(outer_test_idxs) == len(dataset)

                        # Check length consistency
                        if len(dataset) == 1:
                            assert (
                                len(outer_train_idxs)
                                + len(outer_val_idxs)
                                + len(outer_test_idxs)
                                == dataset[0].x.shape[0]
                            )
                        else:
                            assert len(outer_train_idxs) + len(
                                outer_val_idxs
                            ) + len(outer_test_idxs) == len(dataset)
    shutil.rmtree("tests/tmp")


# To each task its own splitter
@pytest.fixture
def link_task_input(link_prediction_dataset):
    """
    Returns tuples (dataset, splitter) to test for data splits overlap
    """
    return [
        (link_prediction_dataset, LinkPredictionSingleGraphSplitter),
    ]


def test_link_split_overlap(link_task_input):
    """
    Tests data splits overlap for link prediction
    """

    def _run_checks(
        original_edge_index,
        pos_train_edges,
        neg_train_edges,
        pos_val_edges,
        neg_val_edges,
        pos_test_edges,
        neg_test_edges,
    ):
        """
        Runs additional memory-intensive checks to ensure the edge split is ok
        and does not contain overlaps.
        """
        original_Adj = to_dense_adj(original_edge_index)[0]

        pos_train_edges = torch.tensor(pos_train_edges).long()
        neg_train_edges = torch.tensor(neg_train_edges).long()
        pos_val_edges = torch.tensor(pos_val_edges).long()
        neg_val_edges = torch.tensor(neg_val_edges).long()
        pos_test_edges = torch.tensor(pos_test_edges).long()
        neg_test_edges = torch.tensor(neg_test_edges).long()

        # First: check positive edges perfectly reconstruct adjacency matrix
        N = original_Adj.shape[0]
        # TEST: restore A with partitioned positive links
        A_pos = to_dense_adj(pos_train_edges, max_num_nodes=N)[0, :, :]
        A_pos += to_dense_adj(pos_val_edges, max_num_nodes=N)[0, :, :]
        A_pos += to_dense_adj(pos_test_edges, max_num_nodes=N)[0, :, :]

        # It is also a way to check that positive edges do not overlap
        # If I do not call to_undirected to val and test edges when
        # undirected==True, this won't hold
        # assert torch.all(A == A_pos)

        # Second: check edges do not overlap
        # Can be done by checking the sum of positive and negative adj matrices
        # holds values smaller or equal than 1.

        # TEST: check negative links are separate from positive links
        A_neg = to_dense_adj(neg_train_edges, max_num_nodes=N)[0, :, :]
        A_neg += to_dense_adj(neg_val_edges, max_num_nodes=N)[0, :, :]
        A_neg += to_dense_adj(neg_test_edges, max_num_nodes=N)[0, :, :]

        assert torch.all((A_neg + A_pos) <= 1.0)

    for dataset, splitter_class in link_task_input:
        for n_outer_folds in [1, 10]:
            for n_inner_folds in [1, 10]:
                for stratify in [False, True]:

                    splitter = splitter_class(
                        n_outer_folds=n_outer_folds,
                        n_inner_folds=n_inner_folds,
                        seed=0,
                        stratify=stratify,
                        shuffle=True,
                        outer_val_ratio=0.1,
                        undirected=False,
                        avoid_opposite_negative_edges=True,
                    )

                    splitter.split(dataset, None)

                    for outer in range(n_outer_folds):

                        (
                            outer_pos_train_edges,
                            outer_train_attr,
                            outer_neg_train_edges,
                        ) = splitter.outer_folds[outer].train_idxs

                        (
                            outer_pos_val_edges,
                            outer_val_attr,
                            outer_neg_val_edges,
                        ) = splitter.outer_folds[outer].val_idxs

                        (
                            outer_pos_test_edges,
                            outer_test_attr,
                            outer_neg_test_edges,
                        ) = splitter.outer_folds[outer].test_idxs

                        _run_checks(
                            dataset[0].edge_index,
                            outer_pos_train_edges,
                            outer_neg_train_edges,
                            outer_pos_val_edges,
                            outer_neg_val_edges,
                            outer_pos_test_edges,
                            outer_neg_test_edges,
                        )

                        for inner in range(n_inner_folds):
                            (
                                inner_pos_train_edges,
                                inner_train_attr,
                                inner_neg_train_edges,
                            ) = splitter.inner_folds[outer][inner].train_idxs

                            (
                                inner_pos_val_edges,
                                inner_val_attr,
                                inner_neg_val_edges,
                            ) = splitter.inner_folds[outer][inner].val_idxs

                            _run_checks(
                                dataset[0].edge_index,
                                inner_pos_train_edges,
                                inner_neg_train_edges,
                                inner_pos_val_edges,
                                inner_neg_val_edges,
                                outer_pos_test_edges,
                                outer_neg_test_edges,
                            )
    shutil.rmtree("tests/tmp")
