import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, KFold, train_test_split
from torch_geometric.utils import negative_sampling, to_undirected, to_dense_adj, add_self_loops

from pydgn.experiment.util import s2c
from pydgn.static import DATA_SPLITTER_BASE_PATH


class Fold:
    """
    Simple class that stores training and validation/test indices
    """

    def __init__(self, train_idxs, val_idxs=None, test_idxs=None):
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs


class InnerFold(Fold):
    """
    Simple extension of the Fold class that returns a dictionary with training and validation indices (model selection)
    """

    def todict(self):
        return {"train": self.train_idxs, "val": self.val_idxs}


class OuterFold(Fold):
    """
    Simple extension of the Fold class that returns a dictionary with training and test indices (risk assessment)
    """

    def todict(self):
        return {"train": self.train_idxs, "val": self.val_idxs, "test": self.test_idxs}


class NoShuffleTrainTestSplit:

    def __init__(self, test_ratio):
        self.test_ratio = test_ratio

    def split(self, idxs, y=None):
        n_samples = len(idxs)
        n_test = int(n_samples * self.test_ratio)
        n_train = n_samples - n_test
        train_idxs = np.arange(n_train)
        test_idxs = np.arange(n_train, n_train + n_test)
        return [(train_idxs, test_idxs)]


class Splitter:
    """
    Class that generates the splits at dataset creation time.
    """

    @classmethod
    def load(cls, path):
        """
        Loads the splits from disk
        :param path: the path of the yaml file with the splits
        :return: a Splitter object
        """
        splits = torch.load(path)

        splitter_classname = splits.get("splitter_class", "Splitter")
        splitter_class = s2c(DATA_SPLITTER_BASE_PATH + splitter_classname)

        splitter_args = splits.get("splitter_args")
        splitter = splitter_class(**splitter_args)

        assert splitter.n_outer_folds == len(splits["outer_folds"])
        assert splitter.n_inner_folds == len(splits["inner_folds"][0])

        for fold_data in splits["outer_folds"]:
            # v0.4.0, backward compatibility with 0.3.2
            if not hasattr(fold_data, "val") and "val" not in fold_data:
                fold_data["val"] = None
            splitter.outer_folds.append(
                OuterFold(fold_data["train"], val_idxs=fold_data["val"], test_idxs=fold_data["test"]))

        for inner_split in splits["inner_folds"]:
            inner_split_data = []
            for fold_data in inner_split:
                # v0.4.0, backward compatibility with 0.3.2
                if not hasattr(fold_data, "val") and "val" not in fold_data:
                    fold_data["val"] = None
                inner_split_data.append(InnerFold(fold_data["train"], val_idxs=fold_data["val"]))
            splitter.inner_folds.append(inner_split_data)

        return splitter

    def __init__(self, n_outer_folds, n_inner_folds, seed, stratify, shuffle, val_ratio=0.1, test_ratio=0.1):
        """
        Initializes the splitter
        :param n_outer_folds: number of outer folds (risk assessment). 1 means hold-out, >1 means k-fold
        :param n_inner_folds: number of inner folds (model selection). 1 means hold-out, >1 means k-fold
        :param seed: random seed for reproducibility (on the same machine)
        :param stratify: whether to apply stratification or not (should be true for classification tasks)
        :param shuffle: whether to apply shuffle or not
        :param val_ratio: percentage of validation set for hold_out model selection
        :param test_ratio: percentage of test set for hold_out model assessment
        """
        self.outer_folds = []
        self.inner_folds = []
        self.processed = False
        self.stratify = stratify
        self.shuffle = shuffle

        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.seed = seed

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        '''
        self.kwargs = kwargs
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        '''

    def _get_splitter(self, n_splits, stratified, test_ratio):
        if n_splits == 1:
            if not self.shuffle:
                assert stratified == False, "Stratified not implemented when shuffle is False"
                splitter = NoShuffleTrainTestSplit(test_ratio=test_ratio)
            else:
                if stratified:
                    splitter = StratifiedShuffleSplit(n_splits, test_size=test_ratio, random_state=self.seed)
                else:
                    splitter = ShuffleSplit(n_splits, test_size=test_ratio, random_state=self.seed)
        elif n_splits > 1:
            if stratified:
                splitter = StratifiedKFold(n_splits, shuffle=self.shuffle, random_state=self.seed)
            else:
                splitter = KFold(n_splits, shuffle=self.shuffle, random_state=self.seed)
        else:
            raise ValueError(f"'n_splits' must be >=1, got {n_splits}")

        return splitter

    def split(self, dataset, targets=None):
        """
        Computes the splits. The outer split does not include validation (can be extracted from the training set if needed)
        :param dataset: the Dataset object
        :param targets: targets used for stratification
        :param test_ratio: percentage of validation/test set when using an internal/external hold-out split. Default value is 0.1.
        :return:
        """
        idxs = range(len(dataset))

        if not self.processed:

            stratified = self.stratify
            outer_idxs = np.array(idxs)

            outer_splitter = self._get_splitter(
                n_splits=self.n_outer_folds,
                stratified=stratified,
                test_ratio=self.test_ratio)  # This is the true test (outer test)

            for train_idxs, test_idxs in outer_splitter.split(outer_idxs, y=targets):

                assert set(train_idxs) == set(outer_idxs[train_idxs])
                assert set(test_idxs) == set(outer_idxs[test_idxs])

                inner_fold_splits = []
                inner_idxs = outer_idxs[train_idxs]  # equals train_idxs because outer_idxs was ordered
                inner_targets = targets[train_idxs] if targets is not None else None

                inner_splitter = self._get_splitter(
                    n_splits=self.n_inner_folds,
                    stratified=stratified,
                    test_ratio=self.val_ratio)  # The inner "test" is, instead, the validation set

                for inner_train_idxs, inner_val_idxs in inner_splitter.split(inner_idxs, y=inner_targets):
                    inner_fold = InnerFold(train_idxs=inner_idxs[inner_train_idxs].tolist(),
                                           val_idxs=inner_idxs[inner_val_idxs].tolist())
                    inner_fold_splits.append(inner_fold)
                self.inner_folds.append(inner_fold_splits)

                # Obtain outer val from outer train in an holdout fashion
                outer_val_splitter = self._get_splitter(n_splits=1, stratified=stratified,
                                                        test_ratio=self.val_ratio)  # Use val ratio to compute outer val
                outer_train_idxs, outer_val_idxs = list(outer_val_splitter.split(inner_idxs, y=inner_targets))[0]

                # False if empty
                assert not bool(set(inner_train_idxs) & set(inner_val_idxs) & set(test_idxs))
                assert not bool(set(inner_idxs[inner_train_idxs]) & set(inner_idxs[inner_val_idxs]) & set(test_idxs))
                assert not bool(set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs))
                assert not bool(set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs))
                assert not bool(set(inner_idxs[outer_train_idxs]) & set(inner_idxs[outer_val_idxs]) & set(test_idxs))

                np.random.shuffle(outer_train_idxs)
                np.random.shuffle(outer_val_idxs)
                np.random.shuffle(test_idxs)
                outer_fold = OuterFold(train_idxs=inner_idxs[outer_train_idxs].tolist(),
                                       val_idxs=inner_idxs[outer_val_idxs].tolist(),
                                       test_idxs=outer_idxs[test_idxs].tolist())
                self.outer_folds.append(outer_fold)

            self.processed = True

    def _splitter_args(self):
        return {
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds": self.n_inner_folds,
            "seed": self.seed,
            "stratify": self.stratify,
            "shuffle": self.shuffle,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }

    def save(self, path):
        """
        Saves the split into a yaml file
        :param path: filepath where to save the object
        """
        print("Saving splits on disk...")
        savedict = {"seed": self.seed,
                    "splitter_class": self.__class__.__name__,
                    "splitter_args": self._splitter_args()
                    }

        savedict["outer_folds"] = [o.todict() for o in self.outer_folds]
        savedict["inner_folds"] = []
        for inner_split in self.inner_folds:
            savedict["inner_folds"].append([i.todict() for i in inner_split])
        torch.save(savedict, path)
        print("Done.")


class OGBGSplitter(Splitter):

    def split(self, dataset, targets=None):
        """
        Computes the splits. The outer split does not include validation (can be extracted from the training set if needed)
        :param dataset: the Dataset object
        :param targets: targets used for stratification
        :param test_ratio: percentage of validation/test set when using an internal/external hold-out split. Default value is 0.1.
        :return:
        """
        assert self.n_outer_folds == 1 and self.n_inner_folds == 1, "OGBGSplitter assumes you want to use the same splits as in the original dataset!"
        original_splits = dataset.get_idx_split()

        if not self.processed:
            outer_train_indices = original_splits['train'].numpy().tolist()
            outer_valid_indices = original_splits['valid'].numpy().tolist()
            outer_test_indices = original_splits['test'].numpy().tolist()

            np.random.shuffle(outer_train_indices)
            np.random.shuffle(outer_valid_indices)
            np.random.shuffle(outer_test_indices)

            inner_fold_splits = []
            inner_fold = InnerFold(train_idxs=outer_train_indices, val_idxs=outer_valid_indices)
            inner_fold_splits.append(inner_fold)
            self.inner_folds.append(inner_fold_splits)

            outer_fold = OuterFold(train_idxs=outer_train_indices, val_idxs=outer_valid_indices,
                                   test_idxs=outer_test_indices)
            self.outer_folds.append(outer_fold)

            self.processed = True


def to_lower_triangular(edge_index):
    """
    Transform Pytorch Geometric undirected edge index into its "lower triangular counterpart"
    """
    row, col = edge_index
    lower_tri_mask = row > col
    row, col = row[lower_tri_mask], col[lower_tri_mask]
    lower_tri_edge_index = torch.cat((row.unsqueeze(0), col.unsqueeze(0)), dim=0)
    return lower_tri_edge_index


class LinkPredictionSingleGraphSplitter(Splitter):
    """
    Class that computes link splits for link classification tasks.
    IMPORTANT: This class implements bootstrapping rather than k-fold cross-validation, as outer test sets may overlap.
    """

    def __init__(self, n_outer_folds, n_inner_folds, seed, stratify, shuffle, val_ratio, test_ratio, undirected,
                 avoid_opposite_negative_edges, run_checks=False):
        super().__init__(n_outer_folds, n_inner_folds, seed, stratify, shuffle, val_ratio, test_ratio)
        self.run_checks = run_checks
        self.undirected = undirected
        self.avoid_opposite_negative_edges = avoid_opposite_negative_edges

    def _run_checks(self, A, pos_train_edges, neg_train_edges, pos_val_edges, neg_val_edges, pos_test_edges,
                    neg_test_edges):

        print(
            f'Inner splits: {pos_train_edges.shape}, {neg_train_edges.shape}, {pos_val_edges.shape}, {neg_val_edges.shape}')

        # First: check positive edges perfectly reconstruct adjacency matrix
        N = A.shape[0]
        # TEST: restore A with partitioned positive links
        A_pos = to_dense_adj(pos_train_edges, max_num_nodes=N)[0, :, :]
        A_pos += to_dense_adj(pos_val_edges, max_num_nodes=N)[0, :, :]
        A_pos += to_dense_adj(pos_test_edges, max_num_nodes=N)[0, :, :]

        # It is also a way to check that positive edges do not overlap
        # If I do not call to_undirected to val and test edges when undirected==True, this won't hold
        # assert torch.all(A == A_pos)

        # Second: check edges do not overlap
        # Can be done by checking the sum of positive and negative adj matrices holds values smaller or equal than 1.

        # TEST: check negative links are separate from positive links
        A_neg = to_dense_adj(neg_train_edges, max_num_nodes=N)[0, :, :]
        A_neg += to_dense_adj(neg_val_edges, max_num_nodes=N)[0, :, :]
        A_neg += to_dense_adj(neg_test_edges, max_num_nodes=N)[0, :, :]

        assert torch.all((A_neg + A_pos) <= 1.)

    def train_val_test_edge_split(self, edge_index, edge_attr, val_ratio, test_ratio, num_nodes):
        """
        Sample training/validation/test edges at random.
        """
        if self.undirected:
            edge_index = to_lower_triangular(edge_index)

        no_edges = edge_index.shape[1]
        permutation = list(range(no_edges))
        random.shuffle(permutation)

        no_val_elements, no_test_elements = int(no_edges * val_ratio), int(no_edges * test_ratio)

        train_elements = permutation[no_val_elements + no_test_elements:]
        val_elements = permutation[:no_val_elements]
        test_elements = permutation[no_val_elements:no_val_elements + no_test_elements]

        assert set(train_elements).intersection(set(test_elements)) == set()
        assert set(train_elements).intersection(set(val_elements)) == set()
        assert set(val_elements).intersection(set(test_elements)) == set()

        train_edges = edge_index[:, train_elements]
        val_edges = edge_index[:, val_elements]
        test_edges = edge_index[:, test_elements]

        # Restore training edges as undirected for training
        # and double validation and test edges to maintain proportions
        if self.undirected:
            train_edges = to_undirected(train_edges, num_nodes=num_nodes) if train_edges.shape[1] > 0 else train_edges
            val_edges = to_undirected(val_edges, num_nodes=num_nodes) if val_edges.shape[1] > 0 else val_edges
            test_edges = to_undirected(test_edges, num_nodes=num_nodes) if test_edges.shape[1] > 0 else test_edges

        # Use the permuted elements to recover the associated train/val/test edge attributes
        # TODO: If undirected, train_attr should be doubled according to train edges and the logic of to_undirected!
        train_attr = None  # edge_attr[train_elements].tolist() if edge_attr is not None else None
        val_attr = None  # edge_attr[val_elements].tolist() if edge_attr is not None else None
        test_attr = None  # edge_attr[test_elements].tolist() if edge_attr is not None else None

        return train_edges, train_attr, val_edges, val_attr, test_edges, test_attr

    def split(self, dataset, targets=None):
        """
        Computes the splits.
        Links are selected at random: this means outer test folds will overlap almost surely with if test_ratio is 10% of the total samples.
        The recommended procedure here is to use the outer folds to do bootstrapping rather than k-fold cross-validation.
        :param dataset: the Dataset object
        :param targets: (UNUSED) targets used for stratification
        :param val_ratio: percentage of validation set when using an internal hold-out split. Default value is 0.05.
        :param test_ratio: percentage of test set when using an external hold-out split. Default value is 0.1.
        :return:
        """
        random.seed(self.seed)
        assert len(dataset) == 1, "LinkPredictionSingleGraphSplitter works on single graph dataset only!"
        edge_index = dataset.data.edge_index
        edge_attr = dataset.data.edge_attr
        num_nodes = dataset.data.x.shape[0]

        if self.undirected and self.run_checks:
            A = to_dense_adj(edge_index)[0]
            assert torch.all(A == A.transpose(1, 0)), "Passed undirected == True with a non-symmetric adj matrix"
        elif not self.undirected and self.run_checks:
            A = to_dense_adj(edge_index)[0]
            assert torch.any(A != A.transpose(1, 0)), "Passed undirected == False with a symmetric adj matrix"

        if not self.processed:

            val_ratio = self.val_ratio if self.n_inner_folds == 1 else float(1 / self.n_inner_folds)
            test_ratio = self.test_ratio if self.n_outer_folds == 1 else float(1 / self.n_outer_folds)

            # Used by negative sampling later
            original_edge_index = edge_index

            # Add self loops to prevent negative edges from being self loops
            original_edge_index_plus_selfloops, _ = add_self_loops(original_edge_index)

            for outer_k in range(self.n_outer_folds):
                print(f'Processing splits for outer fold n {outer_k + 1}')

                # Create positive train and test edges for outer fold
                print(f'Split positive edges from edge index of shape {original_edge_index.shape[1]}')
                pos_train_edges, train_attr, outer_pos_val_edges, outer_val_attr, pos_test_edges, test_attr = self.train_val_test_edge_split(
                    original_edge_index, edge_attr, val_ratio, test_ratio, num_nodes)

                ###
                # Generate NEGATIVE training/validation/test edges for each fold using the original edge_index
                ###

                # We generate directed negative edges of the same number as true edges
                num_neg_samples = original_edge_index.shape[1]

                # Note: negative edges are directed, but can be considered as undirected if the loss/score is symmetric
                if self.avoid_opposite_negative_edges:
                    # We generate symmetric edges to avoid sampling an opposite edge as negative
                    # Remember that negative sampling without additional args samples directed edges, so we are good
                    negative_edge_index = negative_sampling(to_undirected(original_edge_index_plus_selfloops),
                                                            num_nodes=num_nodes, num_neg_samples=num_neg_samples)
                else:
                    negative_edge_index = negative_sampling(original_edge_index_plus_selfloops, num_nodes=num_nodes,
                                                            num_neg_samples=num_neg_samples)

                # Negative edges are directed, and it does not make sense to double them.
                tmp_undirected = self.undirected
                self.undirected = False
                neg_train_edges, _, outer_neg_val_edges, _, neg_test_edges, _ = self.train_val_test_edge_split(
                    negative_edge_index, None, val_ratio, test_ratio, num_nodes=num_nodes)
                self.undirected = tmp_undirected

                if self.run_checks:
                    print(
                        f'Outer splits: {pos_train_edges.shape}, {neg_train_edges.shape}, {outer_pos_val_edges.shape}, {outer_neg_val_edges.shape}, {pos_test_edges.shape}, {neg_test_edges.shape}')

                inner_fold_splits = []
                for inner_k in range(self.n_inner_folds):

                    # Do not need to create additional negative edges here! Just split what we already have

                    # Create positive train and validation edges for outer fold
                    inner_pos_train_edges, inner_train_attr, inner_pos_val_edges, inner_val_attr, _, _ = self.train_val_test_edge_split(
                        pos_train_edges, train_attr, val_ratio, 0., num_nodes=num_nodes)

                    # Negative edges are directed, and it does not make sense to double them.
                    tmp_undirected = self.undirected
                    self.undirected = False
                    # Create negative train and validation edges for outer fold
                    inner_neg_train_edges, _, inner_neg_val_edges, _, _, _ = self.train_val_test_edge_split(
                        neg_train_edges, None, val_ratio, 0., num_nodes=num_nodes)
                    self.undirected = tmp_undirected

                    inner_fold = InnerFold(train_idxs=(inner_pos_train_edges.tolist(),
                                                       inner_train_attr,
                                                       inner_neg_train_edges.tolist()),
                                           val_idxs=(inner_pos_val_edges.tolist(),
                                                     inner_val_attr,
                                                     inner_neg_val_edges.tolist()))
                    inner_fold_splits.append(inner_fold)

                    if self.run_checks:
                        print("Running additional checks: disable this option after testing the function")
                        self._run_checks(A, inner_pos_train_edges, inner_neg_train_edges, inner_pos_val_edges,
                                         inner_neg_val_edges, pos_test_edges, neg_test_edges)

                # Finally, check we are not creating positive links for some mistake
                if self.run_checks:
                    A = to_dense_adj(edge_index)[0]
                    neg_A = to_dense_adj(negative_edge_index, max_num_nodes=A.shape[0])[0, :, :]
                    assert torch.all(A + neg_A <= 1.)

                self.inner_folds.append(inner_fold_splits)

                outer_fold = OuterFold(train_idxs=(pos_train_edges.tolist(),
                                                   train_attr,
                                                   neg_train_edges.tolist()),
                                       val_idxs=(outer_pos_val_edges.tolist(),
                                                 outer_val_attr,
                                                 outer_neg_val_edges.tolist()),
                                       test_idxs=(pos_test_edges.tolist(),
                                                  test_attr,
                                                  neg_test_edges.tolist()))
                self.outer_folds.append(outer_fold)

            self.processed = True

    def _splitter_args(self):
        splitter_args = super()._splitter_args()
        splitter_args.update({"undirected": self.undirected,
                              "avoid_opposite_negative_edges": self.avoid_opposite_negative_edges})
        return splitter_args
