import random
from typing import Tuple

import numpy as np
import torch

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
    KFold,
)
from torch_geometric.utils import (
    negative_sampling,
    to_undirected,
    to_dense_adj,
    add_self_loops,
    is_undirected,
)

import pydgn
from pydgn.data.dataset import OGBGDatasetInterface, IterableDatasetInterface
from pydgn.data.util import to_lower_triangular
from pydgn.experiment.util import s2c


class Fold:
    r"""
    Simple class that stores training, validation, and test indices.

    Args:
        train_idxs (Union[list, tuple]): training indices
        val_idxs (Union[list, tuple]): validation indices. Default is ``None``
        test_idxs (Union[list, tuple]): test indices. Default is ``None``
    """

    def __init__(self, train_idxs, val_idxs=None, test_idxs=None):
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs


class InnerFold(Fold):
    r"""
    Simple extension of the Fold class that returns a dictionary with
    training and validation indices (model selection).
    """

    def todict(self) -> dict:
        r"""
        Creates a dictionary with the training/validation indices.

        Returns:
            a dict with keys ``['train', 'val']`` associated with the
            respective indices
        """
        return {"train": self.train_idxs, "val": self.val_idxs}


class OuterFold(Fold):
    r"""
    Simple extension of the Fold class that returns a dictionary with training
    and test indices (risk assessment)
    """

    def todict(self) -> dict:
        r"""
        Creates a dictionary with the training/validation/test indices.

        Returns:
            a dict with keys ``['train', 'val', 'test']``
            associated with the respective indices
        """
        return {
            "train": self.train_idxs,
            "val": self.val_idxs,
            "test": self.test_idxs,
        }


class NoShuffleTrainTestSplit:
    r"""
    Class that implements a very simple training/test split. Can be used to
    further split training data into training and validation.

    Args:
        test_ratio: percentage of data to use for evaluation.
    """

    def __init__(self, test_ratio):
        self.test_ratio = test_ratio

    # Leave the arguments as they are. The parameter `y` is needed to
    # implement the same interface as sklearn
    def split(self, idxs, y=None):
        """
        Splits the data.

        Args:
            idxs: the indices to split according to the `test_ratio` parameter
            y: Unused argument

        Returns:
            a list of a single tuple (train indices, test/eval indices)
        """
        n_samples = len(idxs)
        n_test = int(n_samples * self.test_ratio)
        n_train = n_samples - n_test
        train_idxs = np.arange(n_train)
        test_idxs = np.arange(n_train, n_train + n_test)
        return [(train_idxs, test_idxs)]


class Splitter:
    r"""
    Class that generates and stores the data splits at dataset creation time.

    Args:
        n_outer_folds (int): number of outer folds (risk assessment).
            1 means hold-out, >1 means k-fold
        n_inner_folds (int): number of inner folds (model selection).
            1 means hold-out, >1 means k-fold
        seed (int): random seed for reproducibility (on the same machine)
        stratify (bool): whether to apply stratification or not
            (should be true for classification tasks)
        shuffle (bool): whether to apply shuffle or not
        inner_val_ratio  (float): percentage of validation set for
            hold_out model selection. Default is ``0.1``
        outer_val_ratio  (float): percentage of validation set for
            hold_out model assessment (final training runs).
            Default is ``0.1``
        test_ratio  (float): percentage of test set for
            hold_out model assessment. Default is ``0.1``
    """

    def __init__(
        self,
        n_outer_folds: int,
        n_inner_folds: int,
        seed: int,
        stratify: bool,
        shuffle: bool,
        inner_val_ratio: float = 0.1,
        outer_val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.outer_folds = []
        self.inner_folds = []
        self.stratify = stratify
        self.shuffle = shuffle

        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.seed = seed

        self.inner_val_ratio = inner_val_ratio
        self.outer_val_ratio = outer_val_ratio
        self.test_ratio = test_ratio

    def get_targets(
        self, dataset: pydgn.data.dataset.DatasetInterface
    ) -> Tuple[bool, np.ndarray]:
        r"""
        Reads the entire dataset and returns the targets.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the dataset

        Returns:
            a tuple of two elements. The first element is a boolean, which
            is ``True`` if target values exist or an exception has not
            been thrown. The second value holds the actual targets or ``None``,
            depending on the first boolean value.
        """
        try:
            _ = hasattr(dataset[0], "y")
        except NotImplementedError as e:
            assert issubclass(dataset.__class__, IterableDatasetInterface)
            return False, False

        if hasattr(dataset[0], "y") and dataset[0].y is not None:
            targets = torch.cat([d.y for d in dataset], dim=0).numpy()
            return True, targets
        else:
            return False, None

    @classmethod
    def load(cls, path: str):
        r"""
        Loads the data splits from disk.

        Args:
            :param path: the path of the yaml file with the splits

        Returns:
            a :class:`~pydgn.data.splitter.Splitter` object
        """
        splits = torch.load(path)

        splitter_classname = splits.get("splitter_class", "Splitter")
        splitter_class = s2c(splitter_classname)

        splitter_args = splits.get("splitter_args")
        splitter = splitter_class(**splitter_args)

        assert splitter.n_outer_folds == len(splits["outer_folds"])
        assert splitter.n_inner_folds == len(splits["inner_folds"][0])

        for fold_data in splits["outer_folds"]:
            splitter.outer_folds.append(
                OuterFold(
                    fold_data["train"],
                    val_idxs=fold_data["val"],
                    test_idxs=fold_data["test"],
                )
            )

        for inner_split in splits["inner_folds"]:
            inner_split_data = []
            for fold_data in inner_split:
                inner_split_data.append(
                    InnerFold(fold_data["train"], val_idxs=fold_data["val"])
                )
            splitter.inner_folds.append(inner_split_data)

        return splitter

    def _get_splitter(
        self, n_splits: int, stratified: bool, eval_ratio: float
    ):
        r"""
        Instantiates the appropriate splitter to use depending on the situation

        Args:
            n_splits (int): the number of different splits to create
            stratified (bool): whether or not to perform stratification.
                **Works with graph classification tasks only!**
            eval_ratio (float): the amount of evaluation (validation/test)
                data to use in case ``n_splits==1`` (i.e., hold-out data split)

        Returns:
            a :class:`~pydgn.data.splitter.Splitter` object
        """
        if n_splits == 1:
            if not self.shuffle:
                assert (
                    stratified is False
                ), "Stratified not implemented when shuffle is False"
                splitter = NoShuffleTrainTestSplit(test_ratio=eval_ratio)
            else:
                if stratified:
                    splitter = StratifiedShuffleSplit(
                        n_splits, test_size=eval_ratio, random_state=self.seed
                    )
                else:
                    splitter = ShuffleSplit(
                        n_splits, test_size=eval_ratio, random_state=self.seed
                    )
        elif n_splits > 1:
            if stratified:
                splitter = StratifiedKFold(
                    n_splits, shuffle=self.shuffle, random_state=self.seed
                )
            else:
                splitter = KFold(
                    n_splits, shuffle=self.shuffle, random_state=self.seed
                )
        else:
            raise ValueError(f"'n_splits' must be >=1, got {n_splits}")

        return splitter

    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch, and
        random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        idxs = range(len(dataset))

        stratified = self.stratify
        outer_idxs = np.array(idxs)

        outer_splitter = self._get_splitter(
            n_splits=self.n_outer_folds,
            stratified=stratified,
            eval_ratio=self.test_ratio,
        )  # This is the true test (outer test)

        for train_idxs, test_idxs in outer_splitter.split(
            outer_idxs, y=targets if stratified else None
        ):

            assert set(train_idxs) == set(outer_idxs[train_idxs])
            assert set(test_idxs) == set(outer_idxs[test_idxs])

            inner_fold_splits = []
            inner_idxs = outer_idxs[
                train_idxs
            ]  # equals train_idxs because outer_idxs was ordered
            inner_targets = (
                targets[train_idxs] if targets is not None else None
            )

            inner_splitter = self._get_splitter(
                n_splits=self.n_inner_folds,
                stratified=stratified,
                eval_ratio=self.inner_val_ratio,
            )  # The inner "test" is, instead, the validation set

            for inner_train_idxs, inner_val_idxs in inner_splitter.split(
                inner_idxs, y=inner_targets if stratified else None
            ):
                inner_fold = InnerFold(
                    train_idxs=inner_idxs[inner_train_idxs].tolist(),
                    val_idxs=inner_idxs[inner_val_idxs].tolist(),
                )
                inner_fold_splits.append(inner_fold)

                # False if empty
                assert not bool(
                    set(inner_train_idxs)
                    & set(inner_val_idxs)
                    & set(test_idxs)
                )
                assert not bool(
                    set(inner_idxs[inner_train_idxs])
                    & set(inner_idxs[inner_val_idxs])
                    & set(test_idxs)
                )

            self.inner_folds.append(inner_fold_splits)

            # Obtain outer val from outer train in an holdout fashion
            outer_val_splitter = self._get_splitter(
                n_splits=1,
                stratified=stratified,
                eval_ratio=self.outer_val_ratio,
            )
            outer_train_idxs, outer_val_idxs = list(
                outer_val_splitter.split(inner_idxs, y=inner_targets)
            )[0]

            # False if empty
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(inner_idxs[outer_train_idxs])
                & set(inner_idxs[outer_val_idxs])
                & set(test_idxs)
            )

            np.random.shuffle(outer_train_idxs)
            np.random.shuffle(outer_val_idxs)
            np.random.shuffle(test_idxs)
            outer_fold = OuterFold(
                train_idxs=inner_idxs[outer_train_idxs].tolist(),
                val_idxs=inner_idxs[outer_val_idxs].tolist(),
                test_idxs=outer_idxs[test_idxs].tolist(),
            )
            self.outer_folds.append(outer_fold)

    def _splitter_args(self) -> dict:
        r"""
        Returns a dict with all the splitter's arguments for subsequent
        re-loading at experiment time.

        Returns:
            a dict containing all splitter's arguments.
        """
        return {
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds": self.n_inner_folds,
            "seed": self.seed,
            "stratify": self.stratify,
            "shuffle": self.shuffle,
            "inner_val_ratio": self.inner_val_ratio,
            "outer_val_ratio": self.outer_val_ratio,
            "test_ratio": self.test_ratio,
        }

    def save(self, path: str):
        r"""
        Saves the split as a dictionary into a ``torch`` file. The arguments
        of the dictionary are
        * seed (int)
        * splitter_class (str)
        * splitter_args (dict)
        * outer_folds (list of dicts)
        * inner_folds (list of lists of dicts)

        Args:
            path (str): filepath where to save the object
        """
        print("Saving splits on disk...")

        # save split class name
        module = self.__class__.__module__
        splitter_class = self.__class__.__qualname__
        if module is not None and module != "__builtin__":
            splitter_class = module + "." + splitter_class

        savedict = {
            "seed": self.seed,
            "splitter_class": splitter_class,
            "splitter_args": self._splitter_args(),
            "outer_folds": [o.todict() for o in self.outer_folds],
            "inner_folds": [],
        }

        for (
            inner_split
        ) in self.inner_folds:  # len(self.inner_folds) == # of **outer** folds
            savedict["inner_folds"].append([i.todict() for i in inner_split])
        torch.save(savedict, path)
        print("Done.")


class TemporalSplitter(Splitter):
    r"""
    Reads the entire dataset and returns the targets. In this case, each
    sample in the dataset represents a temporal graph, so we get the
    classification value at the last time step. Use this method to
    stratify a dataset of multiple temporal graphs.

    Args:
        dataset (:class:`~pydgn.data.dataset.DatasetInterface`): the dataset

    Returns:
        a tuple of two elements. The first element is a boolean, which is
        ``True`` if target values exist or an exception has not been thrown.
        The second value holds the actual targets or ``None``, depending on the
        first boolean value.
    """

    def get_targets(
        self, dataset: pydgn.data.dataset.TemporalDatasetInterface
    ) -> Tuple[bool, np.ndarray]:
        r"""
        Reads the entire dataset and returns the targets.

        Args:
            dataset (:class:`~pydgn.data.dataset.TemporalDatasetInterface`):
                the temporal dataset

        Returns:
            a tuple of two elements. The first element is a boolean, which
            is ``True`` if target values exist or an exception has not
            been thrown. The second value holds the actual targets or
            ``None``, depending on the first boolean value.
        """
        raise NotImplementedError(
            "You should subclass TemporalSplitter and implement this function!"
        )


class OGBGSplitter(Splitter):
    """
    Splitter specific to OGBG Datasets, reuses the already given splits
    (hence it works only in hold-out mode).
    """

    def split(self, dataset: OGBGDatasetInterface, targets=None):
        r"""
        Computes the OGBG splits according to those already provided by
        the authors of the datasets and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy,
        torch, and random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.OGBGDatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        assert self.n_outer_folds == 1 and self.n_inner_folds == 1, (
            "OGBGSplitter assumes you want to use the same splits "
            "as in the original dataset!"
        )
        original_splits = dataset.get_idx_split()

        outer_train_indices = original_splits["train"].numpy().tolist()
        outer_valid_indices = original_splits["valid"].numpy().tolist()
        outer_test_indices = original_splits["test"].numpy().tolist()

        np.random.shuffle(outer_train_indices)
        np.random.shuffle(outer_valid_indices)
        np.random.shuffle(outer_test_indices)

        inner_fold_splits = []
        inner_fold = InnerFold(
            train_idxs=outer_train_indices, val_idxs=outer_valid_indices
        )
        inner_fold_splits.append(inner_fold)
        self.inner_folds.append(inner_fold_splits)

        outer_fold = OuterFold(
            train_idxs=outer_train_indices,
            val_idxs=outer_valid_indices,
            test_idxs=outer_test_indices,
        )
        self.outer_folds.append(outer_fold)


class SingleGraphSequenceSplitter(TemporalSplitter):
    """
    Class for dynamic graphs that generates the splits at dataset creation
    time. It assumes that there is a single graph sequence, so the split
    happens on time steps. What is more, `n_inner_folds` here will create an
    inner K-fold CV split for model selection where, however, the
    training/validation split will not change (because there is no way to
    split time steps in a different way). This allows for different
    initializations of the same model, evaluating the avg performance
    on the VL set.
    """

    def __init__(
        self,
        n_outer_folds: int,
        n_inner_folds: int,
        seed: int,
        stratify: bool,
        shuffle: bool,
        inner_val_ratio: float = 0.1,
        outer_val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        super().__init__(
            n_outer_folds,
            n_inner_folds,
            seed,
            stratify,
            shuffle,
            inner_val_ratio,
            outer_val_ratio,
            test_ratio,
        )
        assert n_outer_folds == 1, (
            "SingleGraphSequenceSplitter does not allow more than "
            "one outer fold, as the graph sequence is unique."
        )
        assert not shuffle, (
            "SingleGraphSequenceSplitter does not allow to shuffle "
            "the time steps of the unique graph sequence."
        )
        assert not stratify, "You cannot stratify a single graph sequence."

    def get_targets(
        self, dataset: pydgn.data.dataset.TemporalDatasetInterface
    ) -> Tuple[bool, np.ndarray]:
        r"""
        Reads the entire dataset and returns the targets.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the dataset

        Returns:
            a tuple of two elements. The first element is a boolean, which is
            ``True`` if target values exist or an exception has not been
            thrown. The second value holds the actual targets or
            ``None``, depending on the first boolean value.
        """
        try:
            targets = np.array([d.targets[-1].item() for d in dataset])
            return True, targets
        except Exception:
            return False, None

    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        IMPORTANT: calling split() sets the seed of numpy, torch,
        and random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        idxs = range(len(dataset))
        print(f"Number of snapshots: {len(dataset)}")

        outer_splitter = self._get_splitter(
            n_splits=self.n_outer_folds,
            stratified=self.stratify,
            eval_ratio=self.test_ratio,
        )  # This is the true test (outer test)

        outer_idxs = np.array(idxs)
        train_idxs, test_idxs = outer_splitter.split(outer_idxs, y=targets)[0]

        assert set(train_idxs) == set(outer_idxs[train_idxs])
        assert set(test_idxs) == set(outer_idxs[test_idxs])

        inner_fold_splits = []
        inner_idxs = outer_idxs[
            train_idxs
        ]  # equals train_idxs because outer_idxs was ordered

        inner_splitter = NoShuffleTrainTestSplit(
            test_ratio=self.inner_val_ratio
        )

        # Repeat the very same inner split for n_inner_folds, as there is no
        # way to split differently.
        # This allows for different initializations of the same model,
        # evaluating the avg performance on the VL set.
        inner_train_idxs, inner_val_idxs = inner_splitter.split(
            inner_idxs, y=None
        )[0]
        for _ in range(self.n_inner_folds):
            inner_fold = InnerFold(
                train_idxs=inner_idxs[inner_train_idxs].tolist(),
                val_idxs=inner_idxs[inner_val_idxs].tolist(),
            )
            inner_fold_splits.append(inner_fold)

        self.inner_folds.append(inner_fold_splits)

        # Obtain outer val from outer train in an holdout fashion
        outer_val_splitter = NoShuffleTrainTestSplit(
            test_ratio=self.outer_val_ratio
        )
        outer_train_idxs, outer_val_idxs = outer_val_splitter.split(
            inner_idxs, y=None
        )[0]

        # False if empty
        assert not bool(
            set(inner_train_idxs) & set(inner_val_idxs) & set(test_idxs)
        )
        assert not bool(
            set(inner_idxs[inner_train_idxs])
            & set(inner_idxs[inner_val_idxs])
            & set(test_idxs)
        )
        assert not bool(
            set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
        )
        assert not bool(
            set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
        )
        assert not bool(
            set(inner_idxs[outer_train_idxs])
            & set(inner_idxs[outer_val_idxs])
            & set(test_idxs)
        )

        outer_fold = OuterFold(
            train_idxs=inner_idxs[outer_train_idxs].tolist(),
            val_idxs=inner_idxs[outer_val_idxs].tolist(),
            test_idxs=outer_idxs[test_idxs].tolist(),
        )
        self.outer_folds.append(outer_fold)


class SingleGraphSplitter(Splitter):
    r"""
    A splitter for a single graph dataset that randomly splits nodes into
    training/validation/test

    Args:
        n_outer_folds (int): number of outer folds (risk assessment).
            1 means hold-out, >1 means k-fold
        n_inner_folds (int): number of inner folds (model selection).
            1 means hold-out, >1 means k-fold
        seed (int): random seed for reproducibility (on the same machine)
        stratify (bool): whether to apply stratification or not
            (should be true for classification tasks)
        shuffle (bool): whether to apply shuffle or not
        inner_val_ratio  (float): percentage of validation set for
            hold_out model selection. Default is ``0.1``
        outer_val_ratio  (float): percentage of validation set for
            hold_out model assessment (final training runs).
            Default is ``0.1``
        test_ratio  (float): percentage of test set for
            hold_out model assessment. Default is ``0.1``
    """

    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Compared with the superclass version, the only difference is that the
        range of indices spans across the number of nodes of the single
        graph taken into consideration.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        assert (
            len(dataset) == 1
        ), "This class works only with single graph datasets"
        idxs = range(dataset.data.x.shape[0])
        stratified = self.stratify
        outer_idxs = np.array(idxs)

        # all this code below could be reused from the original splitter
        outer_splitter = self._get_splitter(
            n_splits=self.n_outer_folds,
            stratified=stratified,
            eval_ratio=self.test_ratio,
        )  # This is the true test (outer test)

        for train_idxs, test_idxs in outer_splitter.split(
            outer_idxs, y=targets
        ):

            assert set(train_idxs) == set(outer_idxs[train_idxs])
            assert set(test_idxs) == set(outer_idxs[test_idxs])

            inner_fold_splits = []
            inner_idxs = outer_idxs[
                train_idxs
            ]  # equals train_idxs because outer_idxs was ordered
            inner_targets = (
                targets[train_idxs] if targets is not None else None
            )

            inner_splitter = self._get_splitter(
                n_splits=self.n_inner_folds,
                stratified=stratified,
                eval_ratio=self.inner_val_ratio,
            )  # The inner "test" is, instead, the validation set

            for inner_train_idxs, inner_val_idxs in inner_splitter.split(
                inner_idxs, y=inner_targets
            ):
                inner_fold = InnerFold(
                    train_idxs=inner_idxs[inner_train_idxs].tolist(),
                    val_idxs=inner_idxs[inner_val_idxs].tolist(),
                )
                inner_fold_splits.append(inner_fold)

                # False if empty
                assert not bool(
                    set(inner_train_idxs)
                    & set(inner_val_idxs)
                    & set(test_idxs)
                )
                assert not bool(
                    set(inner_idxs[inner_train_idxs])
                    & set(inner_idxs[inner_val_idxs])
                    & set(test_idxs)
                )

            self.inner_folds.append(inner_fold_splits)

            # Obtain outer val from outer train in an holdout fashion
            outer_val_splitter = self._get_splitter(
                n_splits=1,
                stratified=stratified,
                eval_ratio=self.outer_val_ratio,
            )
            outer_train_idxs, outer_val_idxs = list(
                outer_val_splitter.split(inner_idxs, y=inner_targets)
            )[0]

            # False if empty
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(outer_train_idxs) & set(outer_val_idxs) & set(test_idxs)
            )
            assert not bool(
                set(inner_idxs[outer_train_idxs])
                & set(inner_idxs[outer_val_idxs])
                & set(test_idxs)
            )

            np.random.shuffle(outer_train_idxs)
            np.random.shuffle(outer_val_idxs)
            np.random.shuffle(test_idxs)
            outer_fold = OuterFold(
                train_idxs=inner_idxs[outer_train_idxs].tolist(),
                val_idxs=inner_idxs[outer_val_idxs].tolist(),
                test_idxs=outer_idxs[test_idxs].tolist(),
            )
            self.outer_folds.append(outer_fold)


class LinkPredictionSingleGraphSplitter(Splitter):
    r"""
    Class that inherits from :class:`~pydgn.data.splitter.Splitter` and
    computes link splits for link classification tasks.
    **IMPORTANT:** This class implements **bootstrapping** rather
    than k-fold cross-validation, so different outer test sets may have
    overlapping indices.

    **Does not support edge attributes at the moment.**

    Args:
        n_outer_folds (int): number of outer folds (risk assessment).
            1 means hold-out, >1 means k-fold
        n_inner_folds (int): number of inner folds (model selection).
            1 means hold-out, >1 means k-fold
        seed (int): random seed for reproducibility (on the same machine)
        stratify (bool): whether to apply stratification or not
            (should be true for classification tasks)
        shuffle (bool): whether to apply shuffle or not
        inner_val_ratio  (float): percentage of validation set for
            hold_out model selection. Default is ``0.1``
        outer_val_ratio  (float): percentage of validation set for
            hold_out model assessment (final training runs).
            Default is ``0.1``
        test_ratio  (float): percentage of test set for
            hold_out model assessment. Default is ``0.1``
        undirected (bool): whether or not the graph is undirected.
            Default is ``False``
        avoid_opposite_negative_edges (bool): whether or not to **avoid**
            creating negative edges that are opposite
            to existing edges Default is ``True``
    """

    def __init__(
        self,
        n_outer_folds: int,
        n_inner_folds: int,
        seed: int,
        stratify: bool,
        shuffle: bool,
        inner_val_ratio: float = 0.1,
        outer_val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        undirected: bool = False,
        avoid_opposite_negative_edges: bool = True,
    ):
        super().__init__(
            n_outer_folds,
            n_inner_folds,
            seed,
            stratify,
            shuffle,
            inner_val_ratio,
            outer_val_ratio,
            test_ratio,
        )
        self.undirected = undirected
        self.avoid_opposite_negative_edges = avoid_opposite_negative_edges

    def train_val_test_edge_split(
        self, edge_index, edge_attr, val_ratio, test_ratio, num_nodes
    ):
        r"""
        Sample training/validation/test edges at random.
        """
        if self.undirected:
            edge_index = to_lower_triangular(edge_index)

        no_edges = edge_index.shape[1]
        permutation = list(range(no_edges))
        random.shuffle(permutation)

        no_val_elements, no_test_elements = int(no_edges * val_ratio), int(
            no_edges * test_ratio
        )

        train_elements = permutation[no_val_elements + no_test_elements :]
        val_elements = permutation[:no_val_elements]
        test_elements = permutation[
            no_val_elements : no_val_elements + no_test_elements
        ]

        assert set(train_elements).intersection(set(test_elements)) == set()
        assert set(train_elements).intersection(set(val_elements)) == set()
        assert set(val_elements).intersection(set(test_elements)) == set()

        train_edges = edge_index[:, train_elements]
        val_edges = edge_index[:, val_elements]
        test_edges = edge_index[:, test_elements]

        if edge_attr is not None:
            train_attr = edge_attr[train_elements]
            val_attr = edge_attr[val_elements]
            test_attr = edge_attr[test_elements]
        else:
            train_attr = None
            val_attr = None
            test_attr = None

        # Restore training edges as undirected for training
        # and double validation and test edges to maintain proportions
        if self.undirected:
            train_edges, train_attr = (
                to_undirected(train_edges, train_attr, num_nodes=num_nodes)
                if train_edges.shape[1] > 0
                else train_edges
            )
            val_edges, val_attr = (
                to_undirected(val_edges, val_attr, num_nodes=num_nodes)
                if val_edges.shape[1] > 0
                else val_edges
            )
            test_edges, test_attr = (
                to_undirected(test_edges, test_attr, num_nodes=num_nodes)
                if test_edges.shape[1] > 0
                else test_edges
            )

        return (
            train_edges,
            train_attr,
            val_edges,
            val_attr,
            test_edges,
            test_attr,
        )

    def split(
        self,
        dataset: pydgn.data.dataset.DatasetInterface,
        targets: np.ndarray = None,
    ):
        r"""
        Computes the splits and stores them in the list fields
        ``self.outer_folds`` and ``self.inner_folds``.
        Links are selected at random: this means outer test folds will
        overlap almost surely with if test_ratio is 10% of the total samples.
        The recommended procedure here is to use the outer folds to do
        bootstrapping rather than k-fold cross-validation. Idea taken
        from: https://arxiv.org/pdf/1811.05868.pdf .
        IMPORTANT: calling split() sets the seed of numpy, torch,
        and random for reproducibility.

        Args:
            dataset (:class:`~pydgn.data.dataset.DatasetInterface`):
                the Dataset object
            targets (np.ndarray]): targets used for stratification.
                Default is ``None``
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

        assert len(dataset) == 1, (
            f"LinkPredictionSingleGraphSplitter works on single graph "
            f"dataset only! Here we have {len(dataset)}"
        )
        edge_index = dataset.data.edge_index
        edge_attr = dataset.data.edge_attr
        num_nodes = dataset.data.x.shape[0]

        if self.undirected:
            assert is_undirected(
                edge_index
            ), "Passed undirected == True with a non-symmetric adj matrix"
        elif not self.undirected:
            assert not is_undirected(
                edge_index
            ), "Passed undirected == False with a non-symmetric adj matrix"

        inner_val_ratio = (
            self.inner_val_ratio
            if self.n_inner_folds == 1
            else float(1 / self.n_inner_folds)
        )
        outer_val_ratio = (
            self.outer_val_ratio
            if self.n_inner_folds == 1
            else float(1 / self.n_inner_folds)
        )
        test_ratio = (
            self.test_ratio
            if self.n_outer_folds == 1
            else float(1 / self.n_outer_folds)
        )

        # Used by negative sampling later
        original_edge_index = edge_index

        # Add self loops to prevent negative edges from being self loops
        original_edge_index_plus_selfloops, _ = add_self_loops(
            original_edge_index
        )

        for outer_k in range(self.n_outer_folds):
            # print(f"Processing splits for outer fold n {outer_k + 1}")

            # Create positive train and test edges for outer fold
            # print(
            #     f"Split positive edges from edge index of shape "
            #     f"{original_edge_index.shape[1]}"
            # )
            (
                pos_train_edges,
                train_attr,
                outer_pos_val_edges,
                outer_val_attr,
                pos_test_edges,
                test_attr,
            ) = self.train_val_test_edge_split(
                original_edge_index,
                edge_attr,
                outer_val_ratio,
                test_ratio,
                num_nodes,
            )

            ###
            # Generate NEGATIVE training/validation/test edges for each
            # fold using the original edge_index
            ###

            # We generate directed negative edges of the same number as true
            # edges
            num_neg_samples = original_edge_index.shape[1]

            # Note: negative edges are directed, but can be considered as
            # undirected if the loss/score is symmetric
            if self.avoid_opposite_negative_edges:
                # We generate symmetric edges to avoid sampling an opposite
                # edge as negative
                # Remember that negative sampling without additional args
                # samples directed edges, so we are good
                negative_edge_index = negative_sampling(
                    to_undirected(original_edge_index_plus_selfloops),
                    num_nodes=num_nodes,
                    num_neg_samples=num_neg_samples,
                )
            else:
                negative_edge_index = negative_sampling(
                    original_edge_index_plus_selfloops,
                    num_nodes=num_nodes,
                    num_neg_samples=num_neg_samples,
                )

            # Negative edges are directed, and it does not make sense to double
            # them.
            tmp_undirected = self.undirected
            self.undirected = False
            (
                neg_train_edges,
                _,
                outer_neg_val_edges,
                _,
                neg_test_edges,
                _,
            ) = self.train_val_test_edge_split(
                negative_edge_index,
                None,
                outer_val_ratio,
                test_ratio,
                num_nodes=num_nodes,
            )
            self.undirected = tmp_undirected

            inner_fold_splits = []
            for inner_k in range(self.n_inner_folds):

                # Do not need to create additional negative edges here! Just
                # split what we already have

                # Create positive train and validation edges for outer fold
                (
                    inner_pos_train_edges,
                    inner_train_attr,
                    inner_pos_val_edges,
                    inner_val_attr,
                    _,
                    _,
                ) = self.train_val_test_edge_split(
                    pos_train_edges,
                    train_attr,
                    inner_val_ratio,
                    0.0,
                    num_nodes=num_nodes,
                )

                # Negative edges are directed, and it does not make sense to
                # double them.
                tmp_undirected = self.undirected
                self.undirected = False
                # Create negative train and validation edges for outer fold
                (
                    inner_neg_train_edges,
                    _,
                    inner_neg_val_edges,
                    _,
                    _,
                    _,
                ) = self.train_val_test_edge_split(
                    neg_train_edges,
                    None,
                    inner_val_ratio,
                    0.0,
                    num_nodes=num_nodes,
                )
                self.undirected = tmp_undirected

                inner_fold = InnerFold(
                    train_idxs=(
                        inner_pos_train_edges.tolist(),
                        inner_train_attr,
                        inner_neg_train_edges.tolist(),
                    ),
                    val_idxs=(
                        inner_pos_val_edges.tolist(),
                        inner_val_attr,
                        inner_neg_val_edges.tolist(),
                    ),
                )
                inner_fold_splits.append(inner_fold)

            self.inner_folds.append(inner_fold_splits)

            outer_fold = OuterFold(
                train_idxs=(
                    pos_train_edges.tolist(),
                    train_attr,
                    neg_train_edges.tolist(),
                ),
                val_idxs=(
                    outer_pos_val_edges.tolist(),
                    outer_val_attr,
                    outer_neg_val_edges.tolist(),
                ),
                test_idxs=(
                    pos_test_edges.tolist(),
                    test_attr,
                    neg_test_edges.tolist(),
                ),
            )
            self.outer_folds.append(outer_fold)

    def _splitter_args(self):
        r"""
        Compared to the superclass version, adds two boolean arguments
        `undirected` and `avoid_opposite_negative_edges`.

        Returns:
            a dict containing all splitter's arguments.
        """
        splitter_args = super()._splitter_args()
        splitter_args.update(
            {
                "undirected": self.undirected,
                "avoid_opposite_negative_edges": self.avoid_opposite_negative_edges,
            }
        )
        return splitter_args
