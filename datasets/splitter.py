import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, KFold

from utils.serialization import load_yaml, save_yaml


class Fold:
    """
    Simple class that stores training and validation/test indices
    """
    def __init__(self, train_idxs=None, test_idxs=None):
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs  # this can also be used for validation sets during model selection


class InnerFold(Fold):
    """
    Simple extension of the Fold class that returns a dictionary with training and validation indices (model selection)
    """
    def todict(self):
        return {"train": self.train_idxs, "val": self.test_idxs}


class OuterFold(Fold):
    """
    Simple extension of the Fold class that returns a dictionary with training and test indices (risk assessment)
    """
    def todict(self):
        return {"train": self.train_idxs, "test": self.test_idxs}


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
        splits = load_yaml(path)

        n_outer_folds = len(splits["outer_folds"])
        n_inner_folds = len(splits["inner_folds"])
        seed = splits["seed"]

        obj = cls(n_outer_folds=n_outer_folds, n_inner_folds=n_inner_folds, seed=seed)

        for fold_data in splits["outer_folds"]:
            obj.outer_folds.append(OuterFold(fold_data["train"], fold_data["test"]))

        for inner_split in splits["inner_folds"]:
            inner_split_data = []
            for fold_data in inner_split:
                inner_split_data.append(InnerFold(fold_data["train"], fold_data["val"]))
            obj.inner_folds.append(inner_split_data)

        return obj

    def __init__(self, n_outer_folds=5, n_inner_folds=5, seed=42, stratify=True):
        """
        Initializes the splitter
        :param n_outer_folds: number of outer folds (risk assessment). 1 means hold-out, >1 means k-fold
        :param n_inner_folds: number of inner folds (model selection). 1 means hold-out, >1 means k-fold
        :param seed: random seed for reproducibility (on the same machine)
        :param stratify: whether to apply stratification or not (should be true for classification tasks)
        """
        self.outer_folds = []
        self.inner_folds = []
        self.processed = False
        self.stratify = stratify

        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.seed = seed

    def _get_splitter(self, n_splits, stratified, test_size, shuffle):
        if n_splits == 1:
            if stratified:
                splitter = StratifiedShuffleSplit(n_splits, test_size=test_size, random_state=self.seed)
            else:
                splitter = ShuffleSplit(n_splits, test_size=test_size, random_state=self.seed)
        elif n_splits > 1:
            if stratified:
                splitter = StratifiedKFold(n_splits, shuffle=shuffle, random_state=self.seed)
            else:
                splitter = KFold(n_splits, shuffle=shuffle, random_state=self.seed)
        else:
            raise ValueError(f"'n_splits' must be >=1, got {n_splits}")

        return splitter

    def split(self, idxs, targets=None, test_size=0.2, shuffle=True):
        """
        Computes the splits
        :param idxs: the indices of the dataset
        :param targets: targets used for stratification
        :param test_size: percentage of test set when using an hold-out split. Default value is 0.2.
        :param shuffle: whether to shuffle indices
        :return:
        """
        if not self.processed:

            stratified = self.stratify
            outer_idxs = np.array(idxs)

            outer_splitter = self._get_splitter(
                n_splits=self.n_outer_folds,
                stratified=stratified,
                test_size=test_size,
                shuffle=shuffle)

            for train_idxs, test_idxs in outer_splitter.split(outer_idxs, y=targets):
                inner_splitter = self._get_splitter(
                    n_splits=self.n_inner_folds,
                    stratified=stratified,
                    test_size=test_size,
                    shuffle=shuffle)

                inner_fold_splits = []
                inner_idxs = outer_idxs[train_idxs]
                inner_targets = targets[train_idxs] if targets is not None else None

                for inner_train_idxs, inner_val_idxs in inner_splitter.split(inner_idxs, y=inner_targets):
                    inner_fold = InnerFold(inner_idxs[inner_train_idxs].tolist(), inner_idxs[inner_val_idxs].tolist())
                    inner_fold_splits.append(inner_fold)
                self.inner_folds.append(inner_fold_splits)

                # train and test indices are returned in order. Even when splits are stratified, this can be a problem
                # e.g. they are ordered by size of the graph etc.
                np.random.shuffle(train_idxs)
                np.random.shuffle(test_idxs)
                outer_fold = OuterFold(train_idxs.tolist(), test_idxs.tolist())
                self.outer_folds.append(outer_fold)

            self.processed = True

    def save(self, path):
        """
        Saves the split into a yaml file
        :param path: filepath where to save the object
        """
        savedict = {"seed": self.seed}
        savedict["outer_folds"] = [o.todict() for o in self.outer_folds]
        savedict["inner_folds"] = []
        for inner_split in self.inner_folds:
            savedict["inner_folds"].append([i.todict() for i in inner_split])
        save_yaml(savedict, path)
