import numpy as np
from pydgn.data.splitter import Splitter, InnerFold, OuterFold


class TemporalSplitter(Splitter):
    """
    Class that generates the splits at dataset creation time.
    """

    def get_graph_targets(self, dataset):
        try:
            targets = np.array([d.targets[-1].item() for d in dataset])
            return True, targets
        except Exception:
            return False, None

class SingleGraphSequenceSplitter(Splitter):
    """
    Class for dynamic graphs that generates the splits at dataset creation time.
    """

    def __init__(self, train_timesteps, valid_timesteps, test_timesteps, seed=None, **kwargs):
        """
        Initializes the splitter
        :param train_timesteps: num of training timesteps to use
        :param valid_timesteps: num of validation timesteps to use
        :param test_timesteps: num of test timesteps to use
        :param seed: random seed for reproducibility on the same machine (unused in this case)
        """
        self.outer_folds = []
        self.inner_folds = []

        # these are fixed in most cases
        self.n_outer_folds = 1
        self.n_inner_folds = 1

        self.train_timesteps = train_timesteps
        self.valid_timesteps = valid_timesteps
        self.test_timesteps = test_timesteps

        self.processed = False
        self.seed = seed

    def split(self, dataset, targets=None):
        """
        Computes the splits. The outer split does not include validation (can be extracted from the training set if needed)
        :param dataset: the Dataset object
        :param targets: targets used for stratification
        :param test_ratio: percentage of validation/test set when using an internal/external hold-out split. Default value is 0.1.
        :return:
        """
        # this is the number of timesteps in the dataset
        idxs = range(len(dataset))

        inner_fold_splits = []

        valid_range = (self.train_timesteps, self.train_timesteps+self.valid_timesteps)
        test_range = (valid_range[1], valid_range[1]+self.test_timesteps)

        if not self.processed:

            inner_fold = InnerFold(train_idxs=list(range(self.train_timesteps)),
                                   val_idxs=list(range(valid_range[0], valid_range[1])))
            inner_fold_splits.append(inner_fold)
            self.inner_folds.append(inner_fold_splits)

            outer_fold = OuterFold(train_idxs=list(range(self.train_timesteps)),
                                   val_idxs=list(range(valid_range[0], valid_range[1])),
                                   test_idxs=list(range(test_range[0], test_range[1])))

            self.outer_folds.append(outer_fold)

            self.processed = True

    def _splitter_args(self):
        return {
            "seed": self.seed,
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds": self.n_inner_folds,
            "train_timesteps": self.train_timesteps,
            "valid_timesteps": self.valid_timesteps,
            "test_timesteps": self.test_timesteps
        }
