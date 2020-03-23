import warnings
from pathlib import Path
import numpy as np

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from config.utils import s2c
from datasets.splitter import Splitter
from utils.utils import get_or_create_dir, check_argument


DATA_DIR = Path("DATA")
SPLITS_DIR = Path("SPLITS")

DEFAULTS = {
    "root": "DATA",
    "inner_folds": 1,
    "outer_folds": 10,
    "seed": 42
}


def has_targets(dataset):
    return all(d.y is not None for d in dataset)


def get_targets(dataset):
    return np.array([d.y.item() for d in dataset])


def preprocess_data(options):
    kwargs = DEFAULTS.copy()
    kwargs.update(**options)

    if "class_name" not in kwargs:
        raise ValueError("You must specify 'class_name' in your kwargs.")

    data_root = Path(kwargs.pop("root"))
    outer_folds = kwargs.pop("outer_folds")
    inner_folds = kwargs.pop("inner_folds")
    seed = kwargs.pop("seed")
    stratify = kwargs.pop("stratify", False)
    other_kwargs = kwargs.pop('other_args', {})

    dataset_class = s2c(kwargs.pop("class_name"))

    pre_transforms = None
    pre_transforms_opt = kwargs.pop("pre_transform", None)
    if pre_transforms_opt is not None:
        pre_transforms = []
        for pre_transform in pre_transforms_opt:
            pre_transform_class = s2c(pre_transform["class_name"])
            args = pre_transform.pop("args", {})
            pre_transforms.append(pre_transform_class(**args))
        kwargs.update(pre_transform=Compose(pre_transforms))

    pre_filters = None
    pre_filters_opt = kwargs.pop("pre_filter", None)
    if pre_filters_opt is not None and check_argument(dataset_class, "pre_filter"):
        pre_filters = []
        for pre_filter in pre_filters_opt:
            pre_filter_class = s2c(pre_filter["class_name"])
            args = pre_filter.pop("args", {})
            pre_filters.append(pre_filter_class(**args))
        kwargs.update(pre_filter=Compose(pre_filters))

    transforms = None
    transforms_opt = kwargs.pop("transforms", None)
    if transforms_opt is not None:
        transforms = []
        for transform in transforms_opt:
            transform_class = s2c(transform["class_name"])
            args = transform.pop("args", {})
            transforms.append(transform_class(**args))
        kwargs.update(transform=Compose(transforms))

    kwargs.update(other_kwargs)

    dataset = dataset_class(root=data_root, **kwargs)

    # Store dataset arguments
    kwargs_path = Path(data_root, kwargs['name'], 'processed', 'dataset_kwargs.pt')
    torch.save(kwargs, kwargs_path)

    splits_dir = get_or_create_dir(SPLITS_DIR / dataset.name)
    splits_path = splits_dir / f"{dataset.name}_outer{outer_folds}_inner{inner_folds}.yml"

    if not splits_path.exists():
        splitter = Splitter(
            n_outer_folds=outer_folds,
            n_inner_folds=inner_folds,
            seed=seed,
            stratify=stratify)
        targets = get_targets(dataset) if has_targets(dataset) else None
        splitter.split(range(len(dataset)), targets=targets)
        splitter.save(splits_path)


def load_dataset(data_root, dataset_name, dataset_class=TUDataset):
    data_root = Path(data_root)

    # Load arguments
    kwargs_path = Path(data_root, dataset_name, 'processed', 'dataset_kwargs.pt')
    kwargs = torch.load(kwargs_path)
    assert kwargs.pop('name') == dataset_name

    with warnings.catch_warnings():
        # suppress PyG warnings
        warnings.simplefilter("ignore")

        dataset = dataset_class(data_root, dataset_name, **kwargs)

    # TODO ADD SUPPORT FOR TRANSFORM AND PRE-FILTERS
    # patch for PyG and pre_transform filters
    # data, slices = torch.load(data_root / dataset_name / "processed" / "data.pt")
    # dataset.data = data
    # dataset.slices = slices
    return dataset


def load_splitter(dataset_name, outer_folds, inner_folds):
    splits_dir = SPLITS_DIR / dataset_name
    splits_path = splits_dir / f"{dataset_name}_outer{outer_folds}_inner{inner_folds}.yml"
    return Splitter.load(splits_path)
