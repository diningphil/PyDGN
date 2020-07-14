import warnings
from pathlib import Path
import numpy as np
import os
import shutil

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from config.utils import s2c
from datasets.splitter import Splitter
from utils.utils import get_or_create_dir, check_argument

DEFAULTS = {
    "root": "DATA/",
    "splits": "SPLITS/",
    "inner_folds": 1,
    "outer_folds": 10,
    "seed": 42
}

def has_targets(dataset):
    return all(d.y is not None for d in dataset)

def get_targets(dataset):
    return np.array([d.y.item() for d in dataset])

def preprocess_data(options, splitter, delete_processed_data):
    kwargs = DEFAULTS.copy()
    kwargs.update(**options)

    if "class_name" not in kwargs:
        raise ValueError("You must specify 'class_name' in your kwargs.")

    data_root = Path(kwargs.pop("root"))
    splits_root = Path(kwargs.pop("splits"))
    outer_folds = kwargs.pop("outer_folds")
    inner_folds = kwargs.pop("inner_folds")
    seed = kwargs.pop("seed")
    stratify = kwargs.pop("stratify", False)
    other_kwargs = kwargs.pop('other_args', {})

    # Enables re-generation of the 'processed' folder
    if delete_processed_data:
        processed_folder = Path(data_root, kwargs['name'], 'processed')
        if os.path.exists(processed_folder):
            print(f"Deleting old processed data in {processed_folder}")
            shutil.rmtree(processed_folder)
        else:
            print(f"No old processed data to delete in {processed_folder}")

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
    kwargs_path = Path(data_root, dataset.name, 'processed', 'dataset_kwargs.pt')
    torch.save(kwargs, kwargs_path)

    splits_dir = get_or_create_dir(splits_root / dataset.name)
    splits_path = splits_dir / f"{dataset.name}_outer{outer_folds}_inner{inner_folds}.yml"

    if not splits_path.exists():
        targets = get_targets(dataset) if has_targets(dataset) else None
        splitter.split(range(len(dataset)), targets=targets)
        splitter.save(splits_path)
    else:
        print("Data splits are already present, I will not overwrite them.")


def load_dataset(data_root, dataset_name, dataset_class=TUDataset):
    data_root = Path(data_root)

    # Load arguments
    kwargs_path = Path(data_root, dataset_name, 'processed', 'dataset_kwargs.pt')
    kwargs = torch.load(kwargs_path)

    with warnings.catch_warnings():
        # suppress PyG warnings
        warnings.simplefilter("ignore")
        dataset = dataset_class(data_root, **kwargs)

    return dataset


def load_splitter(dataset_name, split_root, outer_folds, inner_folds):
    splits_dir = split_root / dataset_name
    splits_path = splits_dir / f"{dataset_name}_outer{outer_folds}_inner{inner_folds}.yml"
    return Splitter.load(splits_path)
