import inspect
import os
import os.path as osp
import warnings

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from pydgn.experiment.util import s2c


def get_or_create_dir(path):
    """ Creates directories associated to the specified path if they are missing, and it returns the path string """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def check_argument(cls, arg_name):
    """ Checks whether arg_name is in the signature of a method or class """
    sign = inspect.signature(cls)
    return arg_name in sign.parameters.keys()


# Adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html
def filter_adj(edge_index, edge_attr, mask):
    row, col = edge_index
    filtered_edge_index = row[mask], col[mask]
    return filtered_edge_index, None if edge_attr is None else edge_attr[mask]


def get_graph_targets(dataset):
    try:
        targets = np.array([d.y.item() for d in dataset])
        return True, targets
    except Exception:
        return False, None


def preprocess_data(options):
    data_info = options.pop("dataset")
    if "class_name" not in data_info:
        raise ValueError("You must specify 'class_name' in your dataset.")
    dataset_class = s2c(data_info.pop("class_name"))
    dataset_args = data_info.pop("args")
    data_root = data_info.pop("root")

    ################################

    # more experimental stuff here

    dataset_kwargs = data_info.pop('other_args', {})

    pre_transforms = None
    pre_transforms_opt = data_info.pop("pre_transform", None)
    if pre_transforms_opt is not None:
        pre_transforms = []
        for pre_transform in pre_transforms_opt:
            pre_transform_class = s2c(pre_transform["class_name"])
            args = pre_transform.pop("args", {})
            pre_transforms.append(pre_transform_class(**args))
        dataset_kwargs.update(pre_transform=Compose(pre_transforms))

    pre_filters = None
    pre_filters_opt = data_info.pop("pre_filter", None)
    if pre_filters_opt is not None and check_argument(dataset_class, "pre_filter"):
        pre_filters = []
        for pre_filter in pre_filters_opt:
            pre_filter_class = s2c(pre_filter["class_name"])
            args = pre_filter.pop("args", {})
            pre_filters.append(pre_filter_class(**args))
        dataset_kwargs.update(pre_filter=Compose(pre_filters))

    transforms_opt = data_info.pop("transform", None)

    # Backward compatibility with 0.5.0
    if transforms_opt is None:
        transforms_opt = data_info.pop("transforms", None)

    if transforms_opt is not None:
        transforms = []
        for transform in transforms_opt:
            transform_class = s2c(transform["class_name"])
            args = transform.pop("args", {})
            transforms.append(transform_class(**args))
        dataset_kwargs.update(transform=Compose(transforms))

    dataset_args.update(dataset_kwargs)

    ################################

    dataset = dataset_class(**dataset_args)
    assert hasattr(dataset, 'name'), "Dataset instance should have a name attribute!"

    # Store dataset additional arguments in a separate file
    kwargs_path = osp.join(data_root, dataset.name, 'processed', 'dataset_kwargs.pt')
    torch.save(dataset_args, kwargs_path)

    # Process data splits

    splits_info = options.pop("splitter")
    splits_root = splits_info.pop("root")
    if "class_name" not in splits_info:
        raise ValueError("You must specify 'class_name' in your splitter.")
    splitter_class = s2c(splits_info.pop("class_name"))
    splitter_args = splits_info.pop("args")
    splitter = splitter_class(**splitter_args)

    splits_dir = get_or_create_dir(osp.join(splits_root, dataset.name))
    splits_path = osp.join(splits_dir,
                           f"{dataset.name}_outer{splitter.n_outer_folds}_inner{splitter.n_inner_folds}.splits")

    if not os.path.exists(splits_path):
        has_targets, targets = get_graph_targets(dataset)
        # The splitter is in charge of eventual stratifications
        splitter.split(dataset, targets=targets if has_targets else None)
        splitter.save(splits_path)
    else:
        print("Data splits are already present, I will not overwrite them.")


def load_dataset(data_root, dataset_name, dataset_class=TUDataset):
    # Load arguments
    kwargs_path = osp.join(data_root, dataset_name, 'processed', 'dataset_kwargs.pt')
    dataset_args = torch.load(kwargs_path)

    # Overwrite original data_root field, which may have changed
    dataset_args['root'] = data_root

    with warnings.catch_warnings():
        # suppress PyG warnings
        warnings.simplefilter("ignore")
        dataset = dataset_class(**dataset_args)

    return dataset
