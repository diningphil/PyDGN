import os
from typing import Tuple, Optional, List

import torch
import torch_geometric.data.batch
from pydgn.static import *


def atomic_save(data: dict, filepath: str):
    r"""
    Atomically stores a dictionary that can be serialized by :func:`torch.save`,
    exploiting the atomic :func:`os.replace`.

    Args:
        data (dict): the dictionary to be stored
        filepath (str): the absolute filepath where to store the dictionary
    """
    try:
        tmp_path = str(filepath) + ATOMIC_SAVE_EXTENSION
        torch.save(data, tmp_path)
        os.replace(tmp_path, filepath)
    except Exception as e:
        os.remove(tmp_path)
        raise e


def extend_lists(data_list: Optional[Tuple[Optional[List[torch.Tensor]]]],
                 new_data_list: Tuple[Optional[List[torch.Tensor]]]) -> Tuple[Optional[List[torch.Tensor]]]:
    r"""
    Extends the semantic of Python :func:`extend()` over lists to tuples
    Used e.g., to concatenate results of mini-batches in incremental architectures such as :obj:`CGMM`

    Args:
        data_list: tuple of lists, or ``None`` if there is no list to extend.
        new_data_list: object of the same form of :obj:`data_list` that has to be concatenated

    Returns:
        the tuple of extended lists
    """
    if data_list is None:
        return new_data_list

    assert len(data_list) == len(new_data_list)

    for i in range(len(data_list)):
        if new_data_list[i] is not None:
            data_list[i].extend(new_data_list[i])

    return data_list


def to_tensor_lists(embeddings: Tuple[Optional[torch.Tensor]],
                    batch: torch_geometric.data.batch.Batch,
                    edge_index: torch.Tensor) -> Tuple[Optional[List[torch.Tensor]]]:
    r"""
    Reverts batched outputs back to a list of Tensors elements.
    Can be useful to build incremental architectures such as :obj:`CGMM` that store intermediate results
    before training the next layer.

    Args:
        embeddings (tuple): a tuple of embeddings :obj:`(vertex_output, edge_output, graph_output, vertex_extra_output, edge_extra_output, graph_extra_output)`.
                            Each embedding can be a :class:`torch.Tensor` or ``None``.
        batch (:class:`torch_geometric.data.batch.Batch`): Batch information used to split the tensors.

        edge_index (:class:`torch.Tensor`): a :obj:`2 x num_edges` tensor as defined in Pytorch Geometric.
                                            Used to split edge Tensors graph-wise.

    Returns:
        a tuple with the same semantics as the argument ``embeddings``, but this time each element holds a list of
        Tensors, one for each graph in the dataset.
    """
    # Crucial: Detach the embeddings to free the computation graph!!
    # TODO this code can surely be made more compact, but leave it as is until future refactoring or removal from PyDGN.
    v_out, e_out, g_out, vo_out, eo_out, go_out = embeddings

    v_out = v_out.detach() if v_out is not None else None
    v_out_list = [] if v_out is not None else None

    e_out = e_out.detach() if e_out is not None else None
    e_out_list = [] if e_out is not None else None

    g_out = g_out.detach() if g_out is not None else None
    g_out_list = [] if g_out is not None else None

    vo_out = vo_out.detach() if vo_out is not None else None
    vo_out_list = [] if vo_out is not None else None

    eo_out = eo_out.detach() if eo_out is not None else None
    eo_out_list = [] if eo_out is not None else None

    go_out = go_out.detach() if go_out is not None else None
    go_out_list = [] if go_out is not None else None

    _, node_counts = torch.unique_consecutive(batch, return_counts=True)
    node_cumulative = torch.cumsum(node_counts, dim=0)

    if e_out is not None or eo_out is not None:
        edge_batch = batch[edge_index[0]]
        _, edge_counts = torch.unique_consecutive(edge_batch, return_counts=True)
        edge_cumulative = torch.cumsum(edge_counts, dim=0)

    if v_out_list is not None:
        v_out_list.append(v_out[:node_cumulative[0]])

    if e_out_list is not None:
        e_out_list.append(e_out[:edge_cumulative[0]])

    if g_out_list is not None:
        g_out_list.append(g_out[0].unsqueeze(0))  # recreate batch dimension by unsqueezing

    if vo_out_list is not None:
        vo_out_list.append(vo_out[:node_cumulative[0]])

    if eo_out_list is not None:
        eo_out_list.append(eo_out[:edge_cumulative[0]])

    if go_out_list is not None:
        go_out_list.append(go_out[0].unsqueeze(0))  # recreate batch dimension by unsqueezing

    for i in range(1, len(node_cumulative)):
        if v_out_list is not None:
            v_out_list.append(v_out[node_cumulative[i - 1]:node_cumulative[i]])

        if e_out_list is not None:
            e_out_list.append(e_out[edge_cumulative[i - 1]:edge_cumulative[i]])

        if g_out_list is not None:
            g_out_list.append(g_out[i].unsqueeze(0))  # recreate batch dimension by unsqueezing

        if vo_out_list is not None:
            vo_out_list.append(vo_out[node_cumulative[i - 1]:node_cumulative[i]])

        if eo_out_list is not None:
            eo_out_list.append(eo_out[edge_cumulative[i - 1]:edge_cumulative[i]])

        if go_out_list is not None:
            go_out_list.append(go_out[i].unsqueeze(0))  # recreate batch dimension by unsqueezing

    return v_out_list, e_out_list, g_out_list, vo_out_list, eo_out_list, go_out_list
