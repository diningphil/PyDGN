import os

import torch

from pydgn.static import *


def atomic_save(checkpoint, filepath):
    try:
        tmp_path = str(filepath) + ATOMIC_SAVE_EXTENSION
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)
    except Exception as e:
        os.remove(tmp_path)


def extend_lists(data_list, embeddings):
    """
    Takes a tuple of lists of Tensors (one for each graph) and extends it with another tuple of the same time.
    Used to concatenate results of mini-batches in incremental architectures.
    :param data_list: tuple where each element holds a list of Tensors, one for each graph. The first element of the
    tuple contains a list of Tensors, each containing vertex outputs for a specific graph. Similarly, the second
    deals with edges, the third with graphs, the fourth with arbitrary vertex information and the last one
    with arbitrary edge information.
    :param embeddings: object of the same form of data list that has to be "concatenated" to data_list
    :return: the extended data_list tuple
    """
    if data_list is None:
        data_list = embeddings
        return data_list

    v_out_list, e_out_list, g_out_list, vo_out_list, eo_out_list, go_out_list = embeddings

    if v_out_list is not None:
        data_list[0].extend(v_out_list)

    if e_out_list is not None:
        data_list[1].extend(e_out_list)

    if g_out_list is not None:
        data_list[2].extend(g_out_list)

    if vo_out_list is not None:
        data_list[3].extend(vo_out_list)

    if eo_out_list is not None:
        data_list[4].extend(eo_out_list)

    if go_out_list is not None:
        data_list[5].extend(go_out_list)

    return data_list


def to_tensor_lists(embeddings, batch, edge_index):
    """
    Converts a graphs outputs back to a list of Tensors elements. Useful for incremental architectures.
    :param embeddings: a tuple of embeddings: (vertex_output, edge_output, graph_output, other_output). Each of
    the elements should be a Tensor.
    :param batch: the usual batch list provided by Pytorch Geometric. Used to split node Tensors graph-wise.
    :param edge_index: the usual edge_index tensor provided by Pytorch Geometric. Used to split edge Tensors graph-wise.
    :return: a tuple where each elements holds a list of Tensors, one for each graph in the dataset.  The semantic
    of each element is the same of the parameter embeddings.
    """
    # Crucial: Detach the embeddings to free the computation graph!!
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
