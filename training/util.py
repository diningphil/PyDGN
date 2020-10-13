import os
import torch
from torch_geometric.data import Data


def atomic_save(checkpoint, filepath):
    try:
        tmp_path = str(filepath) + ".part"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)
    except Exception as e:
        os.remove(tmp_path)


def to_data_list(x, batch, y, single_graph_link_prediction=False):
    """
    Converts a graphs outputs back to a list of Tensors elements. Useful for incremental architectures.
    :param embeddings: a tuple of embeddings: (vertex_output, edge_output, graph_output, other_output). Each of
    the elements should be a Tensor.
    :param x: big Tensor holding information of different graphs
    :param batch: the usual batch list provided by Pytorch Geometric. Used to split Tensors graph-wise.
    :param y: target labels Tensor, used to determine whether the task is graph classification or not (to be changed)
    :param single_graph_link_prediction: to be refactored
    :return: a list of PyTorch Geometric Data objects
    """
    data_list = []

    if not single_graph_link_prediction:
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        cumulative = torch.cumsum(counts, dim=0)

        is_graph_classification = y.shape[0] == len(cumulative)

        y = y.unsqueeze(1) if y.dim() == 1 else y

        data_list.append(Data(x=x[:cumulative[0]],
                              y=y[0] if is_graph_classification else y[:,cumulative[0]]))
        for i in range(1, len(cumulative)):

            g = Data(x=x[cumulative[i-1]:cumulative[i]],
                     y=y[i] if is_graph_classification else y[cumulative[i-1]:cumulative[i]])
            data_list.append(g)
    else:
        # TODO refactor this with a different function in the engine
        # Return node embeddings and their original class, if any (or dumb value which is required nonetheless)
        y = y[0]
        data_list.append(Data(x=x, y=y[0]))

    return data_list


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


def to_tensor_lists(embeddings, batch, edge_index, y):
    """
    Converts a graphs outputs back to a list of Tensors elements. Useful for incremental architectures.
    :param embeddings: a tuple of embeddings: (vertex_output, edge_output, graph_output, other_output). Each of
    the elements should be a Tensor.
    :param batch: the usual batch list provided by Pytorch Geometric. Used to split node Tensors graph-wise.
    :param edge_index: the usual edge_index tensor provided by Pytorch Geometric. Used to split edge Tensors graph-wise.
    :param y: unused parameter with target labels Tensor, which might become useful later
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

    if e_out is not None:
        edge_batch = batch[edge_index[0]]
        _, edge_counts = torch.unique_consecutive(edge_batch, return_counts=True)
        edge_cumulative = torch.cumsum(edge_counts, dim=0)

    # is_graph_classification = y.shape[0] == len(cumulative)
    # y = y.unsqueeze(1) if y.dim() == 1 else y

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
