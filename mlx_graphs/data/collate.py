import mlx.core as mx
import numpy as np

from mlx_graphs.data.data import GraphData
from mlx_graphs.data.utils import validate_list_of_graph_data


@validate_list_of_graph_data
def collate(graph_list: list[GraphData], pad: bool = False) -> dict:
    """Concatenates attributes of multiple graphs based on the specifications
    of each `GraphData`.

    By default, concatenates all default array attributes in dim 0
    apart from `edge_index` which is concatenated along dim 1.
    Each graph remains independent in the final graph by incrementing
    the indices in `edge_index` based on the cumsum of previous number
    of nodes per graph.

    When ``pad=True``, every graph in the batch is padded to the same
    number of nodes and edges **before** concatenation so that the
    resulting batch has uniform per-graph sizes. This makes the batch
    compatible with ``mx.compile`` which requires static shapes.
    Padded node rows are filled with zeros and padded edges are
    self-loops on node 0.  Boolean masks ``_node_mask`` and
    ``_edge_mask`` distinguish real from padded entries.

    Args:
        graph_list: List of `GraphData` objects to collate
        pad: If True, pad each graph to uniform node/edge counts
            before concatenation. Defaults to False.

    Returns:
        Dict containing all the attributes of the unified and disconnected big graph as
        well as the "private" attributes used behind the hood by the batching.
        These private attributes start with an underscore "_" and can be ignore by
        the user.
    """
    # ----------------------------------------------------------------
    # When padding is requested, pad each graph to uniform sizes first
    # ----------------------------------------------------------------
    if pad:
        max_nodes = max(g.num_nodes for g in graph_list)
        max_edges = max(g.num_edges for g in graph_list)

        padded_list: list[GraphData] = []
        node_mask_list: list[mx.array] = []
        edge_mask_list: list[mx.array] = []

        for graph in graph_list:
            pad_kwargs: dict = {}
            n = graph.num_nodes
            e = graph.num_edges
            pad_n = max_nodes - n
            pad_e = max_edges - e

            # Build per-graph masks (True = real, False = padded)
            node_mask_list.append(
                mx.concatenate(
                    [mx.ones(n, dtype=mx.bool_), mx.zeros(pad_n, dtype=mx.bool_)]
                )
                if pad_n > 0
                else mx.ones(n, dtype=mx.bool_)
            )
            edge_mask_list.append(
                mx.concatenate(
                    [mx.ones(e, dtype=mx.bool_), mx.zeros(pad_e, dtype=mx.bool_)]
                )
                if pad_e > 0
                else mx.ones(e, dtype=mx.bool_)
            )

            # Pad edge_index: dummy self-loops on node 0
            if pad_e > 0:
                dummy_edges = mx.zeros((2, pad_e), dtype=graph.edge_index.dtype)
                pad_kwargs["edge_index"] = mx.concatenate(
                    [graph.edge_index, dummy_edges], axis=1
                )
            else:
                pad_kwargs["edge_index"] = graph.edge_index

            # Iterate over all attributes (except edge_index already handled)
            for attr in graph.to_dict():
                if attr == "edge_index":
                    continue
                val = getattr(graph, attr)
                cat_dim = graph.__cat_dim__(key=attr)
                if "index" in attr:
                    # Index-like attributes: pad along dim 1 with zeros
                    if pad_e > 0:
                        pad_shape = list(val.shape)
                        pad_shape[cat_dim] = pad_e
                        pad_kwargs[attr] = mx.concatenate(
                            [val, mx.zeros(pad_shape, dtype=val.dtype)], axis=cat_dim
                        )
                    else:
                        pad_kwargs[attr] = val
                else:
                    # Node-level or edge-level features/labels
                    size_along_dim = val.shape[cat_dim]
                    # Heuristic: if size matches num_nodes, pad with pad_n;
                    # if size matches num_edges, pad with pad_e; else no pad
                    if size_along_dim == n and pad_n > 0:
                        pad_shape = list(val.shape)
                        pad_shape[cat_dim] = pad_n
                        pad_kwargs[attr] = mx.concatenate(
                            [val, mx.zeros(pad_shape, dtype=val.dtype)], axis=cat_dim
                        )
                    elif size_along_dim == e and pad_e > 0:
                        pad_shape = list(val.shape)
                        pad_shape[cat_dim] = pad_e
                        pad_kwargs[attr] = mx.concatenate(
                            [val, mx.zeros(pad_shape, dtype=val.dtype)], axis=cat_dim
                        )
                    else:
                        # Graph-level features or unknown: keep as-is
                        pad_kwargs[attr] = val

            padded_list.append(GraphData(**pad_kwargs))

        graph_list = padded_list

    # ----------------------------------------------------------------

    batch_attr_dict = {}

    # Pre-compute __inc__ and __cat_dim__ for all attributes outside the loop
    attrs_inc_cat_dim = [
        (attr, graph_list[0].__inc__(key=attr), graph_list[0].__cat_dim__(key=attr))
        for attr in graph_list[0].to_dict()
    ]

    # To store pre-computed cumsum for attributes where __inc__ is True
    cumsum_dict = {}

    for attr, __inc__, __cat_dim__ in attrs_inc_cat_dim:
        attr_list, attr_sizes = [], []

        # Compute cumsum outside the inner loop if __inc__ is True
        if __inc__ and attr not in cumsum_dict:
            num_attr_list = [
                getattr(graph, "__inc__")(attr) for graph in graph_list
            ]  # Assuming __num_nodes__ provides the needed value
            cumsum_dict[attr] = mx.cumsum(mx.array([0] + num_attr_list))

        cumsum = cumsum_dict.get(attr)

        for i, graph in enumerate(graph_list):
            attr_array = getattr(graph, attr)

            if __inc__:
                attr_array = attr_array + cumsum[i]  # type: ignore

            attr_list.append(attr_array)
            attr_sizes.append(attr_array.shape[__cat_dim__])

        # Concatenate all attributes at once outside the loop
        batch_attr_dict[attr] = mx.concatenate(attr_list, axis=__cat_dim__)
        batch_attr_dict[f"_size_{attr}"] = mx.array(attr_sizes)
        batch_attr_dict[f"_cat_dim_{attr}"] = __cat_dim__
        if __inc__:
            batch_attr_dict[f"_inc_{attr}"] = True
            batch_attr_dict[f"_cumsum_{attr}"] = cumsum

        # Special handling for "edge_index" to be optimized with vectorization
        if attr == "edge_index":
            cumsum = np.array(cumsum)
            batch_indices = np.hstack(
                [
                    np.full((cumsum[i + 1] - cumsum[i]).item(), i)
                    for i in range(len(cumsum) - 1)
                ]
            )
            batch_attr_dict["_batch_indices"] = mx.array(batch_indices)

    # Store padding masks if padding was applied
    if pad:
        batch_attr_dict["_node_mask"] = mx.concatenate(node_mask_list, axis=0)
        batch_attr_dict["_edge_mask"] = mx.concatenate(edge_mask_list, axis=0)
        batch_attr_dict["_padded"] = True

    return batch_attr_dict
