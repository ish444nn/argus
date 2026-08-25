"""Graph construction for PyTorch Geometric.

Edge semantics, verified against the raw file rather than assumed:

* Edges are directed, transaction -> transaction, representing a flow of
  funds. There are 234,355 of them, with no duplicates and no self-loops.
* **Every edge connects two transactions in the same time step. There are
  zero cross-time-step edges.** Measured, not inferred.

That last fact decides most of this module. Because the graph is already
partitioned by time step, the subgraph induced on the training time steps
cannot reach a validation or test node -- temporal leakage through graph
connectivity is structurally impossible here, not merely guarded against.
It also means each split's induced subgraph is the complete neighbourhood
context that exists for its nodes: nothing is being withheld at inference.

Direction
---------
Message passing runs on the **undirected** view: each edge is added in both
directions. A transaction is informative about its counterparties whichever
way the funds moved, and SAGEConv aggregates over incoming edges only, so on
the directed graph a node would see its senders but never its recipients.
The directed edges are still stored verbatim in Postgres, where in- and
out-degree are meaningful for the structural heuristics.

Self-loops
----------
Not added. `SAGEConv` has a separate root weight that transforms the node's
own features before combining them with the aggregated neighbourhood, so a
self-loop would double-count the node and blur the distinction the operator
is built to preserve.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import Data

from argus.db.enums import Label
from argus.ml.dataset import EllipticData

log = logging.getLogger(__name__)


def to_undirected(edge_index: np.ndarray) -> np.ndarray:
    """Add the reverse of every edge, then drop duplicates."""
    both = np.hstack([edge_index, edge_index[::-1]])
    # Unique over columns, so a pair present in both directions collapses.
    _, keep = np.unique(both.T, axis=0, return_index=True)
    return both[:, np.sort(keep)]


def induced_subgraph(
    edge_index: np.ndarray, node_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the graph to `node_mask`, renumbering nodes from zero.

    Returns the new edge index and the original row indices of the kept
    nodes, so results can be mapped back to transactions.
    """
    kept = np.nonzero(node_mask)[0]
    remap = np.full(node_mask.shape[0], -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)

    src, dst = edge_index
    inside = node_mask[src] & node_mask[dst]
    return np.vstack([remap[src[inside]], remap[dst[inside]]]), kept


def build_split_graph(
    data: EllipticData,
    features: np.ndarray,
    node_mask: np.ndarray,
) -> tuple[Data, np.ndarray]:
    """Build the PyG graph for one temporal split.

    `unknown`-labelled nodes are kept as message-passing neighbours: their
    features are legitimately observable and carry no label information. They
    are marked in `supervised_mask` as excluded from the loss.
    """
    edge_index, kept = induced_subgraph(to_undirected(data.edge_index), node_mask)

    labels = data.labels[kept]
    supervised = labels != Label.UNKNOWN.value
    y = torch.from_numpy((labels == Label.ILLICIT.value).astype(np.float32))

    graph = Data(
        x=torch.from_numpy(features[kept]),
        edge_index=torch.from_numpy(edge_index),
        y=y,
        supervised_mask=torch.from_numpy(supervised),
        num_nodes=len(kept),
    )
    log.info(
        "split graph: %d nodes (%d supervised, %d illicit), %d directed edges",
        len(kept),
        int(supervised.sum()),
        int(y[torch.from_numpy(supervised)].sum()),
        edge_index.shape[1],
    )
    return graph, kept


def count_cross_timestep_edges(data: EllipticData) -> int:
    """Number of edges joining two different time steps.

    Expected to be zero for Elliptic. Asserted in the test suite so a future
    dataset change cannot silently invalidate the leakage argument above.
    """
    src, dst = data.edge_index
    return int(np.sum(data.timesteps[src] != data.timesteps[dst]))
