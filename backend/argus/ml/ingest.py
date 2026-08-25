"""Load the parsed dataset into PostgreSQL.

Uses COPY rather than INSERT: 203,769 rows each carrying a 165-element float
array is about 34 million numbers, and row-at-a-time inserts take minutes
where COPY takes seconds.

Stored per transaction: the id, its time step, its label, its 165 model
features, and its directed in/out degree. The degrees are derived from
`edges`, but the structural heuristics in Phase 4 read them for every queued
transaction, and recomputing a degree from the edge table on each read is
wasted work -- this is the one denormalisation in the schema.

The time step is deliberately **not** inside the `features` array. It has its
own indexed column, and keeping it out of the array means a model literally
cannot be trained on it by accident.
"""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.ml.dataset import EllipticData

log = logging.getLogger(__name__)

COPY_BATCH = 10_000


def _degrees(data: EllipticData) -> tuple[np.ndarray, np.ndarray]:
    """Directed out- and in-degree per node, from the raw (directed) edges."""
    src, dst = data.edge_index
    out_degree = np.bincount(src, minlength=data.n_nodes).astype(np.int32)
    in_degree = np.bincount(dst, minlength=data.n_nodes).astype(np.int32)
    return in_degree, out_degree


def _format_array(row: np.ndarray) -> str:
    """PostgreSQL array literal for the COPY text format."""
    return "{" + ",".join(f"{v:.7g}" for v in row) + "}"


def ingest(session: Session, data: EllipticData, *, truncate: bool = True) -> dict[str, int]:
    """Populate `transactions` and `edges`. Idempotent when `truncate` is set."""
    connection = session.connection().connection  # raw psycopg connection
    in_degree, out_degree = _degrees(data)

    if truncate:
        # RESTART IDENTITY so re-ingesting does not leave sequence gaps;
        # CASCADE because scores and case reports reference transactions.
        session.execute(text("TRUNCATE transactions, edges RESTART IDENTITY CASCADE"))
        log.info("truncated transactions and edges")

    with connection.cursor() as cursor:
        with cursor.copy(
            "COPY transactions (tx_id, timestep, label, features, in_degree, out_degree) FROM STDIN"
        ) as copy:
            for start in range(0, data.n_nodes, COPY_BATCH):
                stop = min(start + COPY_BATCH, data.n_nodes)
                chunk = "".join(
                    f"{data.tx_ids[i]}\t{data.timesteps[i]}\t{data.labels[i]}\t"
                    f"{_format_array(data.features[i])}\t{in_degree[i]}\t{out_degree[i]}\n"
                    for i in range(start, stop)
                )
                copy.write(chunk)
        log.info("copied %d transactions", data.n_nodes)

        src, dst = data.edge_index
        with cursor.copy("COPY edges (src_tx_id, dst_tx_id) FROM STDIN") as copy:
            for start in range(0, data.n_edges, COPY_BATCH):
                stop = min(start + COPY_BATCH, data.n_edges)
                copy.write(
                    "".join(
                        f"{data.tx_ids[src[i]]}\t{data.tx_ids[dst[i]]}\n"
                        for i in range(start, stop)
                    )
                )
        log.info("copied %d edges", data.n_edges)

    session.commit()
    return {"transactions": data.n_nodes, "edges": data.n_edges}


def write_embeddings(
    session: Session,
    tx_ids: np.ndarray,
    embeddings: np.ndarray,
    model_version: str,
) -> int:
    """Upsert GraphSAGE embeddings into the pgvector table.

    One row per transaction: retraining replaces the vector rather than
    versioning it, so similarity search never has to filter by model version.
    """
    if len(tx_ids) != len(embeddings):
        raise ValueError("tx_ids and embeddings must be the same length")

    connection = session.connection().connection
    with connection.cursor() as cursor:
        # Only the three columns being copied. `LIKE transaction_embeddings`
        # would inherit created_at's NOT NULL without its DEFAULT.
        cursor.execute(
            "CREATE TEMP TABLE _emb ("
            "tx_id bigint, model_version varchar(120), embedding vector(64)"
            ") ON COMMIT DROP"
        )
        with cursor.copy("COPY _emb (tx_id, model_version, embedding) FROM STDIN") as copy:
            for start in range(0, len(tx_ids), COPY_BATCH):
                stop = min(start + COPY_BATCH, len(tx_ids))
                copy.write(
                    "".join(
                        f"{tx_ids[i]}\t{model_version}\t"
                        + "["
                        + ",".join(f"{v:.6g}" for v in embeddings[i])
                        + "]\n"
                        for i in range(start, stop)
                    )
                )
        cursor.execute(
            "INSERT INTO transaction_embeddings (tx_id, model_version, embedding) "
            "SELECT tx_id, model_version, embedding FROM _emb "
            "ON CONFLICT (tx_id) DO UPDATE SET "
            "model_version = EXCLUDED.model_version, embedding = EXCLUDED.embedding"
        )

    session.commit()
    log.info("wrote %d embeddings (%s)", len(tx_ids), model_version)
    return len(tx_ids)
