"""Retrieval over the typology corpus.

Filter, then rank.

The pattern tags a case actually exhibited are a hard filter: only chunks
tagged with one of them are candidates. Cosine similarity then orders that
candidate set. This ordering matters -- ranking first and filtering afterwards
would let a semantically close but topically wrong passage win, and it is the
reason a fan-out can never be explained by citing the funnelling note.

Retrieval is a deterministic function of the case's evidence. The language
model does not choose what to retrieve and does not get to retrieve again; it
receives what this returns and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.agent.embeddings import Embedder, get_embedder
from argus.agent.state import EvidenceRecord, RetrievedChunk, RetrievedKnowledge

log = logging.getLogger(__name__)

DEFAULT_K = 4
PER_SOURCE_LIMIT = 1
# Only anti-correlated chunks are dropped. Relevance is decided by the pattern
# filter, not by distance -- a chunk tagged `network_association` *is* the
# right note for a flagged-neighbour case whatever the cosine says. A tighter
# cutoff silently returned nothing for that pattern, because the tag itself
# never appears in the prose, and the case ended up citing no source at all.
MAX_COSINE_DISTANCE = 1.0

# Which corpus pattern each kind of evidence justifies retrieving.
#
# The two entries at the bottom carry most of the traffic. Phase 3 measured
# that the queue is almost entirely degree-1 transactions, so the degree-based
# heuristics rarely fire; keying retrieval only off heuristics would leave most
# cases with no typology reference at all.
EVIDENCE_TO_PATTERNS: dict[str, list[str]] = {
    "flagged_neighbour": ["network_association"],
    "confirmed_neighbour": ["network_association"],
    "structural_similarity": ["behavioural_similarity"],
    "graph_model_corroboration": ["model_risk_scoring"],
}

# Heuristics name their own pattern in `details`, so they map by that.
HEURISTIC_TO_PATTERNS: dict[str, list[str]] = {
    "fan_out": ["structuring"],
    "fan_in": ["funnelling"],
    "layering_chain": ["layering"],
    "relay_chain": ["layering"],
    "dense_cluster": ["layering"],
}


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    patterns: list[str]


def build_query(evidence: list[EvidenceRecord]) -> RetrievalQuery:
    """Turn assembled evidence into a retrieval query.

    Deterministic: the same evidence always produces the same patterns and the
    same query text, so a re-run retrieves the same passages.
    """
    patterns: list[str] = []
    phrases: list[str] = []

    for item in evidence:
        if item.kind == "heuristic":
            name = str((item.details or {}).get("heuristic", ""))
            mapped = HEURISTIC_TO_PATTERNS.get(name, [])
            phrases.append(name.replace("_", " "))
        else:
            mapped = EVIDENCE_TO_PATTERNS.get(item.kind, [])
            phrases.append(item.kind.replace("_", " "))
        for pattern in mapped:
            if pattern not in patterns:
                patterns.append(pattern)

    query = " ".join(
        [
            "money laundering typology",
            *dict.fromkeys(phrases),
            *patterns,
        ]
    ).strip()
    return RetrievalQuery(text=query, patterns=patterns)


def retrieve(
    session: Session,
    query: RetrievalQuery,
    k: int = DEFAULT_K,
    embedder: Embedder | None = None,
) -> RetrievedKnowledge:
    """Top-k typology chunks for a query, at most one per source."""
    if not query.patterns:
        log.info("no patterns to retrieve on; returning nothing")
        return RetrievedKnowledge(query=query.text, patterns=[])

    embedder = embedder or get_embedder()
    _assert_space_present(session, embedder)

    vector = embedder.embed([query.text])[0]
    literal = "[" + ",".join(f"{value:.9g}" for value in vector) + "]"

    # DISTINCT ON keeps the nearest chunk per source, so four results are four
    # different documents rather than four sections of the same one.
    rows = session.execute(
        text("""
        WITH ranked AS (
            SELECT r.id, r.typology_id, r.title, r.publisher, r.source_url,
                   r.document, r.year, r.section_heading, r.text, r.patterns,
                   r.embedding <=> CAST(:vector AS vector) AS distance,
                   ROW_NUMBER() OVER (
                       PARTITION BY r.typology_id
                       ORDER BY r.embedding <=> CAST(:vector AS vector)
                   ) AS rank_in_source
            FROM typology_references r
            WHERE r.patterns && CAST(:patterns AS text[])
              AND r.embedding IS NOT NULL
              -- Select the active embedding space. Several may be stored (the
              -- stub one the tests use, the real one a deployment uses); a
              -- query must only ever compare against vectors from its own.
              AND r.embedding_model = :embedding_model
        )
        SELECT * FROM ranked
        WHERE rank_in_source <= :per_source AND distance <= :max_distance
        ORDER BY distance
        LIMIT :k
        """),
        {
            "vector": literal,
            "patterns": query.patterns,
            "embedding_model": embedder.model_name,
            "per_source": PER_SOURCE_LIMIT,
            "max_distance": MAX_COSINE_DISTANCE,
            "k": k,
        },
    ).all()

    chunks = [
        RetrievedChunk(
            reference_id=int(row.id),
            typology_id=row.typology_id,
            title=row.title,
            publisher=row.publisher,
            source_url=row.source_url,
            document=row.document,
            year=int(row.year) if row.year is not None else None,
            section_heading=row.section_heading,
            text=row.text,
            patterns=list(row.patterns),
            similarity=1.0 - float(row.distance),
        )
        for row in rows
    ]
    log.info("retrieved %d chunk(s) for patterns %s", len(chunks), ", ".join(query.patterns))
    return RetrievedKnowledge(chunks=chunks, query=query.text, patterns=query.patterns)


def _assert_space_present(session: Session, embedder: Embedder) -> None:
    """Fail loudly when the active embedder has no corpus stored.

    Retrieval selects rows matching the active embedding space, so a mismatch
    can no longer return nonsense -- it returns nothing, which is quieter and
    just as wrong. This turns that silence into an error naming the fix.
    """
    stored = (
        session.execute(text("SELECT DISTINCT embedding_model FROM typology_references"))
        .scalars()
        .all()
    )
    stored = [model for model in stored if model]
    if stored and embedder.model_name not in stored:
        raise ValueError(
            f"no corpus embedded with {embedder.model_name!r}; stored spaces are "
            f"{stored}. Run `python -m argus.agent.cli ingest-corpus` with this "
            "provider active."
        )
