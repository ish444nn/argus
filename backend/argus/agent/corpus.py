"""Ingesting the typology corpus into pgvector.

    data/typologies/*.md  ->  frontmatter + sections  ->  chunks
                          ->  embeddings  ->  typology_references

Chunking is one chunk per `##` section. Not a sliding window, not a token
counter: the corpus is hand-written to be chunk-shaped, each section covers one
idea, and a deterministic rule means the same file always produces the same
chunks with the same ids. A retrieved passage is therefore a whole thought
rather than an arbitrary window, which is what makes it quotable in a report.

No document-ingestion framework is involved. This is a directory of files the
project controls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from argus.agent.embeddings import Embedder, get_embedder
from argus.db.enums import EvidenceKind
from argus.db.models import TypologyReference

log = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parents[3] / "data" / "typologies"

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

MIN_CHUNK_WORDS = 20


@dataclass(frozen=True)
class Chunk:
    typology_id: str
    title: str
    publisher: str
    source_url: str
    document: str | None
    year: int | None
    patterns: list[str]
    section_heading: str
    chunk_index: int
    text: str


def _parse_frontmatter(raw: str) -> dict[str, object]:
    """A tiny YAML subset: scalars and inline lists.

    Deliberately not a YAML dependency. The frontmatter this project writes is
    five known keys and a list, and a parser that accepts exactly that fails
    loudly on anything unexpected instead of silently accepting it.
    """
    data: dict[str, object] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"malformed frontmatter line: {line!r}")
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            data[key] = [
                part.strip().strip("\"'") for part in value[1:-1].split(",") if part.strip()
            ]
        else:
            data[key] = value.strip("\"'")
    return data


def parse_file(path: Path) -> list[Chunk]:
    """Split one corpus file into its section chunks."""
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER.match(raw)
    if not match:
        raise ValueError(f"{path.name} has no frontmatter")

    meta = _parse_frontmatter(match.group(1))
    for required in ("id", "title", "publisher", "source_url", "patterns"):
        if required not in meta:
            raise ValueError(f"{path.name} frontmatter is missing {required!r}")

    body = raw[match.end() :]
    headings = list(_SECTION.finditer(body))
    if not headings:
        raise ValueError(f"{path.name} has no '##' sections to chunk")

    year = meta.get("year")
    chunks: list[Chunk] = []
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        text = body[start:end].strip()
        if len(text.split()) < MIN_CHUNK_WORDS:
            log.warning(
                "%s section %r is too short to retrieve; skipping", path.name, heading.group(1)
            )
            continue
        chunks.append(
            Chunk(
                typology_id=str(meta["id"]),
                title=str(meta["title"]),
                publisher=str(meta["publisher"]),
                source_url=str(meta["source_url"]),
                document=str(meta["document"]) if meta.get("document") else None,
                year=int(year) if year else None,
                patterns=list(meta["patterns"]),  # type: ignore[arg-type]
                section_heading=heading.group(1),
                chunk_index=index,
                text=text,
            )
        )
    return chunks


def load_corpus(directory: Path | None = None) -> list[Chunk]:
    """Every chunk in the corpus, in a stable order."""
    directory = directory or CORPUS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"no typology corpus at {directory}")

    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(parse_file(path))

    if not chunks:
        raise ValueError(f"{directory} produced no chunks")
    return chunks


def embed_text(chunk: Chunk) -> str:
    """What actually gets embedded.

    The heading and title are prepended to the body: they carry most of the
    topical signal in a short passage, and including them measurably improves
    ranking under the hashing embedder without changing the stored text.
    """
    return f"{chunk.title}. {chunk.section_heading}. {chunk.text}"


def ingest(
    session: Session,
    directory: Path | None = None,
    embedder: Embedder | None = None,
) -> dict[str, int]:
    """Load, chunk, embed and store the corpus for the active embedder.

    Replaces only this embedder's rows. Two embedding spaces coexist -- the
    stub one the tests use and the Gemini one a key enables -- and
    re-ingesting either leaves the other intact.

    That isolation is the point. Before it, running the test suite (which
    forces the stub provider) silently replaced a Gemini-embedded corpus, and
    the next real investigation was correctly refused by the embedding-space
    guard. Making the embedder part of a chunk's identity fixes the collision
    at its source rather than weakening the guard.

    Within one space the replace is still wholesale: the corpus is small and a
    half-updated one would be worse than a rebuilt one.
    """
    embedder = embedder or get_embedder()
    chunks = load_corpus(directory)
    vectors = embedder.embed([embed_text(chunk) for chunk in chunks])

    if len(vectors) != len(chunks):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    for vector in vectors:
        if len(vector) != embedder.dimension:
            raise ValueError(
                f"embedder returned {len(vector)} dimensions, expected {embedder.dimension}"
            )

    session.execute(
        delete(TypologyReference).where(TypologyReference.embedding_model == embedder.model_name)
    )
    session.add_all(
        TypologyReference(
            typology_id=chunk.typology_id,
            title=chunk.title,
            publisher=chunk.publisher,
            source_url=chunk.source_url,
            document=chunk.document,
            year=chunk.year,
            section_heading=chunk.section_heading,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            patterns=chunk.patterns,
            embedding=vector,
            embedding_model=embedder.model_name,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
    # Replacing this space renumbers its rows, so citations into it are left
    # dangling -- the foreign key sets them to NULL rather than failing. A
    # report citing a source that cannot be resolved is worse than one citing
    # nothing, so those are removed and the affected cases counted: they need
    # re-investigating against the rebuilt corpus. Citations into *other*
    # embedding spaces are untouched, which is what keeps a test run from
    # destroying a real demo state.
    orphaned = (
        session.execute(
            text("""
        DELETE FROM evidence_items
        WHERE kind = :kind AND typology_reference_id IS NULL
        RETURNING case_report_id
        """),
            {"kind": EvidenceKind.TYPOLOGY_REFERENCE.value},
        )
        .scalars()
        .all()
    )
    session.commit()

    if orphaned:
        log.warning(
            "removed %d citation(s) across %d case(s) left dangling by the "
            "re-ingest; those cases should be re-investigated",
            len(orphaned),
            len(set(orphaned)),
        )

    sources = len({chunk.typology_id for chunk in chunks})
    log.info(
        "ingested %d chunks from %d sources using %s",
        len(chunks),
        sources,
        embedder.model_name,
    )
    return {
        "chunks": len(chunks),
        "sources": sources,
        "dimension": embedder.dimension,
        "embedding_model": embedder.model_name,
        "orphaned_citations_removed": len(orphaned),
    }


def corpus_status(session: Session) -> dict[str, object]:
    """What is currently stored, and which embedder produced it."""
    rows = session.execute(
        select(
            TypologyReference.embedding_model,
            TypologyReference.typology_id,
        )
    ).all()
    models = {row[0] for row in rows}
    return {
        "chunks": len(rows),
        "sources": len({row[1] for row in rows}),
        "embedding_models": sorted(model for model in models if model),
    }
