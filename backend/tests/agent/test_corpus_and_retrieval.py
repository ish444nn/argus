"""Corpus parsing, chunking, embeddings and retrieval."""

from __future__ import annotations

import pytest

from argus.agent import corpus
from argus.agent.embeddings import STUB_MODEL_NAME, StubEmbedder, get_embedder
from argus.core.config import Settings

# --------------------------------------------------------------------------
# Parsing and chunking
# --------------------------------------------------------------------------


def test_corpus_parses_and_every_file_has_required_metadata():
    chunks = corpus.load_corpus()

    assert chunks, "the corpus produced no chunks"
    for chunk in chunks:
        assert chunk.typology_id
        assert chunk.title
        assert chunk.publisher
        assert chunk.source_url.startswith("http")
        assert chunk.section_heading
        assert chunk.patterns, f"{chunk.typology_id} has no pattern tags"


def test_every_chunk_can_reconstruct_its_citation():
    """A retrieved passage is only useful if it can be attributed."""
    for chunk in corpus.load_corpus():
        assert chunk.publisher in _citation(chunk)


def _citation(chunk) -> str:
    parts = [chunk.publisher]
    if chunk.document:
        parts.append(chunk.document)
    if chunk.year:
        parts.append(str(chunk.year))
    return ", ".join(parts)


def test_chunking_is_deterministic():
    first = corpus.load_corpus()
    second = corpus.load_corpus()

    assert [c.typology_id for c in first] == [c.typology_id for c in second]
    assert [c.chunk_index for c in first] == [c.chunk_index for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunk_ids_are_unique_within_a_source():
    seen = set()
    for chunk in corpus.load_corpus():
        key = (chunk.typology_id, chunk.chunk_index)
        assert key not in seen, f"duplicate chunk {key}"
        seen.add(key)


def test_chunks_are_a_reasonable_size():
    """Too short to be worth citing, or too long to quote in a report."""
    for chunk in corpus.load_corpus():
        words = len(chunk.text.split())
        assert corpus.MIN_CHUNK_WORDS <= words <= 400, (
            f"{chunk.typology_id}/{chunk.chunk_index} has {words} words"
        )


def test_pattern_tags_come_from_the_known_vocabulary():
    """A typo in a tag silently makes a note unreachable."""
    known = {
        "structuring",
        "funnelling",
        "layering",
        "network_association",
        "behavioural_similarity",
        "model_risk_scoring",
        "placement",
        "integration",
        "virtual_assets",
    }
    for chunk in corpus.load_corpus():
        unknown = set(chunk.patterns) - known
        assert not unknown, f"{chunk.typology_id} uses unknown tags {unknown}"


def test_every_evidence_kind_can_retrieve_something():
    """The mapping is only useful if each pattern has a note behind it.

    Phase 3 measured that the queue is almost all degree-1 transactions, so
    `behavioural_similarity` and `model_risk_scoring` carry most cases. If
    either had no corpus coverage, most reports would cite nothing.
    """
    from argus.agent.retrieval import EVIDENCE_TO_PATTERNS, HEURISTIC_TO_PATTERNS

    covered = {pattern for chunk in corpus.load_corpus() for pattern in chunk.patterns}
    for mapping in (EVIDENCE_TO_PATTERNS, HEURISTIC_TO_PATTERNS):
        for source, patterns in mapping.items():
            for pattern in patterns:
                assert pattern in covered, f"{source} maps to uncovered {pattern!r}"


def test_frontmatter_is_required():
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "broken.md"
        path.write_text("## Section\n\nno frontmatter here", encoding="utf-8")
        with pytest.raises(ValueError, match="frontmatter"):
            corpus.parse_file(path)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def test_stub_embedder_returns_the_configured_dimension():
    embedder = StubEmbedder(dimension=768)
    vectors = embedder.embed(["structuring and smurfing", "layering chains"])

    assert len(vectors) == 2
    assert all(len(vector) == 768 for vector in vectors)


def test_stub_embeddings_are_deterministic():
    """Committed corpus vectors are only reproducible if this holds."""
    a = StubEmbedder().embed(["fan out structuring"])[0]
    b = StubEmbedder().embed(["fan out structuring"])[0]
    assert a == b


def test_stub_embeddings_are_normalised():
    vector = StubEmbedder().embed(["money laundering typology"])[0]
    norm = sum(value * value for value in vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_stub_embedder_carries_real_lexical_signal():
    """Not noise: shared vocabulary must actually move the vectors together.

    This is what makes the keyless path a weak embedding rather than a
    meaningless one.
    """
    embedder = StubEmbedder()
    a, b, c = embedder.embed(
        [
            "structuring divides a transfer into many smaller transfers",
            "structuring divides a payment into many smaller payments",
            "densely connected clusters of controlled counterparties",
        ]
    )
    related = sum(x * y for x, y in zip(a, b, strict=True))
    unrelated = sum(x * y for x, y in zip(a, c, strict=True))
    assert related > unrelated


def test_empty_text_does_not_divide_by_zero():
    assert StubEmbedder().embed([""])[0] == [0.0] * 768


def test_provider_selection_follows_configuration():
    stub = get_embedder(Settings(_env_file=None, llm_provider="stub"))
    assert stub.model_name == STUB_MODEL_NAME


def test_gemini_embedder_refuses_to_run_without_a_key():
    """No key must fail loudly, never fall through to an unusable client."""
    from argus.agent.embeddings import GeminiEmbedder

    settings = Settings(_env_file=None, llm_provider="stub")
    with pytest.raises((ValueError, ImportError)):
        GeminiEmbedder(settings)
