"""Text embeddings for the typology corpus.

Two providers behind one function, chosen by `LLM_PROVIDER`:

`gemini`  the configured Gemini embedding model. Used when an API key is set.
`stub`    a deterministic hashing embedder, used when there is no key.

The stub is not a placeholder that returns noise. It is a hashing vectoriser:
tokens are hashed into dimensions and the vector is L2-normalised, so two
passages sharing vocabulary genuinely land near each other. That is a weak
embedding, but it is a real one, and it is reproducible without credentials --
which is what lets the corpus ship with committed vectors and the tests run
offline.

It works here because retrieval **filters before it ranks**. Only chunks tagged
with a pattern the case actually exhibited are candidates, so the vector never
has to decide whether a typology is relevant, only which of the relevant ones
to show first. A weak ranking over a correct candidate set still returns
correct citations.

Corpus and query vectors must come from the same provider or the distances mean
nothing, which is why `typology_references.embedding_model` records what
produced each row and retrieval refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

from argus.core.config import Settings, get_settings

log = logging.getLogger(__name__)

STUB_MODEL_NAME = "stub-hashing-v1"
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns text into vectors of a fixed width."""

    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class StubEmbedder:
    """Deterministic hashing vectoriser. No network, no credentials."""

    def __init__(self, dimension: int = 768):
        self.model_name = STUB_MODEL_NAME
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN.findall(text.lower())
        for token in tokens:
            # Two hashes per token: one picks the dimension, one the sign, so
            # unrelated tokens cancel instead of always accumulating.
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class GeminiEmbedder:
    """The configured Gemini embedding model."""

    def __init__(self, settings: Settings):
        from google import genai

        if settings.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY is required for the gemini embedder")
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        response = self._client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        vectors = [list(embedding.values) for embedding in response.embeddings]
        # Truncating a Matryoshka embedding leaves it un-normalised, and
        # cosine distance against a normalised corpus would then be skewed.
        return [_normalise(vector) for vector in vectors]


def _normalise(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else vector


def get_embedder(settings: Settings | None = None) -> Embedder:
    """The embedder the current configuration calls for."""
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        return GeminiEmbedder(settings)
    log.info("using the stub embedder; set LLM_PROVIDER=gemini for real embeddings")
    return StubEmbedder(dimension=settings.embedding_dim)
