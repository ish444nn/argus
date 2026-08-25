"""The language model boundary.

One method, two implementations. Deliberately not a generic multi-provider
abstraction: the system makes exactly one kind of call -- turn assembled
evidence into a short, cited narrative -- so the interface is that call and
nothing else.

`GeminiProvider`  the real model, using structured output so the response
                  arrives as fields rather than prose to be parsed.
`StubProvider`    returns nothing on its own; the caller falls back to the
                  rule-built narrative in `argus.agent.prompts`. Tests and a
                  keyless clone use it.

What the model may do is narrow by construction. It receives evidence that has
already been gathered and retrieved, and may only reference the ids in front of
it. It never gathers evidence, never scores anything, and never sets the
confidence -- that is computed from the evidence in `argus.agent.confidence`.
"""

from __future__ import annotations

import logging
from typing import Protocol

from argus.agent.schemas import Narrative
from argus.core.config import Settings, get_settings

log = logging.getLogger(__name__)

PROMPT_VERSION = "p1"


class LLMProvider(Protocol):
    name: str
    model: str

    def narrate(self, prompt: str) -> Narrative | None:
        """Return a structured narrative, or None to use the template."""
        ...


class StubProvider:
    """No model. The caller uses the rule-built narrative instead.

    Returning None rather than a canned string keeps one fallback path in the
    system: the same template that covers a validation failure also covers
    having no credentials, so the keyless path is exercised by every run.
    """

    name = "stub"
    model = "template"

    def narrate(self, prompt: str) -> Narrative | None:
        return None


class GeminiProvider:
    """Gemini with a response schema, so the output is fields, not prose."""

    name = "gemini"

    def __init__(self, settings: Settings):
        from google import genai

        if settings.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY is required for the gemini provider")
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        self.model = settings.gemini_model

    def narrate(self, prompt: str) -> Narrative | None:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Narrative,
                # Low but not zero: the task is summarising supplied facts, and
                # nothing is gained from creative phrasing.
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        parsed = response.parsed
        if isinstance(parsed, Narrative):
            return parsed
        # `parsed` is None when the model returns malformed JSON. Try the raw
        # text once, and let validation handle whatever comes out.
        return Narrative.model_validate_json(response.text)


def get_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings)
    return StubProvider()
