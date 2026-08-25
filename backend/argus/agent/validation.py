"""Citation validation.

Deliberately small. This is not hallucination detection -- there is no general
way to check whether a sentence is true. It checks the one thing that *is*
checkable: that every id the model cited exists in what it was given.

Two id spaces, because they mean different things:

`evidence_ids`  must name an `evidence_items` row belonging to this case.
`source_ids`    must name a typology chunk that retrieval actually returned.

A response citing anything else is rejected. The caller retries once, then
falls back to the rule-built narrative. Nothing unsupported is ever stored.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.agent.schemas import RECOMMENDED_ACTIONS, TYPOLOGIES, Narrative


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]

    def as_feedback(self) -> str:
        """A correction to append to the prompt on retry."""
        return (
            "Your previous response was rejected:\n"
            + "\n".join(f"- {error}" for error in self.errors)
            + "\nCite only the ids supplied above."
        )


def validate(
    narrative: Narrative,
    allowed_evidence_ids: set[int],
    allowed_source_ids: set[str],
) -> ValidationResult:
    errors: list[str] = []

    if narrative.typology_assessment not in TYPOLOGIES:
        errors.append(
            f"typology_assessment {narrative.typology_assessment!r} is not one of {TYPOLOGIES}"
        )
    if narrative.recommended_action not in RECOMMENDED_ACTIONS:
        errors.append(
            f"recommended_action {narrative.recommended_action!r} is not one of "
            f"{RECOMMENDED_ACTIONS}"
        )
    if not narrative.summary.strip():
        errors.append("summary is empty")

    for index, claim in enumerate(narrative.claims):
        invented_evidence = set(claim.evidence_ids) - allowed_evidence_ids
        if invented_evidence:
            errors.append(
                f"claim {index} cites evidence ids {sorted(invented_evidence)} "
                "that were not supplied"
            )
        invented_sources = set(claim.source_ids) - allowed_source_ids
        if invented_sources:
            errors.append(
                f"claim {index} cites sources {sorted(invented_sources)} that were not retrieved"
            )
        if not claim.evidence_ids and not claim.source_ids:
            errors.append(f"claim {index} cites nothing")

    # A typology claim has to rest on a retrieved passage. Without this the
    # model could name a typology and explain it from its own knowledge, which
    # is exactly the substitution the corpus exists to prevent.
    if narrative.typology_assessment != "no_clear_typology":
        cited = {source for claim in narrative.claims for source in claim.source_ids}
        if not cited and not _mentions_a_source(narrative, allowed_source_ids):
            errors.append("a typology was asserted without citing any retrieved source")

    return ValidationResult(ok=not errors, errors=errors)


def _mentions_a_source(narrative: Narrative, allowed: set[str]) -> bool:
    """Allow the rationale to carry the citation instead of a claim."""
    return any(source in narrative.typology_rationale for source in allowed)
