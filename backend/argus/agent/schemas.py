"""The structured response the language model must return.

Kept apart from the provider and the prompt so both can import it without a
cycle, and so the contract is readable in one place.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# A closed list. An open-ended one invites invention, and every value here is
# covered by the typology corpus.
TYPOLOGIES = [
    "structuring",
    "funnelling",
    "layering",
    "mixing_or_obfuscation",
    "mule_network",
    "network_association",
    "no_clear_typology",
]

RECOMMENDED_ACTIONS = ["escalate", "monitor", "dismiss"]


class Claim(BaseModel):
    """One sentence, and what it rests on.

    Splitting the assessment into claims rather than free prose is what makes
    citation checkable: each sentence carries its own references, so a single
    unsupported statement can be found instead of invalidating the whole text.
    """

    text: str = Field(description="A single sentence of the assessment.")
    evidence_ids: list[int] = Field(
        default_factory=list,
        description="Ids of Argus evidence items supporting this sentence.",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Ids of retrieved typology sources supporting this sentence.",
    )


class Narrative(BaseModel):
    summary: str = Field(description="Two or three sentences for the analyst.")
    typology_assessment: str = Field(
        description=f"The most consistent typology. One of: {', '.join(TYPOLOGIES)}."
    )
    typology_rationale: str = Field(description="Why that typology, citing retrieved source ids.")
    claims: list[Claim] = Field(default_factory=list)
    recommended_action: str = Field(description=f"One of: {', '.join(RECOMMENDED_ACTIONS)}.")
