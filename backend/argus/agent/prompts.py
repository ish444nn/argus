"""Prompt construction, and the template narrative used as a fallback.

The prompt's job is to leave the model as little room to invent as possible.
It supplies two clearly separated blocks -- what Argus measured, and what was
retrieved from the corpus -- and states which kinds of claim may draw on which.
Nothing else is available to it.
"""

from __future__ import annotations

from argus.agent.schemas import RECOMMENDED_ACTIONS, TYPOLOGIES, Claim, Narrative
from argus.agent.state import DeterministicEvidence, RetrievedKnowledge

SYSTEM_RULES = """\
You are assisting a financial-crime analyst. You are summarising an
investigation that has already been carried out; you are not conducting one.

Rules, in order of importance:

1. Every statement about THIS transaction must come from the ARGUS EVIDENCE
   block. Cite the evidence item ids you used in `evidence_ids`.
2. Every statement about what a money-laundering typology is, or what a
   pattern usually means, must come from the RETRIEVED SOURCES block. Cite the
   source ids you used in `source_ids`.
3. Do not introduce transaction facts that are not in the evidence block. You
   do not know the amount, the time of day, the parties, or the purpose --
   none of that is in this dataset.
4. Do not introduce sources. If a claim is not supported by a retrieved
   source, do not make it.
5. Distinguish observation from interpretation. Write "consistent with" or
   "resembles", not "is". The analyst decides whether it is laundering; you
   describe what was found.
6. Do not state a confidence, score or probability of your own. Confidence is
   computed from the evidence elsewhere, and a number from you would compete
   with it.

Be concise. Two or three sentences of summary is right. An honest
`no_clear_typology` is a better answer than a stretched one.
"""


def _format_evidence(evidence: DeterministicEvidence) -> str:
    lines = [
        "ARGUS EVIDENCE (measured by the system; the only source for facts about this transaction)",
        "",
        f"Transaction {evidence.tx_id}, batch (time step) {evidence.timestep}.",
        f"Primary risk score {evidence.risk_score:.4f} from model "
        f"{evidence.model_version}"
        + (f", queue rank {evidence.queue_rank}" if evidence.queue_rank else "")
        + ".",
    ]
    if evidence.graph_score is not None:
        lines.append(
            f"Secondary graph-model score {evidence.graph_score:.4f} "
            "(a second opinion; it does not decide the queue)."
        )
    lines += [
        "",
        "Network position: "
        f"in-degree {evidence.in_degree}, out-degree {evidence.out_degree}, "
        f"{evidence.neighbour_count} distinct counterparties, "
        f"{evidence.flagged_neighbours} of them already flagged, "
        f"pass-through chain length {evidence.chain_length}.",
        "",
        "Evidence items:",
    ]
    if evidence.observed:
        for item in evidence.observed:
            source = f" [from transaction {item.neighbour_tx_id}]" if item.neighbour_tx_id else ""
            lines.append(
                f"  [id {item.id}] ({item.kind}, strength {item.strength:.2f})"
                f" {item.summary}{source}"
            )
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def _format_sources(retrieved: RetrievedKnowledge) -> str:
    lines = [
        "RETRIEVED SOURCES (the only source for typology claims)",
        "",
    ]
    if not retrieved.chunks:
        lines.append("  (none retrieved -- make no typology claims and answer no_clear_typology)")
        return "\n".join(lines)

    for chunk in retrieved.chunks:
        lines += [
            f"  [source {chunk.typology_id}] {chunk.title} -- {chunk.section_heading}",
            f"  {chunk.citation()}",
            f"  {chunk.text.strip()}",
            "",
        ]
    return "\n".join(lines)


def build_prompt(evidence: DeterministicEvidence, retrieved: RetrievedKnowledge) -> str:
    return "\n\n".join(
        [
            SYSTEM_RULES,
            _format_evidence(evidence),
            _format_sources(retrieved),
            (
                "Return JSON matching the schema. "
                f"`typology_assessment` must be one of: {', '.join(TYPOLOGIES)}. "
                f"`recommended_action` must be one of: "
                f"{', '.join(RECOMMENDED_ACTIONS)}. "
                "Every claim must cite at least one evidence id or source id."
            ),
        ]
    )


# Maps the pattern tag a retrieval fired on to the typology label, so the
# fallback picks the same answer the evidence points at rather than guessing.
_PATTERN_TO_TYPOLOGY = {
    "structuring": "structuring",
    "funnelling": "funnelling",
    "layering": "layering",
    "network_association": "network_association",
    "behavioural_similarity": "layering",
    "virtual_assets": "mixing_or_obfuscation",
}


def build_template_narrative(
    evidence: DeterministicEvidence, retrieved: RetrievedKnowledge
) -> Narrative:
    """A narrative assembled by rule rather than by a model.

    Used when there is no API key, and as the fallback when a real response
    fails citation validation. It says less than the model would, but
    everything it says is mechanically derived from evidence that exists, so
    it is always safe to publish.
    """
    claims: list[Claim] = []

    similarity = evidence.by_kind("structural_similarity")
    if similarity:
        strongest = max(similarity, key=lambda item: item.strength)
        claims.append(
            Claim(
                text=(
                    f"Its network behaviour resembles {len(similarity)} "
                    "transaction(s) confirmed illicit in the training period, "
                    f"most closely transaction {strongest.neighbour_tx_id}."
                ),
                evidence_ids=[item.id for item in similarity],
            )
        )

    heuristics = evidence.by_kind("heuristic")
    for item in heuristics:
        claims.append(Claim(text=item.summary, evidence_ids=[item.id]))

    neighbours = evidence.by_kind("flagged_neighbour") + evidence.by_kind("confirmed_neighbour")
    if neighbours:
        claims.append(
            Claim(
                text=(
                    f"{len(neighbours)} connected transaction(s) are already flagged or confirmed."
                ),
                evidence_ids=[item.id for item in neighbours],
            )
        )

    typology = "no_clear_typology"
    for pattern in retrieved.patterns:
        if pattern in _PATTERN_TO_TYPOLOGY:
            typology = _PATTERN_TO_TYPOLOGY[pattern]
            break

    rationale = ""
    if retrieved.chunks:
        chunk = retrieved.chunks[0]
        rationale = (
            f"{chunk.title} ({chunk.citation()}) describes this pattern "
            f"[source {chunk.typology_id}]."
        )

    summary = (
        f"Transaction {evidence.tx_id} was scored {evidence.risk_score:.3f} by "
        f"{evidence.model_version} and placed in the review queue"
        + (f" at rank {evidence.queue_rank}" if evidence.queue_rank else "")
        + f". {len(evidence.observed)} supporting evidence item(s) were assembled."
    )
    if not evidence.observed:
        summary += " No corroborating structural evidence was found."

    return Narrative(
        summary=summary,
        typology_assessment=typology,
        typology_rationale=rationale,
        claims=claims,
        recommended_action="escalate" if len(claims) >= 3 else "monitor",
    )
