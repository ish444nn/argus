# AML typology corpus

Twelve curated reference notes on money-laundering typologies, used to ground
the language of Argus case reports. When a heuristic or a similarity match
fires, the investigation retrieves the matching passage from here and cites it,
rather than letting the model explain the pattern from its own knowledge.

## What these files are

Each note is a **paraphrase written for this project**, summarising the
typology described by the cited public source. They are deliberately not
verbatim extracts: the point is a small, uniformly structured corpus that can
be redistributed with the repository, and every note names the document an
analyst should read for the authoritative wording.

Sources are public guidance from FATF, FinCEN, the Egmont Group, Europol,
UNODC and the Wolfsberg Group. `source_url` points at the publisher's page for
the document; the exact URLs are worth re-checking, since these bodies
reorganise their sites periodically.

## Format

YAML frontmatter plus `##` sections. One section becomes one retrievable chunk.

```yaml
---
id: structuring-smurfing        # stable slug, also the citation key
title: ...
publisher: FATF
source_url: https://...
document: "Report title, as published"
year: 2020
patterns: [structuring, placement]   # what fires this typology
---
```

`patterns` is the hard filter applied before ranking: retrieval only ever
considers chunks tagged with a pattern the case actually exhibited, so an
unrelated typology cannot be returned for a matched signal.

## Pattern vocabulary

| Tag | Fired by |
|---|---|
| `structuring` | `fan_out` heuristic |
| `funnelling` | `fan_in` heuristic |
| `layering` | `layering_chain`, `relay_chain`, `dense_cluster` |
| `network_association` | `flagged_neighbour`, `confirmed_neighbour` |
| `behavioural_similarity` | `structural_similarity` |
| `model_risk_scoring` | `graph_model_corroboration` |
| `placement`, `integration`, `virtual_assets` | general context |

`behavioural_similarity` and `model_risk_scoring` matter more than they look:
the queue is almost entirely degree-1 transactions, so the degree-based
heuristics rarely fire and those two tags are what most cases actually
retrieve on.
