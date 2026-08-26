# The investigation agent

How a case with deterministic evidence becomes a written, cited assessment —
and what stops the language model from writing anything the system did not
find.

## The shape of the thing

```
load_case → collect_evidence → build_query → retrieve
          → synthesize → validate ──┬── retry once ──┐
                                    ├── fallback ────┤
                                    └────────────────┴→ persist
```

Seven nodes. Six are database work. The model is called exactly once, at
`synthesize`, and receives evidence that has already been gathered and
passages that have already been retrieved.

That is a deliberate trade against the more fashionable design. A tool-calling
loop that let the model decide what to look up would demo better and be worse:
non-reproducible, more expensive, harder to test, and — the point that
matters — it would give the model an opportunity to introduce facts nobody
measured. The interesting engineering here is in what the model is *prevented*
from doing.

## What the model is not allowed to do

| | Who decides |
|---|---|
| Which transactions are risky | XGBoost (Phase 2/3) |
| What evidence exists | The deterministic tools (Phase 3) |
| Which typology passages are relevant | Pattern filter + vector rank |
| Confidence | `argus.agent.confidence`, from the deterministic evidence |
| Which cases enter the queue | XGBoost's ranking and the alert budget — nothing else |
| **The wording of the assessment** | **The model** |

The last row is the whole of its job.

The state object enforces this structurally rather than by convention.
`DeterministicEvidence` and `RetrievedKnowledge` are **frozen dataclasses**, so
a generation step physically cannot overwrite a measurement — assignment
raises. A test asserts it.

The prompt also forbids the model from stating a confidence of its own. Two
numbers competing for the analyst's attention is worse than one, and the one
that should win is the one computed from evidence.

## The corpus

Thirteen curated notes in `data/typologies/`, drawn from FATF, FinCEN, the
Egmont Group, Europol, UNODC and the Wolfsberg Group.

Each note is a **paraphrase written for this project**, summarising the
typology its cited source describes. Deliberately not verbatim extracts: the
corpus ships with the repository, and every note names the document an analyst
should read for authoritative wording. That the corpus is authored at build
time is not in tension with "typology language is retrieved, never generated" —
the rule is about what happens at *runtime*, where the model quotes retrieved
text rather than explaining a pattern from its own knowledge.

Frontmatter carries the citation; `##` sections become chunks.

```yaml
id: structuring-smurfing
title: Structuring and smurfing
publisher: FATF
source_url: https://...
document: "Professional Money Laundering"
year: 2018
patterns: [structuring, placement]
```

Chunking is one chunk per section — not a sliding window, not a token counter.
The notes are written to be chunk-shaped, so a retrieved passage is a whole
thought rather than an arbitrary window, which is what makes it quotable.
27 chunks from 13 sources.

## Retrieval: filter, then rank

The pattern tags a case actually exhibited are a **hard filter**. Only chunks
carrying one of them are candidates; cosine similarity then orders that set.

The order matters. Ranking first and filtering afterwards would let a
semantically close but topically wrong passage win. Filtering first is why a
fan-out can never be explained by citing the funnelling note.

| Evidence | Retrieves |
|---|---|
| `fan_out` | `structuring` |
| `fan_in` | `funnelling` |
| `layering_chain`, `relay_chain`, `dense_cluster` | `layering` |
| `flagged_neighbour`, `confirmed_neighbour` | `network_association` |
| **`structural_similarity`** | **`behavioural_similarity`** |
| **`graph_model_corroboration`** | **`model_risk_scoring`** |

The bottom two rows carry almost all the traffic, and they exist because of a
Phase 3 measurement: the queue is almost entirely degree-1 transactions, so the
degree-based heuristics rarely fire. Keying retrieval only off heuristics —
the obvious design — would have left most cases citing nothing at all.

A related correction: an early distance cutoff was silently returning **zero**
chunks for `network_association`, because that phrase never appears in the
prose. Relevance is the filter's job, not the vector's, so the cutoff now drops
only anti-correlated chunks.

`DISTINCT ON` by source means four results are four documents, not four
sections of one.

## Embeddings, with and without a key

| `LLM_PROVIDER` | Embedder | Narrative |
|---|---|---|
| `gemini` | `gemini-embedding-001`, 768-d | `gemini-2.5-flash`, structured output |
| `stub` | deterministic hashing vectoriser | rule-built template |

The stub is not a mock returning noise. It is a hashing vectoriser: tokens hash
into dimensions, the vector is L2-normalised, and passages sharing vocabulary
genuinely land near each other. A test asserts that related text scores higher
than unrelated text.

It works because retrieval filters before it ranks — a weak ranking over a
correct candidate set still returns correct citations. Similarity scores are
low (0.08–0.26) under the stub and will be much higher with Gemini; the
*ordering* is what is used.

Corpus and query vectors must come from the same model or the distances are
meaningless, so `typology_references.embedding_model` records what produced
each row and retrieval **refuses a mismatch** rather than returning confident
nonsense. Change provider, re-ingest.

## Citation validation

Not hallucination detection — there is no general way to check whether a
sentence is true. It checks the one thing that is checkable: that every id
cited exists in what the model was given.

Two id spaces, because they mean different things:

- `evidence_ids` must name an `evidence_items` row belonging to this case;
- `source_ids` must name a chunk retrieval actually returned.

Plus three rules that close the obvious gaps: every claim must cite something;
`typology_assessment` and `recommended_action` must come from closed lists; and
**a typology may not be asserted without citing a retrieved source** — that
last one is the substitution the corpus exists to prevent.

A rejected response is retried **once**, with the specific errors appended to
the prompt. If it fails again, the rule-built narrative is stored instead.
Nothing unsupported ever reaches the database.

The fallback is not an error path. It is what runs whenever there is no API
key, so it is exercised on every keyless run rather than rotting until needed.

## Confidence

```
per kind:   score = 1 - Π(1 - strengthᵢ)      (noisy-OR)
            contribution = weight × score      (≤ the kind's weight)
confidence = Σ contributions, clipped to 1
```

Weights, version `w1`:

```
confirmed_neighbour       0.40
structural_similarity     0.25
flagged_neighbour         0.20
heuristic                 0.15
graph_model_corroboration 0.00   a model's own score, not evidence
typology_reference        0.00   explains a signal; is not one
```

The first real run exposed why simple summation was wrong: five near-identical
similarity matches summed past 1.0, and essentially every case came out at full
confidence, which is the same as having no confidence score at all. Five
matches into the same cluster of known-illicit transactions are **one
observation held more firmly**, not five findings.

Noisy-OR per kind says exactly that: more items of a kind raise its score with
diminishing returns and can never take it past the kind's weight. Confidence
therefore rises by finding *different* kinds of support. Corroboration across a
heuristic, a neighbour and a similarity match beats five similarity matches —
which is the behaviour worth having.

Two kinds weigh **zero** on purpose, for two different reasons.

A **typology reference** explains why a signal matters; it is not itself a
signal, and a system that grew more confident the more it read would be
measuring its own reading.

**`graph_model_corroboration`** is GraphSAGE's own probability for the
transaction. Folding one model's score into "how much evidence is there" makes
the two indistinguishable, and a case could then look well-supported on nothing
but a second model agreeing with the first.

It is not evidence at all, and the product says so in one place rather than
five: `agent.evidence.OBSERVED_KINDS` holds it out, and everything that counts
or lists evidence reads that set — the overview chart, the queue's evidence
column, `GET /cases/{id}/evidence`, and the ids a narrative is allowed to cite
(`DeterministicEvidence.evidence_ids`). It reaches the case page as
`graph_score`, one of the three signals.

The row still exists. It carries provenance, and typology retrieval keys off it
(`model_risk_scoring`). The prompt still states the score as a fact about the
transaction, so a report can say the models agree — it just cannot cite an id
the evidence list does not show.

The distinction that has to survive: **`structural_similarity` also comes from
GraphSAGE and still counts.** It is a measurement made *using* the embeddings —
this transaction sits near these named, historically-confirmed illicit ones —
not the model's opinion about this transaction. A test pins the difference by
sweeping the raw graph score across its whole range and asserting confidence
does not move.

Confidence decides nothing. It describes the evidence and nothing else; queue
membership comes from XGBoost's ranking and the alert budget alone.

Confidence is computed as soon as the deterministic evidence is persisted, at
the end of `replay_batch` — before any investigation runs. Pressing *Run
investigation* adds retrieval and a narrative; it does not compute the score,
and re-running it leaves the score unchanged unless the evidence itself
changed.

## Persistence

Everything goes into the existing tables. No second report store.

| Field | Column |
|---|---|
| Summary and claims | `case_reports.narrative` |
| Whether a model wrote it | `case_reports.narrative_source` (`llm` / `template`) |
| Typology, suggested action | `typology_assessment`, `recommended_action` |
| Confidence and its scheme | `confidence`, `confidence_version` |
| Provider, model, attempts, retrieved ids, errors | `investigation_meta` (JSONB) |
| Citations | `evidence_items` (kind `typology_reference`, FK to `typology_references`) |

Only typology evidence is replaced on a re-run. **Phase 3's deterministic
evidence is never touched**, and a test compares every non-typology row before
and after to prove it.

Citations are foreign keys, so a claim in a report always resolves to the
corpus row it came from. A test asserts no stored citation dangles.

## Presenting it honestly

The case panel shows three separately labelled blocks, each captioned with
where it came from:

- **Observed evidence** — measured by Argus.
- **AI interpretation** — written by a model from those facts, in a tinted
  panel, footnoted *"Interpretation, not a finding."* When there is no key it
  is relabelled *Generated summary* and says it was built by rule.
- **External typology sources** — retrieved and quoted verbatim, with
  publisher, document, year and a link.

A reader should never have to guess which of the three they are looking at.

## Running it

```bash
python -m argus.agent.cli ingest-corpus
python -m argus.agent.cli corpus-status
python -m argus.agent.cli retrieve behavioural_similarity --k 3
python -m argus.agent.cli investigate 1

curl -X POST http://localhost:8000/api/cases/1/investigate   # 202
curl http://localhost:8000/api/cases/1
curl http://localhost:8000/api/cases/1/sources
```

For real narratives, get a key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and set
`LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`, then **re-ingest the
corpus** so the vectors come from the same model as the queries.
