# Argus — Product Requirements

*Risk assessment and evidence-grounded investigation for financial transaction
networks: graph-aware scoring paired with an agent that gathers, cites and
scores the evidence behind every case it surfaces.*

---

## 1. Overview

Argus scores transactions in a financial transaction graph for likely illicit
activity, selects the highest-risk slice for human review, and assembles a
cited case report for each one.

The product exists because transaction risk is relational and a bare score is
not actionable. A transaction connected to confirmed illicit activity is
meaningfully riskier than an identical-looking transaction with no such
connections, and that signal is invisible to a model that sees a single row of
features. Equally, an accurate score still leaves an analyst to trace
connections and assemble context by hand for every flagged case, which does
not scale.

Argus addresses both: it uses network structure in scoring and evaluates
honestly whether that helps, and it turns each alert into a report whose every
claim points at the specific transaction, measurement or published passage it
rests on.

## 2. Problem

- Tabular models cannot see a transaction's position in the network.
- A bare risk score provides no evidence and no explanation.
- Reviewing every flagged transaction by hand does not scale with volume.
- Class imbalance makes raw accuracy a misleading measure of usefulness.
- A language model asked to explain a case will invent plausible reasoning that
  the underlying data does not support.

## 3. Goals

| Goal | Measure |
|---|---|
| Test whether network structure adds signal | A graph model and a tabular baseline evaluated on identical held-out time steps, against a promotion rule fixed before the test split is read |
| Turn a score into something reviewable | Every surfaced case carries cited, specific evidence rather than a number |
| Keep interpretation answerable to measurement | Every sentence a language model writes cites an evidence item that exists, or the output is rejected |
| Evaluate honestly under imbalance | Precision, recall and AUPRC on the illicit class; accuracy alone never justifies a model |

### Non-goals

Argus does not connect to a live blockchain or banking feed, take autonomous
action on an account, or attempt real-world identity resolution. It operates on
a static labelled transaction graph replayed in batches. It is not a compliance
system and makes no legal or regulatory claim about the transactions it scores.

## 4. Users

**Analysts** are the only user role. An analyst reviews the queue of flagged
transactions, opens a case to see the model's score alongside the gathered
evidence, and records a decision: confirmed, dismissed, or needs more evidence.
Ingestion and scoring run as background jobs rather than through a user-facing
flow.

Because there is one role and no data one analyst may see that another may not,
authentication would establish only *who* recorded a decision. Decisions are
therefore attributed to a single seeded analyst account, and the interface says
so where they are recorded rather than implying a signed-in user. Sign-in is a
self-contained addition to the review path when a second analyst exists to
distinguish.

## 5. Core workflows

### Replay (system)

1. A batch — one time step of the transaction graph — is scored by the primary
   risk model.
2. The batch is ranked by score and the top *k* becomes the alert queue, where
   *k* is `ceil(batch_size × alert_budget)`.
3. Each alert becomes a case, and every deterministic evidence tool runs
   against it.
4. Evidence confidence is computed from what those tools found, and stored.

Replay is idempotent: re-running a batch restates its rows rather than
appending. Cases that fall out of the queue are deleted, **except** those an
analyst has decided on or that carry a written investigation — work already
done is never discarded by a re-cut.

### Investigation (system, per case)

Triggered per case by an analyst. A fixed node sequence, not a free-running
tool loop:

```
load case → collect evidence → build query → retrieve typologies
          → generate narrative → validate citations ─┬─ retry once
                                                     ├─ fall back to template
                                                     └─ persist
```

The language model is called **once**, at the end, to write prose from evidence
already gathered. It cannot add a finding, and it cannot change a measurement:
the deterministic and retrieved state it reads are frozen structures.

### Analyst

1. Open the operations view: what has been processed and what is waiting.
2. Open the queue, sorted by any column the eye scans down.
3. Open a case: the three signals, the transaction's neighbourhood, the
   observed evidence, the cited typology passages, and the written assessment.
4. Run or re-run the investigation.
5. Record a decision. Decisions are appended, not replaced, so what was
   concluded and when survives.

## 6. Functional requirements

### 6.1 Data and graph

- The graph is built from the Elliptic dataset: 203,769 transactions, 234,355
  directed edges, 49 time steps. Each transaction has 166 engineered features,
  a time step, and a label of `illicit`, `licit` or `unknown`.
- Time steps are preserved and never shuffled. Steps 1–29 are the training
  range, 30–34 validation (thresholds and hyperparameters), 35–49 the held-out
  test range.
- The raw time step is never a model feature. It is a column so it can be
  filtered and indexed without becoming something a model can train on.
- Features are anonymised: no amounts, no wall-clock times, no addresses. Every
  behavioural signal the system derives is computable from the graph and those
  features alone.
- `unknown`-labelled transactions are the majority. They stay in the graph and
  contribute features as neighbours, but are never training targets and never
  enter a reported metric.
- A batch is one time step. Replay covers the test range only; earlier steps
  are loaded as historical graph context and are never scored or queued.

### 6.2 Risk model

- Two tabular baselines are trained: `local94` on each transaction's own
  features, and `all166` adding the dataset's neighbourhood aggregates. They
  are reported separately, because a model given the aggregates already holds
  one hop of graph information and is the harder bar to clear.
- A graph model (GraphSAGE, two `SAGEConv` layers) is trained on the same
  features plus the local neighbourhood.
- All are evaluated on the same held-out split using precision, recall and
  AUPRC on the illicit class.
- The alert budget is a rate, not a score cutoff: a fraction of each batch,
  applied by ranking and taking exactly the top *k*. Reviewer capacity is a
  count, and score distributions shift between time steps — a threshold frozen
  on validation collapses test recall from 0.374 to 0.063.
- **Promotion rule, fixed before the test split was read:** the graph model
  replaces the baseline as primary scorer only if its test recall at the 1%
  budget exceeds the best baseline's by at least 2 absolute points.
- **Result: it did not.** GraphSAGE came in 31.9 points below `xgb-all166`, so
  **XGBoost `xgb-all166` is the primary scorer**. The scoring pipeline loads
  whichever model the exported manifest names, so this required no change to
  the system around it. Full figures are in [the model report](Model%20Report.md).
- The graph model is retained regardless, for the investigation layer, where it
  produces two things a tabular model cannot: a 64-dimensional embedding per
  transaction, and a second opinion on risk. Neither gates the queue.
- Every training run is logged with its split, hyperparameters and metrics, and
  the chosen model is exported with a metadata file naming that run.

### 6.3 The three signals

Argus produces three numbers that mean genuinely different things, and keeping
them apart is a product requirement:

| Signal | Source | Decides |
|---|---|---|
| **Risk score** | XGBoost | Queue membership — ranks the batch, cut at the alert budget |
| **Second opinion** | GraphSAGE | Nothing |
| **Evidence confidence** | The gathered evidence | Nothing |

The case interface states this on the face of each tile, and only the risk
tile is visually accented.

### 6.4 Deterministic evidence

Five tools run against every queued case during replay, before any language
model is involved:

- **Neighbourhood profile** — degrees, distinct counterparties, same-batch
  neighbours, chain length.
- **Structural heuristics** — fan-out, fan-in, layering chain, dense cluster.
  These are graph shapes because the features are anonymised: fan-out stands
  in for structuring, fan-in for funnelling, a low-branching run for layering.
  Thresholds come from the graph's measured degree distribution, not guesses.
- **Flagged and confirmed neighbours** — counterparties an analyst confirmed,
  counterparties the model already flagged, and counterparties labelled illicit
  in a *strictly earlier* time step. It never reads the label of an unreviewed
  transaction in the current batch.
- **Structural similarity** — nearest neighbours in GraphSAGE embedding space,
  restricted to a reference pool of training-range labelled transactions and
  analyst-confirmed cases. Because similarity in representation space is not
  adjacency, this is the only mechanism that reaches across time steps into
  labelled history.
- **Graph model corroboration** — the GraphSAGE probability, recorded as a
  signal rather than a finding.

Each evidence item records its provenance as a foreign key to the row it came
from, never as free text, so every claim can be followed back to what produced
it.

### 6.5 Evidence confidence

- Computed deterministically from the evidence on a case, never self-reported
  by a language model.
- Combined per kind by noisy-OR and capped at that kind's weight, so
  corroboration across *different* kinds beats repetition of one. Five near
  identical similarity matches are one observation held more firmly, not five
  findings.
- `graph_model_corroboration` weighs **zero**: a model's own score folded into
  "how much evidence is there" would let a case with no evidence score highly
  because a second model agreed. `typology_reference` weighs zero too — a
  retrieved passage explains a signal, it is not one.
- Computed during replay, so a case carries a confidence before any
  investigation runs. Re-investigating does not move it.
- The weighting is versioned with the reports it produced.

### 6.6 Typology retrieval

- A hand-curated corpus of public AML typology reference material, split at
  section boundaries, embedded and stored in pgvector alongside the transaction
  data.
- Retrieval filters candidates by the pattern tags a case actually raised
  before ranking by cosine similarity, so an unrelated typology cannot be
  returned for a matched pattern.
- Retrieval is keyed off structural similarity and graph corroboration as well
  as fired heuristics. Keying it off heuristics alone would leave the corpus
  unused, because the queue is overwhelmingly low-degree transactions on which
  those heuristics do not fire.
- Corpus and query embeddings must come from the same model; the store is keyed
  by embedding model and retrieval refuses a mismatch.

### 6.7 Narrative generation

- One language-model call per investigation, after evidence is assembled.
- Structured output: typology assessment, recommended action, and claims.
- **Every claim must cite an evidence item that exists.** Output is validated
  against the assembled evidence set and rejected if it cites anything else.
- On rejection: one retry, then a rule-built narrative assembled from the
  evidence. The same fallback covers "no API key configured", so the fallback
  path is exercised constantly rather than rotting.
- The interface labels which of the two produced the text on the case page.

### 6.8 Alert budget

- The alert budget is the fraction of each batch that becomes an alert.
  1% is the default and the value every reported metric is measured at.
- It is **application state, not a display preference**. Changing it and
  applying it re-runs the selection: one replay per stored batch at the new
  budget, producing cases with full evidence and confidence like any other.
- The interface distinguishes *previewing* a budget from *having applied* one,
  and derives that distinction from the budget the stored batches were actually
  replayed at — not from the numeric value.
- Rebuilding is a background job with a duration; the control reflects the real
  lifecycle and does not return until the queue has actually changed.

### 6.9 Queue

- Lists every case, sorted by any of: batch rank, batch, risk score, second
  opinion, confidence, or workflow status. Both directions.
- Status sorts by position in the workflow, matching what the column displays,
  rather than alphabetically on a value the analyst cannot see.
- Sort state lives in the URL, so a view can be shared and restored by the back
  button.

### 6.10 Operations view

- Counts every figure from the database. No number is estimated.
- Reports batches replayed, transactions scored, alerts raised, cases awaiting
  a decision, and investigation progress.
- The risk distribution plots the whole scored population with the alerted
  slice marked, because the queue is by construction the top slice and a
  distribution of queued cases alone would show nothing.
- The alert count is counted from the case table — the same rows the queue
  lists — so the two screens cannot disagree.

## 7. Non-functional requirements

**Evaluation integrity.** Precision, recall and AUPRC on the illicit class are
reported for every model. Raw accuracy is never used on its own.

**Temporal integrity.** Splits follow time-step ordering. Feature scaling is
fitted on the training range alone; the graph seen during training is
restricted to training-range transactions; the embedding reference pool holds
nothing from a later range; thresholds and hyperparameters are chosen against
validation. The held-out range is read once. Each boundary is covered by a test
that fails if it is crossed.

**Explainability.** Every surfaced report cites specific evidence rather than
presenting a score with no supporting detail or an explanation invented by the
model.

**Reproducibility.** Every scoring run and model version is logged, so a score
traces back to the model and split that produced it. Replay is deterministic:
the same batch and model produce the same rows.

**Scope of data.** All data is static and replayed in batches; there is no
connection to a live financial system.

## 8. System design

```
Elliptic dataset ──▶ ingest ──▶ Postgres + pgvector
                                      │
                     XGBoost ◀────────┤────────▶ GraphSAGE
                        │                            │
                        ▼                            ▼
              replay: rank batch,          embeddings + second
              take top-k by budget         opinion (investigation only)
                        │                            │
                        ▼                            │
                   risk queue ──▶ deterministic evidence ◀┘
                        │                    │
                        │           AML typology corpus
                        │                    │
                        ▼                    ▼
              investigation (LangGraph) ── retrieval ── one LLM call
                        │                                    │
                        ▼                            citation validation
                  case report ──▶ React dashboard ──▶ analyst decision
```

### Components

- **Model** — PyTorch and PyTorch Geometric for GraphSAGE, XGBoost for the
  tabular models. MLflow records training runs; it is an experiment record, not
  a runtime dependency, so the chosen model is exported to a file the pipeline
  loads directly.
- **Agent** — LangGraph orchestrates a fixed node sequence. The language model
  is called through its own SDK; a fixed sequence needs no chat-model
  abstraction over it.
- **Retrieval** — pgvector serves two distinct spaces, typology text (768-d)
  and transaction structure (64-d), which is why no second store is needed.
- **API** — FastAPI. It reads precomputed rows and carries neither the graph
  model nor the scorer, so the API container needs no ML dependencies.
- **Database** — PostgreSQL with pgvector: the transaction graph, scores,
  embeddings, case reports, evidence, decisions and typology references.
- **Frontend** — React, TypeScript, Vite, Tailwind, TanStack Query.
- **Background work** — Celery with Redis, one worker, one queue, two tasks:
  `replay_batch` and `investigate_case`. Progress is written to a batch-run
  table rather than held in the result backend, so the interface reads job
  state from the same database as everything else.

### Data model

| Table | Holds |
|---|---|
| `transactions` | id, features, time step, label, in/out degree |
| `edges` | source transaction, target transaction |
| `risk_scores` | transaction, model version, score |
| `transaction_embeddings` | transaction, model version, 64-d embedding, graph score |
| `case_reports` | transaction, rank, scores, confidence, status, narrative |
| `evidence_items` | case, kind, summary, strength, weight, provenance reference |
| `typology_references` | source text, embedding, pattern tags, citation details |
| `reviews` | case, analyst, decision, note, timestamp |
| `batch_runs` | time step, status, counts, alert budget |
| `users` | the analysts who record decisions |

## 9. Constraints and limitations

These are properties of the finished system, stated plainly.

- **Local execution.** Replay and investigation are background jobs and need
  Redis and a running Celery worker. The API and the read-only interface work
  without them; anything that starts a job does not, and the interface says so
  rather than offering a control that fails.
- **The queue is structurally flat.** XGBoost's top 1% of Elliptic is almost
  entirely degree-≤2 transactions, so fan-in, fan-out and dense-cluster
  heuristics essentially never fire on it. The heuristics are correct and fire
  on high-degree nodes; the queue population simply does not have those shapes.
  Thresholds are not lowered to force hits.
- **Elliptic has zero cross-time-step edges.** Graph-based temporal leakage is
  structurally impossible, and the "illicit neighbour from an earlier time
  step" channel is always empty. Embedding similarity is the only mechanism
  that reaches into labelled history — which is what justifies retaining
  GraphSAGE.
- **Confidence is usually 0.250.** Only structural similarity can fire on this
  queue, and its noisy-OR saturates at that kind's weight. This is the honest
  output of the stated rule, not a defect.
- **The graph model lost the scoring comparison** and is used only in the
  investigation layer. A baseline win was a permitted, pre-registered outcome.
- **No authentication.** Decisions are attributed to a seeded analyst account.
- **Free-tier model limits.** The default provider allows 20 generation
  requests per day. Past that, investigations fall back to the rule-built
  narrative and record the error in the case's metadata — visible, not silent.
- **Recall is capped by the budget.** At 1%, the test range's 676 alert slots
  against 1,083 labelled illicit transactions cap achievable recall at 0.62.
  Reported recall should be read against that ceiling.

## 10. Success criteria

- The graph model's lift over the tabular baseline is measured and reported
  honestly, against a rule fixed before the held-out split was read.
- Every case report reaching the queue cites at least one specific piece of
  evidence, and any typology language is traceable to a retrieved passage.
- An analyst can move from the queue to a recorded decision without leaving the
  interface.
- Evidence confidence never changes which cases reach the analyst: the queue is
  the risk model's ranking cut at the alert budget, and nothing else.
- Precision and recall on the illicit class, not raw accuracy, are the reported
  measures of model quality throughout.

## 11. Future work

- Streaming ingestion in place of static batch replay.
- A feedback loop where analyst decisions inform recalibration or retraining.
- Temporal graph models if a static graph model's gains justify the complexity.
- Multi-hop entity resolution across accounts.
- Role-based case assignment across multiple analysts.
