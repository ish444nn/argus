# Argus — Product Requirements

*A risk-assessment and investigation system for financial transaction networks — graph-based risk scoring paired with an agent that gathers and cites evidence for every case it surfaces.*

**Status:** Planned
**Last updated:** August 2026

---

## 1. Overview

Argus is a risk-assessment and investigation system for financial transaction networks. Given a graph of transactions and the relationships between them, it scores how likely each transaction is to be illicit using both the transaction's own attributes and the structure of the network around it, then hands high-risk transactions to an agent that gathers supporting evidence and produces a structured case report for a human reviewer.

The product exists because transaction risk is relational, not just tabular. A transaction connected to known illicit activity is meaningfully riskier than an identical-looking transaction with no such connections, and that signal is invisible to a model that only sees a single row of features. A risk score alone is also not actionable — a reviewer still has to trace connections and assemble context for every flagged case by hand, which does not scale.

## 2. Problem

Transaction-risk systems commonly score each transaction independently of its neighbours, discarding network structure that carries real signal. Separately, even an accurate score gives a reviewer nothing to act on beyond a number — they still have to manually explore the surrounding graph, judge what's relevant, and decide whether a case is worth escalating.

- Tabular models cannot see a transaction's position in the network.
- A bare risk score provides no evidence and no explanation.
- Reviewing every flagged transaction by hand does not scale with volume.
- Class imbalance in illicit activity data makes naive accuracy a misleading measure of whether a model is actually useful.

## 3. Goals

| Goal | Measure |
|---|---|
| Use network structure that tabular features miss | Measurable lift over a tuned tabular baseline on the illicit class, evaluated on a held-out time window |
| Turn a score into something reviewable | Every surfaced case includes cited, specific evidence, not a bare number |
| Keep the review queue signal-heavy | Only findings above a confidence threshold reach the queue; the rest are held, not discarded |
| Evaluate honestly under imbalance | Precision, recall and AUPRC on the minority class are reported; accuracy alone is never used to justify the model |

### Non-goals

Argus does not connect to a live blockchain or banking feed, does not take autonomous action on an account (freezing, blocking, reporting), and does not attempt real-world identity resolution. It operates on a labelled transaction graph, replayed in batches to simulate an operational feed. It is not a compliance system and makes no legal or regulatory claim about the transactions it scores.

## 4. Users

**Analysts** are the system's only user role. An analyst reviews the queue of flagged transactions, opens a case to see the model's score alongside the agent's evidence, and records a decision: confirmed, dismissed, or needs more evidence. There is no separate "operator" role — ingestion and scoring run as background processes, not through a user-facing flow.

## 5. Core flows

### Scoring and investigation (system flow)

1. A batch of transactions is ingested into the graph, each linked to the transactions it sends to or receives from.
2. The risk model scores every transaction in the batch using its own features and the transactions connected to it.
3. Transactions above a fixed risk threshold are queued for investigation; the rest are stored but not surfaced.
4. For each queued transaction, the investigation agent retrieves its local neighbourhood, checks how many connected transactions are already confirmed illicit or previously flagged, and evaluates a small set of behavioural heuristics (rapid pass-through, structuring into round amounts, unusual burst frequency).
5. The agent assembles a case report citing the specific evidence it found, and computes a confidence score from that evidence.
6. Reports below the confidence threshold remain in a secondary queue; only high-confidence reports reach the analyst's primary queue.

### Analyst

1. Sign in.
2. View the queue of flagged transactions, ordered by risk score.
3. Open a case: see the transaction, its immediate neighbourhood, the model's score, and the agent's report with each claim linked to the specific connected transaction it's based on.
4. Record a decision — confirmed, dismissed, or needs more evidence.
5. Decisions are logged against the case for later review; they are not fed back into training automatically in this version.

## 6. Functional requirements

### 6.1 Data and graph

- The transaction graph is built from a labelled dataset (Elliptic) in which each transaction has engineered features, a timestep, and a label of illicit, licit, or unknown where ground truth is unavailable.
- Edges represent a transaction sending funds to another transaction.
- The dataset's timesteps are preserved and never shuffled; later timesteps are only ever used for evaluation, not training, since the graph and the entities within it evolve over time.

### 6.2 Risk model

- A tabular baseline (gradient-boosted trees) is trained on each transaction's own features alone.
- A graph model (GraphSAGE) is trained on the same features plus the transaction's local neighbourhood.
- Both are evaluated on the same held-out, later-timestep split using precision, recall and AUPRC on the illicit class.
- The graph model is used in the scoring pipeline only if it improves recall on the illicit class at a fixed false-positive budget over the baseline. If it does not, the baseline is used instead — the choice is decided by the evaluation, not assumed in advance.
- Every training run is logged with its data split, hyperparameters and resulting metrics, so any score in production can be traced to a specific model version.

### 6.3 Investigation agent

- The agent has four tools: neighbourhood lookup (return a transaction's connected transactions to a fixed depth), illicit-neighbour count (how many connected transactions are confirmed illicit or previously flagged), behavioural heuristics (burst frequency, round-amount structuring, rapid pass-through), and typology retrieval (semantic search over a corpus of public AML typology reference material for the pattern a heuristic matched).
- When a heuristic fires, the agent retrieves the relevant typology description rather than generating its own explanation of why the pattern is suspicious, so the report's language is grounded in a citable source instead of the model's own claim.
- A case report is a structured object: the transaction, a list of evidence items (each naming the specific connected transaction, pattern, or retrieved reference it's based on), and a confidence score.
- Confidence is computed deterministically from the evidence assembled (count and strength of supporting signals), not self-reported by the language model — a model's own stated confidence is not treated as a reliable signal.
- Reports whose confidence falls below the queue threshold are stored but not surfaced as primary findings.
- When a report names a typology pattern (structuring, layering, rapid pass-through), the explanation of that pattern is retrieved from a small embedded corpus of public AML typology reference material rather than generated from the model's own unverified knowledge, and the report cites the retrieved passage alongside the transaction-level evidence.

### 6.4 Review dashboard

- A queue view lists flagged cases ordered by score, with basic filtering by status.
- A case detail view shows the transaction, its neighbourhood, the model's score, and the full evidence report.
- An analyst can record a decision without leaving the case view.

## 7. Non-functional requirements

**Evaluation integrity.** The illicit class is a small minority of labelled transactions. Precision, recall and AUPRC on that class are reported for every model; raw accuracy is never used on its own to justify a model, since a model that predicts every transaction as licit would still score highly on accuracy alone.

**Temporal integrity.** Training and evaluation splits follow the dataset's timestep ordering. A model is never evaluated on a timestep that overlaps or precedes any timestep it was trained on.

**Explainability.** Every case report surfaced to an analyst cites specific evidence — a connected transaction, a heuristic that fired, a retrieved reference for any typology language used — rather than presenting a score with no supporting detail, or an explanation invented by the model itself. A reviewer must be able to verify the reasoning behind a finding, not take it on trust.

**Reproducibility.** Every scoring run and every model version is logged, so a given risk score can be traced back to the model and data split that produced it.

**Scope of data.** All transaction data is static and replayed in batches; there is no connection to a live financial system.

## 8. System design

```
Transaction graph (batched) ──▶ Feature + graph pipeline ──▶ Risk model (baseline + GraphSAGE) ──▶ Risk queue
                                                                                                        │
                                            AML typology corpus ──▶ pgvector ◀── retrieval ──▶ Investigation agent (LangGraph) ◀───┘
                                                                                                        │
                                                                                Case report ──▶ Review dashboard (React)
```

- **Model** — PyTorch and PyTorch Geometric for the graph model, scikit-learn/XGBoost for the tabular baseline. MLflow tracks every training run, its data split, and its metrics.
- **Agent** — LangGraph orchestrates the investigation steps; the neighbourhood-lookup, illicit-neighbour-count and heuristic tools are backed by the same graph data in Postgres, and the typology-retrieval tool queries pgvector for the reference text matching a fired heuristic.
- **Retrieval** — a small corpus of public AML typology reference material, embedded and stored in pgvector alongside the transaction data, so no separate database is needed for it.
- **API** — FastAPI serving the queue, case detail and review endpoints.
- **Database** — PostgreSQL (with the pgvector extension) storing the transaction graph, model scores, case reports, analyst decisions, and the typology reference embeddings.
- **Frontend** — React, consuming the API through a single typed client.
- **Background work** — a batch job scores newly ingested transactions and enqueues investigations above the risk threshold.

### Deployment

| Component | Hosting |
|---|---|
| Frontend | Render Static Site, serving the built Vite bundle |
| Backend | Render Web Service, running the project's Docker image |
| Database | Supabase managed PostgreSQL |
| Model artifacts | MLflow tracking server, or a local tracking store checked into the model registry |

### Data model

- `transactions` — id, features, timestep, label (illicit / licit / unknown).
- `edges` — source transaction, target transaction.
- `risk_scores` — transaction, model version, score.
- `case_reports` — transaction, evidence items, confidence, status.
- `reviews` — case, analyst, decision, timestamp.
- `typology_references` — source text, embedding, description of the pattern it documents.

### The risk model

The graph is used because transaction risk is genuinely relational: a transaction's connections carry information a tabular row cannot express. That relational signal is not assumed to be worth the added complexity by default — a tabular baseline is trained and evaluated on identical splits, and the graph model earns its place in the pipeline only if it measurably improves recall on the illicit class at a fixed false-positive budget. If it does not, the simpler model is used. The comparison, and its result, is part of the system's output, not a footnote.

### The investigation agent

The agent is not asked to rate its own confidence, since a language model's self-reported certainty is not a reliable signal of correctness. Confidence is instead computed from the evidence the agent actually assembles — how many corroborating signals it found and how strong each one is — and only reports crossing a fixed threshold reach the analyst's primary queue. Everything else is retained in a lower-priority queue rather than discarded, so no finding disappears silently.

When a heuristic fires, the agent does not generate its own explanation of why the pattern matters — it retrieves the matching description from the typology corpus and cites it. The distinction matters: an invented explanation is a claim from the model, while a retrieved one is a claim the analyst can trace back to a source and check for themselves.

## 9. Success criteria

- The graph model's lift over the tabular baseline is measured and reported honestly, whichever way the comparison lands.
- Every case report reaching the analyst queue cites at least one specific piece of evidence, and any typology language it uses is traceable to a retrieved reference rather than generated freely.
- An analyst can move from the queue to a recorded decision without leaving the dashboard.
- Below-threshold findings never appear in the primary queue.
- Precision and recall on the illicit class, not raw accuracy, are the reported measures of model quality throughout.

## 10. Future work

- Streaming ingestion in place of static batch replay.
- A feedback loop where analyst decisions inform model recalibration or retraining.
- Temporal graph models (e.g. EvolveGCN) if the static graph model's gains justify the added complexity.
- Multi-hop entity resolution across accounts.
- Role-based case assignment across multiple analysts.
