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
4. For each queued transaction, the investigation agent retrieves its local neighbourhood, checks how many connected transactions are already confirmed illicit or previously flagged, evaluates a small set of structural heuristics (relay, fan-out, fan-in, layering chain, dense cluster), and looks for historically labelled transactions whose network behaviour resembles this one.
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
- The dataset's timesteps are preserved and never shuffled; later timesteps are only ever used for evaluation, not training, since the graph and the entities within it evolve over time. Timesteps 1–29 are the training range, 30–34 the validation range used to select thresholds and hyperparameters, and 35–49 the held-out test range.
- The dataset's features are anonymised and carry no monetary amounts, wall-clock timestamps or addresses, so any behavioural signal the system derives must be computable from the graph and those features alone. Requirements elsewhere in this document are written against that constraint.
- Transactions labelled unknown are the majority of the dataset. They are retained in the graph and contribute their features as neighbours, but they are never used as training targets and never enter a reported metric, since treating absent ground truth as "licit" would misstate both.
- A batch is one timestep. Replay covers the test range only; earlier timesteps are loaded as historical graph context but are never scored or queued.

### 6.2 Risk model

- A tabular baseline (gradient-boosted trees) is trained on each transaction's own features alone, and a second variant is trained on those features plus the dataset's neighbourhood-aggregate features. The two are reported separately, because a model given the aggregates already holds one hop of graph information and is therefore the harder bar for a graph model to clear.
- A graph model (GraphSAGE) is trained on the same features plus the transaction's local neighbourhood.
- All are evaluated on the same held-out, later-timestep split using precision, recall and AUPRC on the illicit class.
- The fixed false-positive budget is an alert rate: 1% of the transactions scored in a batch. Because reviewer capacity is a count rather than a score, the budget is applied by ranking a batch and taking its top 1%, which depends only on scores and never on labels. The score threshold that produces that rate is selected on the validation range and exported with the model.
- The graph model is used in the scoring pipeline only if it improves recall on the illicit class at that budget by at least two absolute percentage points over the best baseline. The rule and its margin are recorded before the held-out split is evaluated, so the comparison cannot be settled after the fact. If the graph model does not clear the bar, the baseline is used instead — the choice is decided by the evaluation, not assumed in advance.
- Whichever model wins, the graph model is retained for the investigation layer, where it produces two things a tabular model cannot: a fixed-length embedding per transaction, and a second opinion on risk from a neighbourhood-aware model. Neither gates the queue unless the promotion rule above is satisfied.
- The scoring pipeline reads whichever model the exported manifest names, so a baseline win requires no change to the system around it.
- Every training run is logged with its data split, hyperparameters and resulting metrics, so any score in production can be traced to a specific model version.

### 6.3 Investigation agent

- The agent has five tools: neighbourhood lookup (return a transaction's connected transactions to a fixed depth), illicit-neighbour count (how many connected transactions are confirmed illicit or previously flagged), structural heuristics (relay, fan-out, fan-in, layering chain, dense cluster), structural similarity (find transactions whose learned representation resembles this one, among historically labelled cases), and typology retrieval (semantic search over a corpus of public AML typology reference material for the pattern a heuristic matched).
- The heuristics are graph-structural because the dataset's features are anonymised: it carries no amounts and no wall-clock times, so amount- and timing-based patterns are not computable from it. Fan-out stands in for structuring, fan-in for funnelling, and a low-branching chain for layering — each is a network shape the data can actually evidence, rather than a claim the data cannot support.
- The illicit-neighbour count draws only on information the system legitimately holds at the time of the batch: transactions an analyst has confirmed, transactions the model has already flagged, and ground truth from timesteps earlier than the batch under review. It never reads the label of an unreviewed transaction in the current batch, which would be reading the answer.
- Structural similarity searches a reference pool restricted to transactions labelled in the training range or confirmed by an analyst. Because similarity in representation space is not the same as adjacency in the graph, this is the one tool that can reach across timesteps into labelled history, and it is the reason the graph model is retained regardless of the scoring comparison.
- When a heuristic fires, the agent retrieves the relevant typology description rather than generating its own explanation of why the pattern is suspicious, so the report's language is grounded in a citable source instead of the model's own claim.
- A case report is a structured object: the transaction, a list of evidence items (each naming the specific connected transaction, pattern, or retrieved reference it's based on), and a confidence score. Each evidence item records its provenance as a reference to the row it came from, not as free text, so every claim in a report can be followed back to the thing that produced it.
- Confidence is computed deterministically from the evidence assembled (count and strength of supporting signals), not self-reported by the language model — a model's own stated confidence is not treated as a reliable signal. The weighting is versioned alongside the reports it produced, so a confidence score means the same thing every time it is read.
- The agent's steps run as a fixed sequence rather than a free-running tool loop: evidence is gathered deterministically, and the language model is called once, at the end, to write the narrative from evidence already collected. This keeps the report reproducible and leaves the model no opportunity to introduce a fact the system did not find.
- Every sentence the language model writes must cite an evidence item that exists. Output is validated against the assembled evidence set and rejected if it references anything else, falling back to a templated narrative rather than publishing an unsupported claim.
- Reports whose confidence falls below the queue threshold are stored but not surfaced as primary findings.
- When a report names a typology pattern (structuring, layering, funnelling), the explanation of that pattern is retrieved from a small embedded corpus of public AML typology reference material rather than generated from the model's own unverified knowledge, and the report cites the retrieved passage alongside the transaction-level evidence.

### 6.4 Review dashboard

- A queue view lists flagged cases ordered by score, with basic filtering by status.
- A case detail view shows the transaction, its neighbourhood, the model's score, and the full evidence report.
- An analyst can record a decision without leaving the case view.

## 7. Non-functional requirements

**Evaluation integrity.** The illicit class is a small minority of labelled transactions. Precision, recall and AUPRC on that class are reported for every model; raw accuracy is never used on its own to justify a model, since a model that predicts every transaction as licit would still score highly on accuracy alone.

**Temporal integrity.** Training and evaluation splits follow the dataset's timestep ordering. A model is never evaluated on a timestep that overlaps or precedes any timestep it was trained on. This extends past the obvious case of labels: feature scaling is fitted on the training range alone, the graph a model sees during training is restricted to training-range transactions, the embedding reference pool holds no transaction from a later range, and thresholds and hyperparameters are chosen against the validation range. The held-out range is read once, to produce the reported figures. Each of these is covered by a test that fails if the boundary is crossed.

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

- **Model** — PyTorch and PyTorch Geometric for the graph model, XGBoost for the tabular baseline. MLflow tracks every training run, its data split, and its metrics; it is an experiment record rather than a runtime dependency, so the chosen model is exported to a file the pipeline loads directly and no tracking server is needed to serve a score.
- **Agent** — LangGraph orchestrates the investigation steps; the neighbourhood-lookup, illicit-neighbour-count and heuristic tools are backed by the same graph data in Postgres, the structural-similarity tool queries the transaction embeddings, and the typology-retrieval tool queries pgvector for the reference text matching a fired heuristic. The language model is called directly through its own SDK, since a fixed node sequence needs no chat-model abstraction over it.
- **Retrieval** — a small corpus of public AML typology reference material, curated by hand, split at section boundaries, embedded and stored in pgvector alongside the transaction data, so no separate database is needed for it. Retrieval filters candidates by the pattern tags a heuristic fired before ranking them by similarity, so an unrelated typology cannot be returned for a matched pattern.
- **API** — FastAPI serving the queue, case detail and review endpoints. It reads precomputed rows and carries neither the graph nor the tabular model, which keeps the deployed image small.
- **Database** — PostgreSQL (with the pgvector extension) storing the transaction graph, model scores, transaction embeddings, case reports, analyst decisions, and the typology reference embeddings. pgvector serves two distinct vector spaces — typology text and transaction structure — which is why no second store is required.
- **Frontend** — React, consuming the API through a single typed client generated from the API's own schema.
- **Background work** — Celery with Redis runs the long jobs: a batch job scores newly ingested transactions and enqueues an investigation for each one above the risk threshold. Scoring and investigation are the only two tasks. Progress is written to a batch-run table rather than held in the queue's result store, so the dashboard reads job state from the same database as everything else.

### Deployment

| Component | Hosting |
|---|---|
| Frontend | Render Static Site, serving the built Vite bundle |
| Backend | Render Web Service, running the project's Docker image |
| Database | Supabase managed PostgreSQL |
| Queue and worker | Local only |
| Model artifacts | Exported to the repository's model directory with a metadata file naming the training run that produced them |

One Docker image is built with two targets: a light one for the API, and one carrying the machine-learning dependencies for the worker. Only the worker needs them, so the deployed web service stays small.

A managed queue and a always-on worker are the two components with no free tier, so they run locally rather than being hosted. The public demo is therefore seeded with the results of a batch replay run locally — the transaction metadata, edges, scores, case reports, evidence and embeddings for the test range — while analyst decisions are still written live against the hosted database, since recording a decision is a requirement of the product rather than a demonstration of it. Batch replay is disabled in the hosted interface, with the reason stated in the interface rather than left for the reader to infer.

### Data model

- `transactions` — id, features, timestep, label (illicit / licit / unknown), in- and out-degree.
- `edges` — source transaction, target transaction.
- `risk_scores` — transaction, model version, score.
- `transaction_embeddings` — transaction, model version, the graph model's learned representation.
- `case_reports` — transaction, confidence, queue tier, status, narrative.
- `evidence_items` — case, kind, summary, strength, and a reference to the transaction or typology passage it rests on.
- `reviews` — case, analyst, decision, timestamp.
- `typology_references` — source text, embedding, pattern tags, and the citation details of the document it came from.
- `batch_runs` — timestep, status, counts, so replay progress is queryable.
- `users` — the analysts who record decisions.

The timestep is stored as its own column rather than left among a transaction's features, so that it can be filtered and indexed without ever becoming something a model can train on.

### The risk model

The graph is used because transaction risk is genuinely relational: a transaction's connections carry information a tabular row cannot express. That relational signal is not assumed to be worth the added complexity by default — a tabular baseline is trained and evaluated on identical splits, and the graph model earns its place in the *scoring* pipeline only if it measurably improves recall on the illicit class at a fixed false-positive budget. If it does not, the simpler model is used. The comparison, and its result, is part of the system's output, not a footnote.

Scoring is not the only job on offer, though, and the comparison is not the point of the project. Deciding which transactions an analyst should look at, and helping them understand one once it is in front of them, are different problems, and a model can be the wrong tool for the first while being the only tool for the second. The graph model produces a representation of each transaction that places structurally similar transactions near one another; that representation supports a kind of evidence the tabular scorer cannot generate at all — pointing at previously confirmed cases that look, in network terms, like this one. So the graph model is retained for the investigation layer whichever way the scoring comparison lands, and the comparison decides one question rather than the system's shape.

The dataset's own construction makes the temporal argument unusually clean: its edges never join two different timesteps, so a model trained on the earlier range cannot reach a later transaction through the graph at all. The same property is what makes the learned representation valuable, since it is the only mechanism in the system that can relate a transaction under review to labelled history.

### The investigation agent

The agent is not asked to rate its own confidence, since a language model's self-reported certainty is not a reliable signal of correctness. Confidence is instead computed from the evidence the agent actually assembles — how many corroborating signals it found and how strong each one is — and only reports crossing a fixed threshold reach the analyst's primary queue. Everything else is retained in a lower-priority queue rather than discarded, so no finding disappears silently.

When a heuristic fires, the agent does not generate its own explanation of why the pattern matters — it retrieves the matching description from the typology corpus and cites it. The distinction matters: an invented explanation is a claim from the model, while a retrieved one is a claim the analyst can trace back to a source and check for themselves.

## 9. Success criteria

- The graph model's lift over the tabular baseline is measured and reported honestly, whichever way the comparison lands, against a rule fixed before the held-out split was read.
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
