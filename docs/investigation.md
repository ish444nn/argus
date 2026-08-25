# Replay, the risk queue and deterministic evidence

How a batch of transactions becomes a queue of cases, and how each case gets
evidence an analyst can check. Everything here runs without a language model;
the narrative and confidence layers come next.

## Replay

A batch is one Elliptic time step. `replay_batch(timestep)`:

1. loads the production scorer named by its manifest (`xgb-all166`);
2. scores every transaction in the batch, `unknown`-labelled ones included;
3. **ranks them and takes exactly the top 1%**;
4. upserts `risk_scores`, the `batch_runs` row and one `case_report` per alert;
5. runs the deterministic evidence tools over each case.

Scoring plus evidence for a ~5,500-transaction batch takes about 11 seconds.

### Why ranking, not a threshold

Phase 2 measured what a frozen probability threshold does on later time steps:
the validation threshold alerted **0.12%** of the test range instead of 1%, and
recall fell from 0.374 to 0.063. Elliptic's score distribution shifts sharply in
the later steps, so a stored cutoff stops meaning what it meant.

Reviewer capacity is a count, not a score. `select_alerts` therefore takes
`ceil(budget x n)` transactions by rank. That needs only the scores of the batch
in front of it — never labels — which is exactly what a live queue can do.

The threshold the model was calibrated at is still exported in its manifest, for
reference and for anything that needs a single number.

### Idempotency

Re-running a batch restates it. Three unique constraints do most of the work:

| Constraint | Effect on re-run |
|---|---|
| `batch_runs.timestep` | one run row per time step, updated in place |
| `risk_scores (tx_id, model_version)` | scores upsert |
| `case_reports.tx_id` | one case per transaction, upserted |

Evidence is derived, so it is deleted and rewritten per case rather than
accumulated.

The one judgement call: a case that was queued last time and is not this time —
possible if the model changes. Those are deleted, **unless an analyst has
already reviewed them**, in which case the case is kept and a warning is logged.
Discarding reviewed work silently would be worse than a slightly stale queue.

Verified: two consecutive replays of time step 35 produce identical row counts
(1 run, 56 cases, 313 evidence items, 5,507 scores).

### Failure is visible

The `batch_runs` row is created and committed **before** anything that can
fail, including loading the model. A failure then updates that row to `failed`
with its error. This was not academic: the first containerised run failed on a
model-path bug and reported "time step 41 has not been replayed", which is the
least useful answer available. The run row now exists from the start.

Job state lives in `batch_runs`, never in Celery's result backend, so the API
and the dashboard read it from the same database as everything else and it
survives a worker restart.

## The risk queue

`GET /api/queue` exposes transaction id, risk score, queue rank, time step,
GraphSAGE score, status, evidence count and the latest analyst decision, with
filtering by time step, status, tier and decision, and sorting by score, rank,
graph score or age.

`queue_rank` is stored rather than recomputed: it is fixed the moment the batch
is ranked, and a window function on every page load would be wasted work.

## Deterministic evidence

Five tools, all plain SQL over the graph. None of them asks a model for
anything.

### Structural heuristics

Elliptic's features are anonymised — no amounts, no wall-clock times, no
addresses — so the PRD's original heuristics (round-amount structuring, rapid
pass-through, burst frequency) are not computable. They were redefined as
network shapes that are:

| Heuristic | Fires when | Typology | Base rate |
|---|---|---|---|
| `fan_out` | out-degree ≥ 5, in-degree ≤ 2 | structuring | 1.06% |
| `fan_in` | in-degree ≥ 5, out-degree ≤ 2 | funnelling | 1.90% |
| `layering_chain` / `relay_chain` | run of ≥ 3 one-in/one-out hops | layering | ~15% of a batch |
| `dense_cluster` | ≥ 3 neighbours that also transact with each other | layering | — |

Thresholds come from the graph's own distribution, measured over all 203,769
transactions: in-degree median 1, p95 2, p99 9, max 284; out-degree median 1,
p95 2, p99 5, max 472.

That distribution is why bare pass-through is not evidence on its own.
**74,310 transactions — 36% of the network — are exactly one-in/one-out.** A
rule firing on that shape would flag a third of everything and mean nothing, so
`relay_chain` requires the pass-through to sit inside a chain.

### A finding worth stating plainly

Across four replayed batches (221 cases), **only one case had in- or out-degree
above 1**. XGBoost's top 1% is almost entirely degree-≤1 transactions, so the
degree-based heuristics essentially never fire on the queue: four `relay_chain`
items in total, and no `fan_in`, `fan_out` or `dense_cluster` at all.

The tools are not broken — they were verified firing correctly on transactions
that do have those shapes (a 14-out fan-out, an 86-in fan-in, a 6-hop chain).
The queue population simply does not have them. The honest response is to report
that, not to lower thresholds until something fires; tuning the evidence layer
until it agrees with the scorer would make the evidence worthless.

In practice a case is currently supported by roughly 4.6 similarity items plus
one graph-model corroboration. Similarity is doing the real work — which is what
the Phase 0.1 decision to keep GraphSAGE for the investigation layer was
betting on.

It also constrains Phase 4: typology retrieval keyed only off fired heuristics
would almost never run, so it must key off similarity and corroboration too.

### Neighbour evidence

The illicit-neighbour tool draws only on what the system legitimately holds at
the time of the batch:

1. neighbours an analyst has confirmed;
2. neighbours the model has already flagged into the queue;
3. neighbours labelled illicit in a **strictly earlier** time step.

It never reads the ground-truth label of an unreviewed transaction in the
current batch. That would be reading the answer.

Channel 3 always returns nothing on Elliptic, because the graph has zero
cross-time-step edges — every neighbour is in the same batch. The query is
written correctly anyway so the logic holds for any dataset, and its emptiness
is precisely why structural similarity exists.

## Structural similarity

The only tool that reaches across time. Adjacency cannot connect a transaction
to labelled history here; similarity in embedding space can.

Given a queued transaction, it finds the nearest GraphSAGE embeddings among the
**reference pool**: transactions from the training range (time steps 1–29) with
a known label, plus analyst-confirmed cases as they accumulate. Validation and
test labels are excluded — citing them would mean quoting an answer the system
is not supposed to have. The query transaction itself is excluded. Only illicit
matches become evidence; licit ones are counted in the details, because "three
of its five nearest historical neighbours were licit" is context worth seeing.

### The pgvector trap

pgvector's HNSW index **post-filters**: it walks the graph for `ef_search`
approximate neighbours and only then applies the `WHERE` clause. The reference
pool is ~13% of the table, so the default settings return **zero rows** for a
query with perfectly good answers. Confirmed by `EXPLAIN ANALYZE`:

```
Index Scan using ix_transaction_embeddings_embedding  (actual rows=40)
  -> after filtering: rows=0
```

The fix is `hnsw.iterative_scan = relaxed_order` with `ef_search = 200`,
applied **per connection** in `argus.db.session.PGVECTOR_OPTIONS` rather than
per query, so it cannot be forgotten and does not depend on transaction scope.

Verified against a brute-force scan over 12 sampled cases: **mean recall 1.0,
zero empty results**. A regression test asserts both that the configured query
fills its `k` and that turning `iterative_scan` off makes it under-return, so
the guard cannot quietly become a no-op.

One related trap: `SET LOCAL` is **not** undone by releasing a savepoint. The
exact-search path disables index scans and must `RESET` them in a `finally`;
without that, every later query on the session silently runs without indexes.
That bug made an earlier version of the regression test pass by accident.

## The evidence model

One table, `evidence_items`, used by everything. Phase 4 adds retrieval and a
narrative on top of the same drafts rather than a second pipeline.

Each item carries:

| Field | Meaning |
|---|---|
| `kind` | which tool produced it |
| `summary` | the claim, in plain language |
| `strength` | how strong *this* signal is, 0–1 |
| `weight` | how much this *kind* contributes to confidence — fixed, versioned |
| `neighbour_tx_id` / `typology_reference_id` | provenance, as a foreign key |
| `details` | the numbers behind the claim (degrees, distances, thresholds, base rates) |

Provenance is a real foreign key rather than free text: that is what makes a
report checkable instead of merely readable.

Weights (version `w1`):

```
confirmed_neighbour       0.40
structural_similarity     0.25
flagged_neighbour         0.20
heuristic                 0.15
graph_model_corroboration 0.15
typology_reference        0.00   explains a signal; is not one
```

Confidence is computed from `sum(strength x weight)` in Phase 4 — a pure
function of assembled evidence, never self-reported by a language model.

## Leakage controls

| Vector | Control |
|---|---|
| Similarity reference pool | Training-range labelled transactions only; asserted per match in tests |
| Neighbour evidence | Model flags and analyst reviews only; historical labels must predate the batch |
| Persisted evidence | A test scans every written row and fails if any cites a transaction outside the training range |
| Ground-truth labels | Surfaced in the case view for evaluation, never read by any tool and never the basis of an evidence item |

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/batches/{timestep}/replay` | Queue a replay. 202. Refuses time steps 1–34. |
| `GET /api/batches` / `GET /api/batches/{timestep}` | Progress, read from `batch_runs` |
| `GET /api/queue` | The queue, filtered and sorted |
| `GET /api/cases/{id}` | Case with neighbourhood profile and evidence |
| `GET /api/cases/{id}/evidence` | Evidence alone, ordered by contribution |
| `GET /api/transactions/{tx_id}/case` | Case lookup by transaction |

Handlers validate and delegate; the logic is in `argus.services`. Neither
service imports an ML library, so the deployed API image carries no torch,
xgboost, numpy, pandas or scikit-learn — verified in the running container.

The neighbourhood profile is computed on read rather than stored: it is the raw
material the heuristics come from, not a signal in its own right, and the
queries behind it are indexed and cheap.
