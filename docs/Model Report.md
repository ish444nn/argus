# Model report

How Argus turns the Elliptic dataset into a risk score and a set of transaction
embeddings, and what stops future information leaking into either.

## Dataset

The [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
is a Bitcoin transaction graph: 203,769 transactions, 234,355 directed
transaction-to-transaction edges, 49 time steps roughly two weeks apart.

Ingestion pulls the files from `https://data.pyg.org/datasets/elliptic`, the
copy maintained by the PyTorch Geometric project. The original Kaggle host
requires an account, which would make a clean clone non-reproducible. The raw
files are **not** committed — `python -m argus.ml.cli download` fetches them.

### File layout (verified, not assumed)

| File | Shape | Notes |
|---|---|---|
| `elliptic_txs_features.csv` | 203,769 x 167 | **No header row** |
| `elliptic_txs_classes.csv` | 203,769 + header | `txId,class` |
| `elliptic_txs_edgelist.csv` | 234,355 + header | `txId1,txId2` |

The feature file's columns are `txId`, `time step`, then 165 floats. The
literature's "166 features" counts the time step as the first of the 94 local
features, so the real split is:

```
166 features = 1 time step + 93 local + 72 aggregated
```

The 72 aggregated features are one-hop neighbour summaries computed by the
dataset authors — this matters for the model comparison below.

Class codes are `1` = illicit (4,545), `2` = licit (42,019), and the literal
string `unknown` (157,205). `unknown` means ground truth was unavailable, not
that the transaction is licit.

Verified properties: transaction ids are unique; the feature and class files
cover exactly the same transactions; the edge list has no duplicates, no
self-loops, and references no transaction absent from the feature file.

### The property everything else rests on

**Every edge connects two transactions in the same time step. There are zero
cross-time-step edges.** This was measured, not assumed, and
`test_elliptic_has_zero_cross_timestep_edges` fails if a future dataset
revision changes it.

Two consequences:

1. Temporal leakage through graph connectivity is *structurally impossible*.
   The subgraph induced on training time steps cannot reach a test node
   because no edge goes there.
2. A transaction's graph neighbours are always in its own batch. Nothing in
   the graph reaches back into labelled history — which is precisely the gap
   the GraphSAGE embeddings are there to fill (see *Structural similarity*).

## Temporal split

| Split | Time steps | Nodes | Labelled | Illicit | Used for |
|---|---|---|---|---|---|
| train | 1–29 | 120,804 | 26,381 | 2,871 | fitting, scaler, reference pool |
| validation | 30–34 | 15,461 | 3,513 | 591 | early stopping, threshold choice |
| test | 35–49 | 67,504 | 16,670 | 1,083 | reported once, at the end |

Nothing is ever shuffled. The published Elliptic benchmarks use a two-way
1–34 / 35–49 split, which leaves nowhere to pick a threshold except the test
set; carving validation out of the training range costs five time steps and
buys an honest threshold.

### What is available at each stage

**Fitting.** Features and labels of train-range transactions, plus the
*features only* of `unknown` transactions in that range, which act as
message-passing neighbours.

**Selection.** Validation scores and labels, used to early-stop and to choose
the operating threshold. No gradient ever flows from validation.

**Reporting.** Test scores and labels, used only for the final numbers.

## Leakage prevention

| Vector | Control |
|---|---|
| Labels | `unknown` nodes are never supervision targets and never enter a metric. Test labels touch nothing but the final report. |
| Graph neighbours | Each split is an induced subgraph. Elliptic's zero cross-step edges make this airtight. |
| Aggregated features | The dataset's 72 aggregates are one-hop *within* a time step, so they cannot summarise a future node. |
| Embeddings | The similarity reference pool is train-range labelled nodes only. |
| Scaling | `fit_scaler` fits on train-range rows only; validation and test are transformed with those statistics. |
| Target-derived features | None exist. No feature is computed from a label. |
| Time step | Kept out of the feature matrix entirely and stored in its own column, so no model can key on it and fail to extrapolate. |

Four tests enforce this, and each was verified to fail against deliberately
mutated code:

- `test_scaler_uses_training_rows_only`
- `test_training_graph_contains_no_future_nodes`
- `test_unknown_nodes_are_neighbours_but_never_supervision_targets`
- `test_poisoning_future_labels_cannot_change_training_supervision` — flips
  every validation and test label to `illicit` and asserts the training
  targets, supervision mask and edges are byte-identical.

## Feature variants

| Variant | Columns | What it knows |
|---|---|---|
| `local94` | 93 | The transaction's own attributes. No network information at all. |
| `all166` | 165 | Those plus the 72 one-hop neighbour aggregates. |

Both names keep the dataset's terminology; both exclude the time step, hence
93 and 165 rather than 94 and 166.

`all166` is the bar GraphSAGE must clear, because a model trained on it
already has one hop of graph information. Reporting `local94` alongside shows
how much of the graph signal the hand-crafted aggregates already capture.

## Graph construction

- **Undirected** for message passing: each directed edge is added both ways.
  `SAGEConv` aggregates over incoming edges only, so on the directed graph a
  node would see its senders but never its recipients. The directed edges are
  still stored verbatim in Postgres, where in- and out-degree feed the
  structural heuristics.
- **No self-loops.** `SAGEConv` has a separate root weight for the node's own
  features; a self-loop would double-count it.
- **Per-split induced subgraphs**, as described above.

## Evaluation

Accuracy is never reported as evidence. Illicit transactions are ~10% of
labelled nodes and ~2% of all nodes, so predicting "licit" everywhere scores
98% accuracy while catching nothing.

### The alert budget

A team reviews a fixed number of cases per batch, so the operational question
is "of the transactions we can afford to look at, how many illicit ones did we
find?". Every model is reported two ways:

- **`at_budget`** — rank the split's own scores and alert exactly the top 1%.
  This is what "recall at a fixed 1% alert budget" means, and what a live
  queue does. It uses only scores, never labels, so it is not test-set
  fitting.
- **`at_fixed_threshold`** — the validation threshold applied verbatim, with
  no recalibration.

Two implementation details that materially changed the results:

1. **Select a fixed count, not a threshold.** Boosted probabilities saturate
   near 1.0, so `scores >= threshold` admits every row tied at the cut. That
   handed `local94` 1.53% of the batch while `all166` got exactly 1.00% —
   not a like-for-like comparison. `alerts_at_budget` ranks and takes exactly
   *k*.
2. **The budget caps recall.** Test has 67,504 transactions and 1,083 labelled
   illicit ones, so 1% buys 676 alerts and caps achievable recall at **0.62**.
   Read the recall column against that ceiling, not against 1.0.

## Results

Trained on train, selected on validation, test evaluated once.

| Model | Variant | Val recall | **Test recall** | Test precision | Test AUPRC | Frozen-threshold recall |
|---|---|---|---|---|---|---|
| `xgb-local94` | local94 | 0.178 | 0.245 | 1.000 | 0.775 | 0.030 |
| **`xgb-all166`** | all166 | 0.206 | **0.374** | 1.000 | 0.790 | 0.063 |
| `graphsage` | all166 | 0.142 | 0.054 | 0.634 | 0.413 | 0.029 |

Recall and precision are at the 1% alert budget, illicit class, ceiling 0.62.

### The promotion rule, pre-registered

> GraphSAGE replaces XGBoost as the primary scorer if and only if its test
> recall at the 1% alert budget exceeds the best XGBoost variant's by at least
> 2 absolute percentage points.

**GraphSAGE came in 31.9 points below `xgb-all166`. XGBoost stays the primary
scorer.** This is the expected outcome — the original Elliptic paper also
found tree ensembles beat GNNs on this dataset — and it is a result, not a
failure. The pipeline is model-agnostic: it loads whichever model the manifest
names.

Three things worth noting:

- The one-hop aggregates earn their place: `all166` beats `local94` by 12.9
  points of recall. Hand-crafted neighbourhood features do carry real signal.
- Both XGBoost variants reach **precision 1.000** at the budget — inside the
  top 1%, every labelled alert is genuinely illicit. The binding constraint is
  budget, not precision.
- The frozen-threshold column is the interesting failure. Recall collapses
  from 0.374 to 0.063 when the validation threshold is applied without
  recalibration, because score distributions shift sharply in the later time
  steps (Elliptic's "dark market shutdown" around step 43). **A scorer must
  recalibrate its threshold per batch**, which is what the replay job does: it
  ranks each batch and takes the top *k*, and treats the stored threshold as a
  reference value only.

## Model roles

Deciding the queue and investigating a case are different jobs.

**XGBoost (`xgb-all166`) is the primary scorer.** It ranks each batch; the top
1% become cases. Nothing else gates the queue.

**GraphSAGE serves the investigation layer** with two outputs:

**A. 64-dimensional embeddings** (`transaction_embeddings`, pgvector). Taken
from the second `SAGEConv` layer, so the embedding is exactly the
representation the classifier scores. These enable *structural similarity*:
given a queued transaction, find historically labelled transactions the model
sees the same way. Because embedding similarity is not graph adjacency, it
reaches **across time steps** into labelled history — something neither the
graph nor XGBoost can do.

**B. A second opinion** — the graph model's own probability for the
transaction, quoted in a case report beside the risk score. Its role is
deliberately bounded: it never gates the queue, it is not one of the five
observed evidence kinds, and its raw score contributes nothing to the
deterministic evidence confidence. Folding a model's opinion into a measure of
how much evidence exists would let a case with no evidence score highly because
a second model agreed. Structural similarity (A) is the opposite case and does
count, because it is a measurement made *using* the embeddings rather than the
model's verdict on the transaction.

### Architecture

```
x -> SAGEConv(165, 128) -> ReLU -> Dropout(0.3)
  -> SAGEConv(128, 64)  -> ReLU        <- the embedding
  -> Linear(64, 1)                     <- the logit
```

Two layers because the aggregated features already summarise one hop, so the
second hop is the first thing the model can add. Deeper stacks oversmooth on a
graph this sparse.

Class imbalance is handled with weights on both sides — `scale_pos_weight` for
XGBoost, `pos_weight` in the BCE loss for GraphSAGE — never resampling.
Duplicating or synthesising nodes would change the graph structure the model
is meant to learn from, and keeping both families on plain class weighting
keeps the comparison like-for-like.

### Structural similarity: reference pool and a pgvector trap

The reference pool is **train-range labelled transactions only** (time steps
1–29). A validation or test node in the pool would mean citing an answer the
system is not supposed to have.

> **A pgvector trap worth knowing.** The HNSW index post-filters: it finds
> approximate neighbours first, then applies the `WHERE` clause. The reference
> pool is ~13% of rows, so a naive filtered query returns **zero rows** rather
> than a slow answer. The fix, applied on every connection, is
> `hnsw.iterative_scan = relaxed_order` (pgvector 0.8+), which returns results
> identical to an exact scan. A regression test compares the two.

## Where things are stored

| Output | Location |
|---|---|
| Transactions, labels, time steps, 165 features, degrees | `transactions` |
| Directed edges | `edges` |
| 64-d embeddings | `transaction_embeddings` (pgvector, HNSW cosine) |
| Trained XGBoost models | `models/xgb-{variant}/model.ubj` |
| Trained GraphSAGE | `models/graphsage/model.pt` |
| Threshold, split, metrics, scaler, MLflow run id | `models/*/metadata.json` |
| Experiment history | `mlflow.db` (SQLite) |

Model binaries, the raw dataset and `mlflow.db` are git-ignored;
`metadata.json` is committed so a score is always traceable to the run that
produced it.

MLflow is an experiment record, not a runtime dependency — it lives in the
optional `train` extra, and the worker image does not carry it. MLflow's
plain-directory backend is in maintenance mode and refuses to open, so
tracking uses a local SQLite file instead; `mlflow ui --backend-store-uri
sqlite:///mlflow.db` reads it.

## Reproducing

```bash
cd backend
uv sync --extra gnn --extra train

python -m argus.ml.cli download   # ~150 MB, once
python -m argus.ml.cli inspect    # dataset shape and split sizes
python -m argus.ml.cli ingest     # -> Postgres  (~47 s)
python -m argus.ml.cli train      # all three models  (~3 min, CPU)
python -m argus.ml.cli embed      # -> pgvector  (~5 min)
```

Seeds are fixed (42) for both model families. Training is CPU-only; there is
no GPU in this project's environment and none is needed at this graph size.
