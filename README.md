# Argus

**Risk scoring and evidence-grounded investigation for financial transaction
networks.**

Argus scores a graph of Bitcoin transactions for likely illicit activity, takes
the highest-risk slice into a review queue, and builds each alert into a case
report where every claim points at the specific transaction, measurement or
published passage it rests on.

The hard part is not the score. It is making the thing that explains the score
answerable to evidence — so a language model cannot introduce a finding the
system never made.

---

## What it does

```
Elliptic dataset  ─▶  score a batch  ─▶  rank, take the top k%  ─▶  risk queue
                                                                       │
                                                    deterministic evidence tools
                                                                       │
                                            typology retrieval ─▶ one LLM call
                                                                       │
                                                     citation validation ─▶ case
                                                                       │
                                                          analyst records a decision
```

1. **Replay** scores one time step of the graph, ranks it, and takes exactly the
   top *k* — where *k* is the alert budget, not a probability cutoff. Score
   distributions shift between time steps; a frozen threshold collapses recall
   from 0.374 to 0.063.
2. **Evidence** runs deterministic tools over every alert: the neighbourhood
   profile, structural heuristics, flagged and confirmed counterparties, and
   embedding-space similarity to historically labelled transactions. The graph
   model's second opinion is recorded beside them as a signal, not as an
   observed finding. No language model is involved.
3. **Confidence** is derived from the observed evidence by a fixed rule, before
   any investigation runs. It is not a model score, and the second opinion does
   not feed it.
4. **Investigation** retrieves matching AML typology passages, calls a language
   model **once** to write the assessment, and **rejects the output** if it
   cites anything not in the evidence set — falling back to a rule-built
   narrative rather than publishing an unsupported claim.
5. **Review**: an analyst reads the evidence, the cited sources and the
   assessment, and records a decision.

## Key capabilities

- **Three signals, one decider.** The risk score (XGBoost) decides queue
  membership. The graph model's second opinion and the evidence confidence
  decide nothing, and the interface says so on the face of each tile. The
  second opinion is not one of the five observed evidence kinds and carries no
  weight in the confidence.
- **Every claim is traceable.** Evidence items carry a foreign key to the row
  that produced them, never free text. A narrative citing an id that does not
  exist is rejected before it is stored.
- **Retrieved, not generated.** Typology language is quoted from a curated
  corpus of public AML reference material and cited; the model never explains a
  pattern in its own words.
- **A real alert budget.** Changing it re-runs the selection and rebuilds the
  queue: every stored batch is replayed at the new rate and the case count
  scales with it. It is application state, not a slider that moves a label.
- **Honest evaluation.** GraphSAGE was compared to a tuned tabular baseline
  against a promotion rule fixed before the test split was read. It lost by
  31.9 points of recall, and XGBoost is the primary scorer. See
  [the model report](docs/Model%20Report.md).
- **Provenance as a visual system.** Every block in the interface carries a rail
  showing whether it was measured, inferred by a model, or quoted from a
  source — distinguished by line style as well as colour.

## Architecture

| Layer | Choice |
|---|---|
| API | FastAPI, Pydantic v2 |
| Database | PostgreSQL 16 + pgvector, SQLAlchemy 2.0, Alembic |
| Jobs | Celery + Redis — one worker, one queue, two tasks |
| Risk score (primary scorer) | XGBoost `xgb-all166` (165 feature columns) |
| Graph model | GraphSAGE (PyTorch + PyTorch Geometric) → 64-d embeddings |
| Vector search | pgvector, two spaces: typology text (768-d), transaction structure (64-d) |
| Agent | LangGraph, deterministic nodes, one LLM call at the end |
| Language model | Gemini via `google-genai`, structured output |
| Frontend | React, TypeScript, Vite, Tailwind, TanStack Query |
| Experiments | MLflow (record only, not a runtime dependency) |

pgvector serves both vector spaces, so there is no second datastore. The API
container carries neither PyTorch nor XGBoost — it reads precomputed rows.

## Local setup

**Requires** Docker (with Compose) and, for the data pipeline,
[uv](https://docs.astral.sh/uv/) with Python 3.12.

```bash
git clone https://github.com/ish444nn/argus.git
cd argus
cp .env.example .env
docker compose up -d
```

That brings up PostgreSQL, Redis, the API, the Celery worker and the frontend
dev server. Compose publishes Postgres on **5433** and Redis on **6380** so
they never collide with a native install.

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

`/health` reports Postgres, pgvector, Redis and the worker individually, so a
missing dependency names itself.

### Environment

`.env.example` documents every variable; copy it and adjust. Nothing needs
changing to run locally. The two worth knowing:

- `LLM_PROVIDER` — `stub` (default) renders narratives from an evidence
  template, so the whole system works with no API key. Set to `gemini` and
  supply `GEMINI_API_KEY` for real narratives.
- `ALERT_BUDGET` — the fraction of each batch that becomes an alert. `0.01` is
  the value every reported metric is measured at.

Never commit `.env`; it is git-ignored.

### Loading the data

The transaction graph is not in the repository — it is fetched, ingested and
scored locally. From `backend/`:

```bash
uv sync --extra gnn --extra agent
uv run alembic upgrade head

uv run python -m argus.ml.cli download   # fetch the Elliptic dataset
uv run python -m argus.ml.cli ingest     # load the graph into Postgres
uv run python -m argus.ml.cli train      # train and compare the models
uv run python -m argus.ml.cli embed      # write GraphSAGE embeddings to pgvector

uv run python -m argus.agent.cli ingest-corpus   # embed the typology corpus
```

Then replay a batch from the interface (**Overview → Batches → Replay**), or
run an investigation from a case page.

## Usage

1. **Overview** — what has been processed, what is waiting, and how the queue
   is composed. Every figure is counted from the database.
2. **Alert budget** — move the slider to preview a different budget against the
   stored scores. The card distinguishes previewing from applied. Choose
   *Rebuild the queue* to actually re-run the selection; it takes a few seconds
   per batch and the control stays in its running state until the queue has
   genuinely changed.
3. **Queue** — every case, sortable by rank, batch, risk score, second opinion,
   confidence or status. Sort state lives in the URL.
4. **Case** — the three signals, the transaction's neighbourhood and ego graph,
   the observed evidence, the cited typology passages in full, and the written
   assessment.
5. **Run investigation** — retrieves typology sources and writes a cited
   assessment. This is a real background job; the button stays in its running
   state until the report lands.
6. **Record a decision** — confirmed, dismissed, or needs more evidence.
   Decisions are appended, never replaced.

## Project structure

```
backend/
  argus/
    api/          FastAPI routers, schemas, dependencies
    agent/        LangGraph investigation, evidence contract, tools, retrieval
    core/         settings and logging
    db/           SQLAlchemy models, enums, session
    jobs/         Celery app and the two tasks
    ml/           dataset, features, training, scoring, embeddings, CLI
    services/     replay, queue, overview, investigation, review
  alembic/        migrations
  tests/          unit and integration suites
frontend/
  src/
    routes/       Overview, Queue, Case
    components/   AppShell, Signals, Evidence, Investigation, Review, EgoGraph
    api/          typed client
data/typologies/  the AML typology corpus, with citation frontmatter
models/           exported model artifacts and their metadata
docker/           Postgres init
```

## Development

```bash
# Backend
cd backend
uv run pytest                 # test suite (integration tests self-skip)
uv run ruff check . && uv run ruff format --check .
uv run alembic check          # migrations still describe the models

# Frontend
cd frontend
npm run check                 # lint, typecheck, build
```

Integration tests skip themselves when the stack or the dataset is absent, so a
bare `pytest` works on a fresh clone. CI runs the same commands against a real
PostgreSQL, pgvector, Redis and Celery worker.

## Limitations

- **Replay and investigation need the worker.** They are background jobs. The
  API and the read side work without Redis and Celery; controls that start a
  job are disabled with the reason shown.
- **The queue is structurally flat.** The top 1% of Elliptic is almost entirely
  degree-≤2 transactions, so fan-in, fan-out and dense-cluster heuristics
  rarely fire on it. They are correct and do fire on high-degree nodes — the
  queue population simply does not have those shapes, and thresholds are not
  lowered to force hits.
- **Confidence is usually 0.250.** Only structural similarity can fire on this
  queue, and its noisy-OR saturates at that kind's weight. That is the honest
  output of the rule.
- **Recall is capped by the budget.** At 1%, 676 alert slots against 1,083
  labelled illicit transactions cap achievable recall at 0.62.
- **No authentication.** Decisions are attributed to a single seeded analyst.
- **Static data.** The graph is replayed from a fixed dataset; there is no live
  feed and no action is taken on any account.

## Documentation

- [PRD Argus.md](docs/PRD%20Argus.md) — what the product does and why, as a
  specification.
- [Model Report.md](docs/Model%20Report.md) — dataset, splits, leakage controls,
  evaluation methodology and results.

## License

MIT — see [LICENSE](LICENSE).
