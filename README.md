# Argus

Graph-based financial transaction risk assessment and evidence-grounded investigation.

Argus scores transactions in a financial transaction network using both their own
features and the structure of the network around them, then hands high-risk
transactions to an agent that gathers cited evidence and produces a structured case
report for a human analyst.

Two ideas sit at the centre, and both are tested rather than assumed:

1. **Network structure carries signal a single row cannot.** A tabular baseline and a
   graph model are trained on identical temporal splits, and the graph model is used
   for scoring only if it measurably wins.
2. **A score is not actionable; evidence is.** Case confidence is computed
   deterministically from the evidence assembled — never self-reported by a language
   model — and any typology language is retrieved from a reference corpus and cited,
   not invented.

Full requirements: [docs/prd.md](docs/prd.md). Architecture decisions and
implementation rules: [CLAUDE.md](CLAUDE.md).

> **Status: Phase 2 — models.** Infrastructure, schema, the Elliptic ingest, both
> model families and the transaction embeddings exist. The retrieval corpus,
> investigation agent and analyst dashboard are not built yet.
>
> The model comparison landed on the baseline: `xgb-all166` reaches 0.374 recall
> at a 1% alert budget on the held-out range against GraphSAGE's 0.054, so
> XGBoost is the primary scorer and GraphSAGE serves the investigation layer.
> Details and method in [docs/modeling.md](docs/modeling.md).

## Repository structure

```
backend/            One Python package, one pyproject
  argus/
    api/            FastAPI app, routers, response schemas
    core/           Configuration and logging
    db/             SQLAlchemy models, enums, session
    jobs/           Celery app and tasks
    ml/             Dataset, splits, features, graph, models, evaluation
  alembic/          Migrations
  tests/
frontend/           Vite + React + TypeScript + Tailwind + TanStack Query
data/               Datasets (git-ignored) and the curated typology corpus
models/             Exported model artifacts (git-ignored)
docs/               PRD and design notes
docker/             Container support files
```

## Local architecture

```
   frontend (Vite :5173)
         │  HTTP
         ▼
   API (FastAPI :8000) ──dispatch──▶ Redis ──▶ Celery worker
         │                                          │
         └──────────────▶ PostgreSQL + pgvector ◀───┘
```

The API and the worker run the same code from two Docker targets. Only the worker
carries PyTorch, PyTorch Geometric and XGBoost; the API just reads rows, which keeps
the deployed image small.

## Prerequisites

- **Docker Desktop**, running. Everything else runs inside it.
- **[uv](https://docs.astral.sh/uv/)** — for backend commands run on the host.
- **Node 20+** — for the frontend.

Python 3.12 is pinned in `backend/pyproject.toml` and installed by uv automatically;
you do not need it on your system.

## Configuration

```bash
cp .env.example .env
```

Defaults work as-is. `LLM_PROVIDER=stub` means **no API key is required** — the
system runs, and tests pass, entirely offline. Set `LLM_PROVIDER=gemini` and
`GEMINI_API_KEY` when you want real narratives.

Never commit `.env`; it is git-ignored.

> **Ports.** Compose publishes PostgreSQL on **5433** and Redis on **6380**, not the
> standard 5432/6379, so they cannot collide with a natively installed PostgreSQL or
> Redis. Inside the Compose network the services still use standard ports.

## Running

```bash
docker compose up            # postgres, redis, api, worker
```

The first build downloads PyTorch for the worker image and takes several minutes.
Afterwards it is cached.

Then, in another terminal, create the schema:

```bash
cd backend
uv run alembic upgrade head
```

Check it worked:

```bash
curl http://localhost:8000/health
```

All four dependencies (`api`, `postgres`, `pgvector`, `redis`) should report `ok`.
The endpoint always returns HTTP 200 and puts the per-dependency breakdown in the
body, so the dashboard can show you *which* dependency is down.

### Verifying the task queue

```bash
curl -X POST "http://localhost:8000/api/tasks/ping?message=hello"
curl http://localhost:8000/api/tasks/<task_id>
```

A `SUCCESS` state with `{"pong": "hello", ...}` proves FastAPI → Redis → worker.

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/health` and `/api` to the API in development, so there is nothing to
configure.

## Development

Backend, from `backend/`:

```bash
uv sync                              # core + dev dependencies
uv sync --extra gnn                  # add torch / PyG / xgboost
uv sync --extra gnn --extra train    # ...plus MLflow, for training
uv run ruff check . && uv run ruff format --check .
uv run pytest                        # unit tests
uv run pytest -m integration         # needs the compose stack up
uv run alembic upgrade head          # apply migrations
uv run alembic check                 # models and migrations agree
uv run alembic revision --autogenerate -m "..."
```

Frontend, from `frontend/`:

```bash
npm run check        # lint + typecheck + build
npm run dev
```

Tests marked `integration` need a live Postgres and Redis; they skip themselves when
the stack is not running, so a plain `uv run pytest` works anywhere.

## Machine learning

```bash
cd backend
uv sync --extra gnn --extra train

python -m argus.ml.cli download   # Elliptic from the PyG mirror, ~150 MB, once
python -m argus.ml.cli inspect    # dataset shape and split sizes
python -m argus.ml.cli ingest     # -> Postgres   (~47 s)
python -m argus.ml.cli train      # all three models, CPU  (~3 min)
python -m argus.ml.cli embed      # -> pgvector   (~5 min)

mlflow ui --backend-store-uri sqlite:///../mlflow.db   # experiment history
```

The raw dataset (~950 MB once extracted) and the trained model binaries are
git-ignored; each model's `metadata.json` is committed so any score stays
traceable to the run that produced it.

See [docs/modeling.md](docs/modeling.md) for the split, the leakage controls, the
alert-budget metric and the results.

## Database schema

Ten tables, created in one migration:

| Table | Holds |
|---|---|
| `users` | Analyst accounts |
| `transactions` | One row per transaction: timestep, label, raw features |
| `edges` | Directed transaction → transaction flows |
| `batch_runs` | One replayed batch (a batch is one timestep) |
| `risk_scores` | Model score per transaction, tagged with the model version |
| `transaction_embeddings` | GraphSAGE node embeddings, `vector(64)` |
| `case_reports` | One investigation, with its deterministic confidence |
| `evidence_items` | Cited facts; provenance is a real foreign key |
| `reviews` | Analyst decisions |
| `typology_references` | AML corpus chunks, `vector(768)` |

Both vector columns have HNSW cosine indexes. Enumerated values are stored as
`VARCHAR` with `CHECK` constraints, defined once in `argus/db/enums.py`, so adding a
value later is a constraint change rather than a PostgreSQL `ALTER TYPE`.

## Licence

MIT — see [LICENSE](LICENSE).
