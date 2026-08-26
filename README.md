# Argus

Graph-based transaction risk assessment and evidence-grounded investigation.

Argus scores transactions in a financial transaction network using both their
own features and the structure of the network around them, ranks each batch and
takes the top 1% for review, then runs an investigation that gathers cited
evidence and writes an assessment an analyst can check line by line.

Two ideas sit at the centre, and both are tested rather than assumed:

1. **Network structure carries signal a single row cannot.** A tabular baseline
   and a graph model are trained on identical temporal splits, and the graph
   model is used for scoring only if it measurably wins. It did not — that
   result is reported rather than buried.
2. **A score is not actionable; evidence is.** Case confidence is computed
   deterministically from assembled evidence — never self-reported by a
   language model — and every typology claim is retrieved from a reference
   corpus and cited, never invented.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
   Elliptic ──────▶ │  ingest ▸ splits ▸ XGBoost ▸ GraphSAGE    │  offline
   (203,769 tx)     └──────────────────┬───────────────────────┘
                                       │ model artifacts + embeddings
                                       ▼
   Browser ──▶ React ──▶ FastAPI ──▶ PostgreSQL + pgvector
                            │              ▲
                            │ dispatch     │ scores, cases, evidence, reports
                            ▼              │
                          Redis ──▶ Celery worker
                                       │
                        ┌──────────────┴───────────────┐
                        │ replay_batch                 │  XGBoost ▸ rank ▸ top 1%
                        │ investigate_case             │  LangGraph ▸ RAG ▸ Gemini
                        └──────────────────────────────┘
```

The API and the worker run the same code from two Docker targets. Only the
worker carries PyTorch, XGBoost, LangGraph and the Gemini SDK; the API reads
precomputed rows, which is what keeps the deployed image small.

| Layer | Choice |
|---|---|
| Backend | Python 3.12, FastAPI, SQLAlchemy 2.0 (sync), Alembic, `uv` |
| Database | PostgreSQL 16 + pgvector (two vector spaces: 768-d text, 64-d transaction) |
| Jobs | Celery + Redis, one worker, two tasks |
| ML | XGBoost (primary scorer), PyTorch + PyTorch Geometric (GraphSAGE) |
| AI | LangGraph, Gemini `gemini-3.6-flash`, `gemini-embedding-001` |
| Frontend | Vite, React 19, TypeScript, Tailwind v4, TanStack Query, React Router |
| Experiments | MLflow (local SQLite; not a runtime dependency) |

## Project structure

```
backend/
  argus/
    api/          FastAPI app, routers, schemas, dependencies
    agent/        Evidence, tools, RAG corpus, LangGraph workflow, LLM providers
    core/         Configuration and logging
    db/           Models, enums, session
    jobs/         Celery app and tasks
    ml/           Dataset, splits, features, graph, models, evaluation, scoring
    services/     Replay, investigation, queue, overview, review, snapshot
  alembic/        Migrations — the only thing that may change the schema
  tests/
frontend/
  src/
    api/          Typed fetch client
    components/   Shell, evidence, investigation, review, ego graph, primitives
    routes/       Overview, Queue, Case
data/
  typologies/     13 curated AML notes (committed)
  elliptic/       Dataset (git-ignored, downloaded by script)
models/           Trained artifacts (git-ignored; manifests committed)
docs/             Modeling, investigation, agent, design, deployment
```

## Local setup

**Prerequisites:** Docker Desktop, [uv](https://docs.astral.sh/uv/), Node 20+.
Python 3.12 is installed by uv; you do not need it on your system.

```bash
cp .env.example .env      # defaults work as-is
docker compose up -d      # postgres, redis, api, worker, frontend
```

Then create the schema:

```bash
cd backend
uv run alembic upgrade head
```

Open **http://localhost:5173**. The API is on **http://localhost:8000**
(`/docs` for the OpenAPI browser).

> **Ports.** Compose publishes PostgreSQL on **5433** and Redis on **6380**, not
> the standard ports, so they cannot collide with a natively installed
> PostgreSQL or Redis. Inside the Compose network the standard ports still
> apply.

### Running the frontend

Two options — both give hot reload.

**In Docker** (default). `docker compose up` starts a `frontend` service that
runs the Vite dev server with the source bind-mounted. Nothing else to do.

**On the host**, if you prefer. The `cd` matters — `npm run dev` fails from the
repository root, because `package.json` lives in `frontend/`:

```bash
cd frontend      # <- required
npm install      # first time only
npm run dev
```

If both are running they will fight over port 5173, so stop the container
first: `docker compose stop frontend`.

The dev server proxies `/api` and `/health` to the API, so the browser talks to
one origin and CORS never enters the picture during development.

## Environment variables

Copy `.env.example` to `.env`. It is git-ignored and must never be committed.

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `local` | `production` enables startup checks that reject development defaults |
| `LOG_LEVEL` | `INFO` | |
| `DATABASE_URL` | `…@localhost:5433/argus` | SQLAlchemy URL — needs the `postgresql+psycopg://` scheme |
| `REDIS_URL` | `redis://localhost:6380/0` | Broker; db 1 is the result backend |
| `CORS_ORIGINS` | `http://localhost:5173,…` | Comma-separated. Production must set its real frontend origin. |
| `ALERT_BUDGET` | `0.01` | Fraction of each batch that becomes an alert |
| `LLM_PROVIDER` | `stub` | `stub` or `gemini` |
| `GEMINI_API_KEY` | *(empty)* | Required only when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` | `gemini-3.6-flash` | |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | |
| `EMBEDDING_DIM` | `768` | Must match the `vector(768)` column |

**No API key is needed.** With `LLM_PROVIDER=stub` the corpus is embedded by a
deterministic hashing vectoriser and narratives are built from the evidence by
rule, so the whole pipeline runs and every test passes offline.

## Gemini configuration

Get a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
— the free tier covers `gemini-3.6-flash` and `gemini-embedding-001`. In `.env`:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
```

Then embed the corpus in the Gemini space:

```bash
cd backend
uv run python -m argus.agent.cli ingest-corpus
```

Corpus and query vectors must come from the same model. Both spaces are stored
side by side and retrieval selects the active one, so switching providers is
non-destructive — but each space needs ingesting once.

## Running the ML pipeline

```bash
cd backend
uv sync --extra gnn --extra train        # torch, PyG, xgboost, mlflow

uv run python -m argus.ml.cli download   # ~150 MB, once
uv run python -m argus.ml.cli inspect    # dataset shape and split sizes
uv run python -m argus.ml.cli ingest     # -> Postgres  (~47 s)
uv run python -m argus.ml.cli train      # all three models  (~3 min, CPU)
uv run python -m argus.ml.cli embed      # embeddings + graph scores -> pgvector
```

`train` evaluates the held-out split once and applies a promotion rule fixed
before that split was read. See [docs/modeling.md](docs/modeling.md).

## Running a batch replay

A batch is one Elliptic time step. Replay scores every transaction, takes the
top 1% by rank, and gathers deterministic evidence for each case.

```bash
curl -X POST http://localhost:8000/api/batches/35/replay   # 202, runs async
curl http://localhost:8000/api/batches/35                  # poll progress
```

Job state is read from the `batch_runs` table, not Celery's result backend, so
it survives a worker restart. Replay is idempotent — running it twice produces
the same rows and never disturbs a written investigation.

Only time steps 35–49 can be replayed; earlier ones are training data.

> After adding or renaming a Celery task, restart the worker
> (`docker compose restart worker`). Celery registers tasks at boot.

## Running an investigation

```bash
curl -X POST http://localhost:8000/api/cases/1/investigate   # 202, async
curl http://localhost:8000/api/cases/1                       # poll
curl http://localhost:8000/api/cases/1/sources               # what it cited
```

Or from the CLI:

```bash
cd backend
uv run python -m argus.agent.cli investigate 1
uv run python -m argus.agent.cli investigate-top --count 8   # rebuild a demo state
```

See [docs/agent.md](docs/agent.md) for the workflow, the grounding rules and
the citation validator.

## Running tests

```bash
cd backend
uv run pytest                    # everything available in this environment
uv run pytest -m "not integration and not dataset"   # offline, no stack needed
uv run ruff check . && uv run ruff format --check .
uv run alembic check             # models and migrations agree
```

Integration tests skip themselves when the stack or dataset is absent, so a
plain `uv run pytest` works on a fresh clone.

Tests run against the stub provider and use their own throwaway case and their
own embedding space, so **a test run never disturbs real investigation state**.

Frontend:

```bash
cd frontend
npm run check     # lint + typecheck + production build
```

## Deployment

Free, on Render and Supabase. Exact steps, including what to click and what to
paste where: **[docs/deployment.md](docs/deployment.md)**.

| Component | Where |
|---|---|
| Frontend | Render Static Site |
| API | Render Web Service (Docker `api` target) |
| Database | Supabase Postgres + pgvector |
| Worker + Redis | Local only — not free on Render |

The hosted app serves the queue, cases, evidence, investigations and citations
from a snapshot exported by `argus.ml.cli export-demo`, and **analyst decisions
write live**. Batch replay and investigation run locally.

## CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`main` and every pull request:

- **Backend** — ruff lint and format, `alembic upgrade head` from an empty
  database, `alembic check` for model/migration drift, and pytest against a
  real Postgres+pgvector service container.
- **Frontend** — `npm ci` then `npm run check` (lint, typecheck, build).

No credentials: the suite runs on the deterministic stub provider, and no ML
training or real model call happens in CI. A separate
[`gemini-smoke.yml`](.github/workflows/gemini-smoke.yml) checks the live Gemini
contract; it is manual-only and skips without a key.

**Deployment is triggered by a push to `main`** — Render watches the branch
directly (`autoDeploy: true`). CI and deploy run in parallel; protect `main` if
you want a green tick to gate it.

## Known limitations

Honest, and each is a deliberate choice rather than an oversight.

- **No sign-in.** The PRD describes analysts signing in; there is one seeded
  demo analyst and decisions are attributed to them, labelled as such in the
  UI. Adding real authentication would be a day of work that demonstrates
  nothing the rest of the project does not already show.
- **The worker and Redis are not hosted.** Neither has a free tier. The
  architecture supports it — uncomment two blocks in `render.yaml` — but the
  deployed demo serves precomputed results and says so.
- **GraphSAGE lost the scoring comparison** (0.054 vs 0.374 recall at a 1%
  budget). It is retained for the investigation layer, where it produces
  evidence a tabular model cannot. This is reported, not hidden.
- **The structural heuristics rarely fire.** XGBoost's top 1% is almost
  entirely degree-≤1 transactions, so fan-in/fan-out/dense-cluster almost never
  trigger. The thresholds were not lowered to force hits.
- **Elliptic is anonymised**, so amount- and timing-based typologies are not
  computable. The PRD's heuristics were redefined as network shapes the data
  can actually evidence.
- **Free-tier behaviour.** Render web services cold-start (~50 s after 15
  minutes idle); Supabase projects pause after 7 days idle.
- **The typology corpus is a curated paraphrase**, not verbatim source text, so
  it can ship with the repository. Every note cites the document to read for
  authoritative wording.

## Documentation

| Document | Covers |
|---|---|
| [docs/prd.md](docs/prd.md) | Product requirements |
| [docs/modeling.md](docs/modeling.md) | Dataset, splits, leakage controls, model comparison |
| [docs/investigation.md](docs/investigation.md) | Replay, alert budget, deterministic evidence, pgvector |
| [docs/agent.md](docs/agent.md) | RAG, LangGraph, Gemini, grounding, confidence |
| [docs/design.md](docs/design.md) | Visual system and interface decisions |
| [docs/deployment.md](docs/deployment.md) | Hosting, step by step |
| [CLAUDE.md](CLAUDE.md) | Working notes and locked decisions |

## Licence

MIT — see [LICENSE](LICENSE).
