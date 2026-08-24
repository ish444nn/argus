# CLAUDE.md — Argus working notes

Source of truth for requirements: `docs/prd.md`. This file is *how* we build it.
Status: **Phase 1 complete** — foundation and infrastructure. Next: Phase 2 (data
ingest, temporal splits, model comparison).

## What this project is

A portfolio project by a 3rd-year B.Tech CSE student, for internship applications.
Target: excellent *student* project with strong engineering judgment.
NOT a production compliance platform.

**Hard rule: simple > clever.** Do not add Kafka, Kubernetes, microservices,
service meshes, multiple databases, or extra abstraction layers. Every dependency
must be justified by a concrete PRD requirement.

## Git rules (non-negotiable)

- NEVER `git commit`, `git push`, create/merge PRs, or rewrite history.
- At the end of each phase, print a `Suggested commit` line only. The student commits.

## Environment (verified 2026-08-25)

- Windows 11, Git Bash + PowerShell both available.
- Python 3.14.6 system, `uv` 0.11.28 → **pin the project to Python 3.12 via uv**
  (PyTorch / PyTorch Geometric wheel coverage on 3.14 is not dependable).
- Node 24.14.1, npm 11.6.0 (no pnpm).
- Docker 29.7.2 + Compose v5.4.0. Docker Desktop must be running before any
  compose work.
- No NVIDIA GPU → CPU-only training. Fine: Elliptic is ~204k nodes / ~234k edges.
- **A native PostgreSQL 17 Windows service owns host port 5432.** Do not stop it.
  Compose therefore publishes postgres on **5433** and redis on **6380**. Host-side
  defaults in `core/config.py` and `.env.example` match. Inside the compose network
  the standard ports are still used.
- Frontend scaffold gave Vite 8 / React 19 / TypeScript 6 / oxlint. Tailwind v4 uses
  the `@tailwindcss/vite` plugin and `@import "tailwindcss"` — there is no
  `tailwind.config.js` and none is needed.

## Tech stack (LOCKED — Phase 0.1)

| Layer | Choice |
|---|---|
| Monorepo | `backend/` (one Python package) + `frontend/` + `data/` + `models/` + `docs/` |
| Python | 3.12, `uv`, single `pyproject.toml`, ruff + pytest |
| API | FastAPI + Pydantic v2 + pydantic-settings |
| DB | PostgreSQL 16 + pgvector, SQLAlchemy 2.0 (sync), Alembic |
| Jobs | Celery + Redis, one worker, one queue, 2 tasks |
| Primary scorer | XGBoost (all 166 features) |
| Graph model | GraphSAGE (PyTorch + PyG) -> 64-d node embeddings + secondary graph score |
| Experiments | MLflow, local file store (`mlruns/`), not a runtime dependency |
| LLM | Gemini `gemini-2.5-flash` via `google-genai`, structured output |
| Embeddings | Gemini `gemini-embedding-001`, 768-d |
| Vector store | pgvector — TWO spaces: typology text (768) + tx structure (64) |
| Agent | LangGraph, deterministic nodes, ONE LLM call at the end |
| Frontend | Vite + React + TS + Tailwind + TanStack Query, types from OpenAPI |
| Deploy | Render (web + static) + Supabase Postgres; worker/Redis local only |

## ML/DL split of responsibility (LOCKED — option B)

- **XGBoost decides the queue.** It is the production risk scorer. Threshold from
  validation at a 1% alert budget.
- **GraphSAGE serves the investigation layer.** Trained supervised on train
  timesteps; frozen; produces per-transaction 64-d embeddings written to pgvector,
  plus a secondary graph risk score. Neither gates the queue.
- **Two evidence types only GraphSAGE can produce:**
  1. `structural_similarity` — kNN in embedding space against a REFERENCE POOL of
     transactions with known labels from train timesteps 1-29 or analyst-confirmed
     reviews. Never same-batch unreviewed ground truth.
  2. `graph_model_corroboration` — the GraphSAGE illicit probability quoted as a
     second opinion in the report.
- The PRD 6.2 comparison still runs and is reported honestly in
  `docs/model-report.md`, but it is ONE PAGE, not the project thesis. If GraphSAGE
  clears the pre-registered bar (+2pt recall@1% budget on test), it is promoted to
  primary scorer; otherwise XGBoost stays. Pipeline is model-agnostic either way.

## RAG pipeline (LOCKED)

`data/typologies/*.md` (YAML frontmatter, committed) -> section-level chunks
(100-250 words, no overlap) -> Gemini embeddings 768-d -> `typology_references`
in pgvector -> retrieval filtered by `patterns && ARRAY[tag]` then ranked by
cosine, top-2 per fired heuristic -> `EvidenceItem(kind="typology_reference")`
-> LangGraph narrative node quotes ONLY retrieved text and cites evidence ids.

Corpus embeddings AND the canonical per-heuristic query vectors are committed to
`data/typologies/embeddings.json`, so retrieval is real vector search even with no
API key. Live query embedding is used when a key is present.

## Celery tasks (minimum viable — exactly 2)

- `replay_batch(timestep)` — score with XGBoost, compute GraphSAGE embeddings,
  write `risk_scores` + queued `case_reports`, then `.delay()` one
  `investigate_case` per queued case. No chord, no chain, no callbacks.
- `investigate_case(case_id)` — run the LangGraph investigation, write evidence
  and deterministic confidence, set primary/secondary queue tier.

Progress lives in the `batch_runs` table, NOT in Celery's result backend.

## Hosting (LOCKED)

- Free: Render Static Site (frontend), Render Web Service free (API), Supabase
  free Postgres (500 MB). API image needs neither torch nor xgboost — it only
  reads precomputed rows.
- Not free, therefore LOCAL ONLY: Redis + Celery worker.
- Hosted demo = precomputed snapshot seeded into Supabase (test timesteps 35-49
  metadata + edges + scores + case reports + evidence + typology refs + embeddings
  for the reference pool and queued cases; NO 166-feature arrays). Analyst
  decisions still write live — that is a PRD success criterion.
- "Run batch" is disabled in the hosted UI with a visible explanation.
- Storage measured: full local DB ~180 MB; hosted subset ~40 MB.

## Docker (LOCKED)

One `Dockerfile`, two targets: `api` (light) and `worker` (adds the `gnn` extra:
torch + PyG from the CPU-only index). Compose runs postgres, redis, api, worker.

## Non-negotiable correctness rules

1. **Temporal integrity.** Splits are by Elliptic timestep, never shuffled.
   Train 1–29, Val 30–34, Test 35–49. Test is touched ONCE, at the end.
2. **Threshold/hyperparameter selection uses validation only.** Never test.
3. **Scalers fit on train timesteps only.**
4. **The raw timestep column is never a model feature.**
5. **`unknown`-labelled nodes**: used as message-passing neighbours (features only),
   never as training targets, never in metrics.
6. **Metrics on the illicit class**: precision, recall, AUPRC, recall@fixed alert
   budget. Raw accuracy is never reported as justification.
7. **Model choice is pre-registered** before looking at test results. A baseline
   win is a valid, publishable outcome — the pipeline must be model-agnostic.
8. **Agent confidence is a pure deterministic function of assembled evidence.**
   The LLM never sets confidence.
9. **Typology language is retrieved, never generated.** Every typology claim cites
   a `typology_references` row.
10. **Every LLM-authored sentence must reference an existing evidence item id**;
    output is validated and rejected if it cites anything not in the evidence set.
11. **Agent tools must not read future ground truth.** See open decision Q4.

## Open decisions

All Phase 0 questions Q1-Q7 are RESOLVED (see Phase 0.1). Remaining unknowns are
empirical, to be measured in Phase 2, not decided by discussion:

- cross-timestep edge count in Elliptic (affects the flagged-neighbour tool)
- whether GraphSAGE clears the pre-registered promotion bar
- actual Gemini free-tier rate limits under batch investigation load

## Decisions already made (no need to ask)

- Auth: one `users` table, argon2 password hash, JWT bearer, seeded demo analyst.
  No OAuth, no refresh-token rotation, no RBAC (single role).
- Raw 166 features live in Postgres as `REAL[]` — one source of truth, no parquet
  shipped in the image.
- A "batch" = one Elliptic timestep. Demo replay covers test timesteps 35–49 only;
  1–34 are loaded as historical graph context but never scored or queued.
- MLflow is an experiment record, not a runtime dependency. The chosen model is
  exported to `models/` with a metadata file naming its MLflow run id.
- Heuristics are GRAPH-STRUCTURAL (relay, fan-out, fan-in, layering chain, dense
  cluster). Elliptic has no amounts or timestamps, so the PRD's "round amounts"
  and "rapid pass-through" heuristics are not computable and were redefined.
- FP budget = 1% alert rate, threshold chosen on validation. Promotion margin for
  GraphSAGE = +2 absolute points of recall. Pre-registered before the test run.
- LangGraph calls Gemini directly via `google-genai`. No LangChain chat-model
  wrappers, no LangChain retrievers — LangGraph does not require them.

## Phase 1 conventions worth keeping

- Enum-ish columns are `VARCHAR` + `CHECK`, generated from `argus/db/enums.py` via
  the `_check()` helper. Adding a value = replace one constraint, not `ALTER TYPE`.
  Evidence `kind` will grow in Phase 4; this is why.
- `alembic/env.py` has a `render_item` hook that adds the pgvector import to
  generated migrations. Without it autogenerate emits code that cannot run.
- `uv run alembic check` proves models and migrations agree. Run it after every
  model change; it is the cheapest schema-drift guard we have.
- FastAPI dependencies use `Annotated[X, Depends(...)]` aliases, not defaults
  (avoids ruff B008 and is the current FastAPI idiom).
- Integration tests are marked `@pytest.mark.integration` and self-skip when the
  compose stack is down, so bare `pytest` works anywhere.
- Docker: `api` target has no torch/xgboost. Do not import ML libraries from
  `argus.api.*` or `argus.core.*` — that would break the API image.

## Phase end protocol

Report: What I changed / Why / Tests-checks / Important decisions / Suggested commit.
3–7 bullets each. Plain language — the student needs to explain this in interviews.
