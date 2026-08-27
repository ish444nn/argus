# CLAUDE.md — Argus working notes

Source of truth for requirements: `docs/prd.md`. This file is *how* we build it.
Status: **Phase 6 complete — project finished.** Hardening, CI, deployment
config, docs. All six phases done. Outstanding vs the PRD by decision: analyst
sign-in (documented as a known limitation, not an oversight).

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
| LLM | Gemini `gemini-3.6-flash` via `google-genai`, structured output |
| Embeddings | Gemini `gemini-embedding-001`, 768-d |
| Vector store | pgvector — TWO spaces: typology text (768) + tx structure (64) |
| Agent | LangGraph, deterministic nodes, ONE LLM call at the end |
| Frontend | Vite + React + TS + Tailwind + TanStack Query + react-router |
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

- `replay_batch(timestep)` — DONE (Phase 3). Score with XGBoost, rank, take the
  top 1%, upsert `risk_scores` + `case_reports`, run the deterministic evidence
  tools. Idempotent. GraphSAGE scores are read from `transaction_embeddings`,
  written by `argus.ml.cli embed`.
- `investigate_case(case_id)` — Phase 4. LangGraph investigation, typology
  retrieval, narrative. Confidence is computed earlier, during replay.

Progress lives in the `batch_runs` table, NOT in Celery's result backend.
**Heavy imports go INSIDE task bodies** — the API imports `argus.jobs.tasks` to
dispatch, and the API image has no torch/xgboost/numpy.
**Restart the worker after adding a task**; Celery registers at boot.

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

All Phase 0 questions Q1-Q7 are RESOLVED. The Phase 2 empirical unknowns are now
MEASURED:

- **Cross-timestep edges in Elliptic: ZERO** (all 234,355 edges are intra-timestep).
  Consequences: (a) graph-based temporal leakage is structurally impossible;
  (b) the "illicit neighbour from an earlier timestep" channel is ALWAYS EMPTY, so
  the Phase 4 flagged-neighbour tool must rest on same-batch model-flagged
  neighbours + analyst-confirmed reviews; (c) GraphSAGE embedding similarity is the
  ONLY mechanism that reaches across timesteps into labelled history. This is what
  justifies keeping the GNN.
- **GraphSAGE did NOT clear the promotion bar** (-31.9 recall points). XGBoost
  `xgb-all166` is the primary scorer. GraphSAGE stays in the investigation layer.
- Gemini free-tier limits under batch load: still unmeasured, deferred to Phase 4.

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

## Phase 2 findings that constrain later phases

- **Threshold recalibration is mandatory.** A threshold frozen on validation
  collapses test recall from 0.374 to 0.063, because Elliptic's score distribution
  shifts hard in the later timesteps (dark-market shutdown ~ts43). Phase 3's
  `replay_batch` MUST rank the batch and take the top 1%, not apply a stored
  score cutoff. The stored threshold is a fallback/reference only.
- **pgvector HNSW post-filters.** A kNN query filtered to the reference pool
  (~13% of rows) returns ZERO rows with default settings. Phase 4 must
  `SET hnsw.iterative_scan = relaxed_order` (verified identical to exact scan) or
  build a partial index. This bit once already; do not rediscover it.
- **Alert budget = exact top-k, never `score >= threshold`.** Boosted probabilities
  saturate; threshold comparison gave one variant 1.53% of the batch vs another's
  1.00% and made the comparison meaningless. Use `evaluate.alerts_at_budget`.
- **Feature dims are 93 / 165, not 94 / 166.** The literature counts the timestep
  as the first local feature. We drop it, so `local94`->93 cols, `all166`->165.
  The names keep the dataset's terminology; the dims are what the code uses.
- **MLflow file store is dead.** `./mlruns` refuses to open (maintenance mode).
  Tracking uses `sqlite:///mlflow.db`. MLflow lives in the optional `train` extra
  and is NOT in the worker image.
- Dataset comes from `data.pyg.org/datasets/elliptic` (no Kaggle account needed).
  `python -m argus.ml.cli download|inspect|ingest|train|embed`.
- Local DB after ingest + embeddings: ~384 MB. The hosted Supabase subset must
  drop the `features` arrays (they are 168 MB of the total) -- API never reads them.

## Phase 3 findings that constrain Phase 4

- **The queue is structurally flat.** Across 4 replayed batches (221 cases), only
  ONE case had in- or out-degree > 1. XGBoost's top 1% is almost entirely
  degree-<=1 transactions, so fan_in / fan_out / dense_cluster essentially never
  fire on the queue; only 4 `relay_chain` items fired in total. The heuristics are
  correct (verified firing on high-degree nodes) — the queue population just does
  not have those shapes. DO NOT lower the thresholds to force hits; that would be
  fitting the evidence layer to the answer.
- **Consequence for Phase 4 RAG**: typology retrieval keyed only off fired
  heuristics would almost never run. Key it off `structural_similarity` and
  `graph_model_corroboration` too, or map the pass-through/low-degree shape to a
  typology. Otherwise the corpus is dead weight.
- **In practice evidence is: ~4.6 structural_similarity + 1 graph_model_corroboration
  per case.** Similarity is doing the real work, which vindicates the Phase 0.1
  decision to keep GraphSAGE for the investigation layer.
- **pgvector settings are engine-level** (`argus.db.session.PGVECTOR_OPTIONS`),
  not per-query. `hnsw.iterative_scan=relaxed_order` + `ef_search=200`. Without it
  the filtered kNN returns ZERO rows — confirmed again by EXPLAIN in Phase 3.
- **`SET LOCAL` leaks past a savepoint.** `find_similar(exact=True)` disables
  index scans and MUST `RESET` them in a `finally`; without that every later query
  on the session silently runs without indexes. This made an early version of the
  regression test pass by accident.
- **Alert budget = exact top-k by rank** (`ml.scoring.select_alerts`), never a
  stored threshold. Verified: 5507 -> 56, 6393 -> 64.
- Replay idempotency rests on three unique constraints: `batch_runs.timestep`,
  `risk_scores(tx_id, model_version)`, `case_reports.tx_id`. Cases that fall out
  of the queue are deleted UNLESS reviewed.

## Phase 4 findings

- **Confidence must not sum linearly across items of one kind.** The first real
  run saturated at 1.0 for every case (5 similarity items x 0.25). Now noisy-OR
  per kind, capped at the kind's weight, so diverse corroboration beats
  repetition. Version string is `w1-noisyor+t0.35`.
- **Retrieval relevance is the pattern filter's job, not the vector's.** A
  cosine cutoff silently returned ZERO chunks for `network_association` (the tag
  never appears in the prose). Cutoff now only drops anti-correlated chunks.
- **Retrieval MUST key off `structural_similarity` + `graph_model_corroboration`,
  not just heuristics** — otherwise the Phase 3 degree-1 queue cites nothing.
- **Corpus and query embeddings must come from the same model.** Enforced by
  `typology_references.embedding_model`; retrieval raises on mismatch. Re-ingest
  after switching LLM_PROVIDER.
- **The stub embedder is a hashing vectoriser, not noise** — real lexical
  similarity, deterministic, so the keyless path is genuinely exercised.
- `StubProvider.narrate()` returns None -> caller uses the rule-built template.
  ONE fallback path covers both "no key" and "validation failed", so the
  fallback is exercised constantly instead of rotting.
- Deterministic + retrieved state are FROZEN dataclasses. The LLM physically
  cannot overwrite a measurement.
- `agent` extra (langgraph, google-genai) is worker-only, like `gnn`. The API
  still has no torch/xgboost/numpy/langgraph.
- Only `typology_reference` evidence is replaced on re-investigation; Phase 3
  evidence is never touched (test compares before/after).

## Phase 5 findings

- **`gemini-2.5-flash` is retired for new API keys.** The API returns 404 naming
  `gemini-3.6-flash`. Default updated in `core/config.py` and `.env.example`.
- **Gemini 3.x flash reasons before answering and the thinking tokens come out of
  `max_output_tokens`.** At 2048 the real investigation prompt returned truncated
  JSON; a trivial prompt already spent ~790 on reasoning. Now 8192.
  `thinking_budget=0` is REJECTED by 3.x flash -- you cannot switch it off.
- **The embedding-space guard earns its keep.** Running the test suite leaves the
  corpus embedded by the stub (conftest forces LLM_PROVIDER=stub); the next real
  run refuses rather than returning nonsense. Fix: re-ingest, then
  `argus.agent.cli investigate-top` to rebuild the demo state.
- **Running the test suite clears investigation citations** (agent fixtures
  re-ingest the corpus, which drops citations that no longer resolve). Rebuild
  with `ingest-corpus` + `investigate-top`.
- **Overview risk distribution reads `risk_scores`, not `case_reports`.** The
  queue is by construction the top 1%, so a distribution of queued cases puts
  every row in one band and shows nothing.
- Design identity is deliberately opposite to Last Call (warm paper / Fraunces
  serif / ember / rounded / mobile-first): Argus is cool dark slate, IBM Plex
  Sans + Mono, ~0 radius, no shadows, desktop-dense. Signature = the provenance
  rail (solid = measured, dashed = model, double = cited).

## Phase 6 findings

- **Test isolation is by embedding space, not a separate database.**
  `typology_references` is now keyed by (typology_id, chunk_index,
  embedding_model), ingest replaces only the active embedder's rows, and
  retrieval selects them. Tests (stub) and the demo (gemini) coexist. VERIFIED:
  a full pytest run leaves gemini citations/reports/ready-cases unchanged.
- **Two real bugs found while doing that.** (a) `evidence.persist` deleted ALL
  evidence, so replay wiped an investigation's typology citations -- now scoped
  via `kinds=DETERMINISTIC_KINDS`. (b) replay reset an investigated case to
  `queued`, discarding the report -- now only new cases start queued.
- **Agent tests build their own throwaway case** rather than investigating the
  top real case with the stub provider. Deleted on teardown.
- **`send_task` hangs for ~30s when Redis is down.** Celery's transport options
  do not bound the publish path. `api.deps.dispatch` calls
  `connection_for_write().ensure_connection(max_retries=0, timeout=3)` first and
  returns 503 in ~4s. Do not remove this.
- **`control.ping()` costs a flat second** unless given `limit=1`; /health went
  1015ms -> 10ms.
- **CORS is env-driven** (`CORS_ORIGINS`) and `APP_ENV=production` refuses to
  start on localhost DB/Redis/CORS values.
- **`docker compose up` now includes the frontend** (Vite dev server, bind
  mount, `VITE_DEV_API_TARGET=http://api:8000`). `npm run dev` on the host still
  works from `frontend/` -- it fails from the repo root, which is what the
  student hit.
- Deployment: Render (static + docker api) + Supabase. Worker/Redis stay local.
  `argus.ml.cli export-demo` writes 55MB of seed SQL (features NULL); verified
  loading into a clean migrated database -> 90MB, 0 dangling citations.

## Polish pass findings (post-Phase 6)

- **Three signals, one decider.** Risk score (XGBoost) decides the queue.
  Second opinion (GraphSAGE) and evidence confidence decide nothing. The case
  page states this on the face of each tile; only the risk tile is accented.
- **Primary/secondary queue tiers are GONE**, column and all (migration
  `0005_drop_tier`). Letting evidence confidence re-sort the queue conflated
  "how likely is this illicit" with "how much did we find out about it".
  `docs/prd.md` keeps the original requirement text with a recorded deviation.
- **`graph_model_corroboration` now weighs 0.0.** A model's own score is not
  evidence -- otherwise a case looks well-supported because a second model
  agreed. `structural_similarity` still weighs 0.25 even though it comes from
  the same embeddings, because it is a measurement against named labelled
  transactions. A test sweeps the raw graph score across its range and asserts
  confidence does not move. Version string is now `w1-noisyor` (no threshold).
- **Confidence is written during replay**, by `services.replay.score_confidence`,
  reusing `agent.confidence.compute` on the persisted rows. A case carries a
  confidence before any investigation. Re-investigating does not move it.
- **The "Run investigation" button looked broken** because the case query only
  polls while `status == "investigating"`, and status did not change until a
  manual refresh. Fixed with an optimistic `setQueryData` in `onSuccess` plus
  `run.isPending`, which both shows the running state and starts the poll.
- **Alert budget is explorable but not changed.** 1% remains canonical and is
  what every replay and every reported metric uses. `GET /api/overview?budget=`
  re-cuts the *stored* scores with a window function; it writes nothing. At 1%,
  670 transactions scoring >=0.99 go unselected; at 3%, 158. Showing that number
  is the answer to the obvious interview question.
- **CI installed only `--extra agent`**, so `tests/ml` and parts of
  `tests/services` failed at *collection* on numpy -- a skip marker never gets
  to run if the import fails first. Now `--extra agent --extra gnn`. That is
  test tooling only; the API Docker stage still installs core deps alone.
- **Render's Blueprint schema has no `dockerTarget`.** It builds the final
  stage, and the Dockerfile's last stage used to be `worker` -- so the fix also
  removed a live risk of shipping torch to the web service. `api` is now last.
  **Do not append a stage after it.** Verified: a bare `docker build .` has
  fastapi/sqlalchemy/celery and none of torch/xgboost/numpy/PyG/langgraph.
- **The typology chart was one purple bar.** Every typology drew from
  `--model`. Per-typology colours now live in `frontend/src/evidence.ts`
  alongside the labels, with `no_clear_typology` deliberately neutral.
- **Headless Edge is enough to actually look at the UI**:
  `msedge --headless=new --screenshot=out.png --window-size=W,H
  --virtual-time-budget=12000 <url>`. No Playwright needed.

## Phase 6.2 findings

- **CI needed a worker, not a weaker test.** `test_health_against_the_real_stack`
  failed with `degraded` because CI started Postgres and Redis but no Celery
  worker, and /health correctly calls that an error. CI now starts a native
  worker (`--extra gnn --extra agent` is already installed, and worker boot is
  light) and polls `celery inspect ping` until it answers. Background processes
  persist across steps in a job.
- **Render's `plan` is per *runtime*, not per type.** `services[1].plan no such
  plan free for service type web` pointed at the **static site**: static sites
  take no plan at all. The Docker web service keeps `plan: free`. No worker or
  Redis service was added; they stay local.
- **The API image caught a layering mistake at build time.** Importing
  `services.replay` from a router pulled `agent.tools.similarity` ->
  `argus.ml.splits` -> numpy, and the API container died on boot. Batch removal
  now lives in **`argus/services/batches.py`**, which imports only SQLAlchemy and
  models. Do not import `services.replay` from `argus.api.*`.
- **`OBSERVED_KINDS` is the one definition of "evidence".** The second opinion
  is a signal, not a finding, so it is excluded from the overview chart, the
  queue's evidence count, `/cases/{id}/evidence`, and the ids a narrative may
  cite. The row still exists and still drives typology retrieval. Before this,
  a report cited an evidence id the case page did not list.
- **`DELETE /api/batches/{ts}` undoes a replay**: run row, scores and unreviewed
  cases, scoped to that time step. Reviewed cases survive -- the same rule
  `_sync_queue` already applied. Synchronous, because needing a worker to tidy
  up after one would be perverse.
- **A test of mine was quietly degrading the demo.**
  `test_investigating_does_not_move_the_confidence` investigated the *top real
  case* with the stub provider, overwriting a Gemini report with a template one.
  It now picks an uninvestigated case and restores it in a `finally`. Verified:
  a full pytest run leaves every Gemini report intact.
- **Gemini free tier is 20 `generate_content` requests per day** for
  `gemini-3.6-flash` (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`).
  This is the number CLAUDE.md previously listed as unmeasured. Past it, every
  investigation falls back to the rule-built narrative and records the 429 in
  `investigation_meta.provider_errors` -- visible, not silent.
- **Categorical colours are `--cat-1..5`**, separate from the provenance trio.
  Reusing `--measured` for two evidence kinds made them one blue blob in every
  chart. Labels and colours now both come from `frontend/src/evidence.ts`.
- Range inputs are themed, not rebuilt: `.range` in `index.css` paints a 2px
  rail and a square thumb, keeping arrow keys and screen-reader support.

## Phase 6.3 findings

- **The CI failure was never the worker itself.** `Post Install uv` exited 2
  because `setup-uv@v5` prunes the cache in its post-job step, and the
  background worker had been started with `uv run` -- which keeps a uv process
  alive for the child's whole lifetime, holding the uv cache lock. `uv cache
  prune --ci` could not take the lock and failed a job whose tests had all
  passed. Two fixes: launch the worker from `.venv/bin/celery` (no uv process
  outlives the step) and move to `setup-uv@v10`, which runs on Node 24 and no
  longer prunes by default. Never start a long-lived process with `uv run` in
  CI.
- **An empty `REDIS_URL` now means "this deployment has no job broker".**
  `docs/deployment.md` used to tell you to set the placeholder
  `redis://localhost:6379/0`, which `APP_ENV=production` correctly refuses --
  the documented deployment could not boot. `Settings.has_broker` is the one
  test; `/health` reports redis and worker as **`disabled`** (not `error`, so
  production is not permanently "degraded"), and `deps.dispatch` returns 503
  instantly instead of trying to connect. A localhost value in production is
  still refused.
- **Render appends a suffix to a taken service name.** The blueprint's
  `CORS_ORIGINS` and `VITE_API_BASE_URL` must be the hosts the services
  actually got (`argus-api-oamt`, `argus-web-oamt`), not the names asked for.
- **The alert-budget arithmetic was right; the label was missing.** The budget
  is top-k **per batch**, so a batch with 262 transactions above 0.99 and 166
  slots leaves 96 unselected while another batch with spare slots reaches down
  into 0.95-0.99. Verified against the database: at 3%, 458 selected =
  166+192+100, of which 440 are >=0.99 and 18 are below. The chart now says
  "top 3% of each batch", which is what makes the composition add up.
- **Confidence sitting at 0.250 is the honest answer, not a bug.** Every
  deterministic tool runs on every case during replay, after the whole queue
  exists, and before confidence is computed. Measured on the current queue:
  153 of 154 cases have degree <= 2, no case has a chain longer than 2, and
  **zero** cases have a flagged or historically-illicit neighbour available --
  Elliptic's edges are all intra-time-step, so the historical channel is
  structurally empty. Only `structural_similarity` (weight 0.25) can fire, and
  its noisy-OR saturates. 21 cases score 0.000 because their five nearest
  reference-pool neighbours were all licit. Do not lower a threshold to move
  this number.
- **A 202 is not a running state.** `POST /investigate` returns as soon as the
  task is queued, before the worker sets `investigating`, so the case page's
  invalidate-and-refetch immediately overwrote the optimistic status and the
  button came back within a frame. The page now records `updated_at` at
  dispatch and treats the run as live until that value changes *and* the status
  is no longer `investigating`. Verified against a real Gemini call that took
  103 seconds.
- **`now()` is transaction start time, not commit time.** The investigation's
  final write stamps `updated_at` with when its transaction opened, ~100s
  before it commits. The watermark comparison still works because time moves
  forward, but do not use `updated_at` as a duration.
- **A discrete range input's ticks need the thumb's travel, not the width.**
  The thumb centre runs from half a thumb in to half a thumb from the end, so
  `justify-between` labels sit off their own values. `--range-thumb` is now a
  token shared by the thumb and the tick maths; measured drift is <0.01px.
- **The alert budget is session state.** It lives in a context above the
  router, persisted to `localStorage`, so it survives navigating to a case and
  back. It still writes nothing server-side -- it re-cuts stored scores with a
  window function, exactly as before.

## Phase end protocol

Report: What I changed / Why / Tests-checks / Important decisions / Suggested commit.
3–7 bullets each. Plain language — the student needs to explain this in interviews.
