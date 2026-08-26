# Deployment

Argus deploys for **£0/month** on Render and Supabase free tiers. This page is
the exact sequence — every command has been run against a clean database.

## What is hosted, and what is not

| Component | Where | Why |
|---|---|---|
| Frontend | Render Static Site | Free |
| API | Render Web Service (Docker, `api` target) | Free |
| Database | Supabase Postgres + pgvector | Free, 500 MB |
| **Celery worker** | **Local only** | Render background workers are not free |
| **Redis** | **Local only** | Render Key Value is not free |

That split is deliberate. Batch replay and investigation are the expensive,
long-running parts and they run on your machine; their **results** are exported
and loaded into the hosted database once. The hosted app then serves the queue,
cases, evidence, investigations and citations — and **analyst decisions still
write live**, because "an analyst can move from the queue to a recorded
decision" is a PRD success criterion and a screenshot would not satisfy it.

To host the whole thing, uncomment the `worker` and `redis` blocks in
[`render.yaml`](../render.yaml). Expect roughly $7–14/month. Nothing else
changes — the worker runs the same image and reads the same variables.

---

## Before you start

You need, all free:

- a **GitHub** account with this repository pushed to it;
- a **Render** account ([render.com](https://render.com)) — sign in with GitHub;
- a **Supabase** account ([supabase.com](https://supabase.com));
- optionally a **Gemini API key** ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

You also need the local stack working, because the snapshot comes from it:

```bash
docker compose up -d
cd backend
uv run alembic upgrade head
uv run python -m argus.ml.cli download    # ~150 MB, once
uv run python -m argus.ml.cli ingest
uv run python -m argus.ml.cli train       # or use the committed models/
uv run python -m argus.ml.cli embed
uv run python -m argus.agent.cli ingest-corpus
```

---

## Step 1 — Create the Supabase database

1. [supabase.com/dashboard](https://supabase.com/dashboard) → **New project**.
2. Name it `argus`, pick a region near you, and **save the database password**
   — Supabase shows it once.
3. Wait for provisioning (~2 minutes).
4. **Database** → **Extensions** → search `vector` → enable it.
5. **Project Settings** → **Database** → **Connection string** → **URI**.
   Copy it and rewrite the scheme for SQLAlchemy:

   ```
   Supabase gives:  postgresql://postgres.abcd:PASSWORD@aws-0-eu-west-2.pooler.supabase.com:5432/postgres
   Argus needs:     postgresql+psycopg://postgres.abcd:PASSWORD@aws-0-eu-west-2.pooler.supabase.com:5432/postgres
                              ^^^^^^^^
   ```

   Keep this to hand as `SUPABASE_URL`. It is a credential — do not paste it
   into a file that gets committed.

## Step 2 — Create the schema

Migrations are the only thing that may change the production schema. Never
edit it by hand.

```bash
cd backend
DATABASE_URL="postgresql+psycopg://postgres.abcd:PASSWORD@...:5432/postgres" \
  uv run alembic upgrade head
```

Expect four migrations. This creates the ten tables, both HNSW vector indexes,
the GIN pattern index and every CHECK constraint.

## Step 3 — Export and load the demo snapshot

```bash
cd backend
uv run python -m argus.ml.cli export-demo --out ../demo-snapshot.sql
```

About 55 MB and 20 seconds. It excludes `transactions.features` — 168 MB of
model input the hosted API never reads, since the worker is not hosted.

Load it:

```bash
psql "postgresql://postgres.abcd:PASSWORD@...:5432/postgres" -v ON_ERROR_STOP=1 \
  -f demo-snapshot.sql
```

The file is one transaction, so a failure leaves nothing behind. Verified on a
clean database: 67,811 transactions, 275 cases, 1,120 evidence items, 54 corpus
chunks, **0 dangling citations**, total size **90 MB**.

> No `psql`? Supabase's **SQL Editor** has a file upload, or install the
> Postgres client (`winget install PostgreSQL.PostgreSQL` on Windows).

## Step 4 — Deploy to Render

1. [dashboard.render.com](https://dashboard.render.com) → **New** →
   **Blueprint**.
2. Connect the repository. Render reads [`render.yaml`](../render.yaml) and
   proposes **argus-api** and **argus-web**.
3. Before the first deploy, set the secret variables. `render.yaml` marks these
   `sync: false`, which means Render asks rather than reading them from the
   committed file:

   | Service | Variable | Value |
   |---|---|---|
   | `argus-api` | `DATABASE_URL` | your `postgresql+psycopg://…` URI |
   | `argus-api` | `REDIS_URL` | `redis://localhost:6379/0` (placeholder — no hosted worker) |

4. **Apply**. The first build takes 5–10 minutes.

## Step 5 — Point the two services at each other

Render assigns URLs like `argus-api-x7k2.onrender.com`. The names in
`render.yaml` are guesses, so fix both after the first deploy:

1. `argus-api` → **Environment** → set `CORS_ORIGINS` to the **frontend's**
   URL, e.g. `https://argus-web-p3n8.onrender.com`. No trailing slash.
2. `argus-web` → **Environment** → set `VITE_API_BASE_URL` to the **API's**
   URL, e.g. `https://argus-api-x7k2.onrender.com`.
3. Redeploy `argus-web` — Vite inlines this at **build** time, so an env change
   alone does nothing.

Getting these wrong is the most common failure, and it looks like an empty UI
with CORS errors in the browser console rather than an obvious misconfiguration.

## Step 6 — Verify

```bash
curl https://argus-api-x7k2.onrender.com/health
```

Expect `postgres` and `pgvector` `ok`. **`redis` and `worker` will report
`error`, and overall status will be `degraded` — that is correct**: there is no
hosted worker. The read-only product is fully functional.

Then open the frontend and walk: Overview → Queue → a case → evidence →
investigation → record a decision → reload and confirm it persisted.

> Render free web services **spin down after 15 minutes idle**; the next
> request takes ~50 seconds. Supabase free projects **pause after 7 days**
> without activity and need waking from the dashboard. Worth mentioning in your
> README so a reviewer is not confused.

---

## What triggers a deployment

`autoDeploy: true` on both services: **every push to `main` deploys**. Render
watches the branch directly; there is no deployment step in GitHub Actions.

```
push to main ──▶ GitHub Actions CI (lint, tests, build)
             └─▶ Render build ──▶ health check ──▶ live
```

CI and deployment run **in parallel** — Render does not wait for the tests. To
gate on CI, turn `autoDeploy` off and use Render's deploy hook from a workflow
step, or simply protect `main` so nothing merges without a green tick.

Render's own health check (`/health`) gates the cutover: a deploy that never
returns 200 is rolled back automatically.

## Refreshing the hosted data

The snapshot is a point-in-time export. After replaying more batches or running
more investigations locally, re-export and reload:

```bash
cd backend
uv run python -m argus.ml.cli export-demo --out ../demo-snapshot.sql
psql "$SUPABASE_URL" -c "TRUNCATE users, transactions, edges, batch_runs, risk_scores,
  transaction_embeddings, typology_references, case_reports, evidence_items, reviews CASCADE;"
psql "$SUPABASE_URL" -v ON_ERROR_STOP=1 -f demo-snapshot.sql
```

This discards decisions recorded on the hosted app. Export them first if they
matter.

## Running the worker against the hosted database

Useful for pushing fresh investigations without a full re-export. Point a local
worker at Supabase:

```bash
DATABASE_URL="$SUPABASE_URL" LLM_PROVIDER=gemini GEMINI_API_KEY=... \
  uv run python -m argus.agent.cli investigate-top --count 10
```

The worker is on your machine; only the results land in the hosted database.

## Troubleshooting

| Symptom | Cause |
|---|---|
| UI loads, no data, CORS errors in console | `CORS_ORIGINS` and `VITE_API_BASE_URL` disagree — Step 5 |
| UI calls `localhost:8000` | `VITE_API_BASE_URL` set but frontend not rebuilt |
| API exits at boot with "APP_ENV=production but…" | A development default reached production. The message names the variable. |
| `/health` shows `redis`/`worker` error | Expected — no hosted worker |
| First request takes ~50 s | Render free tier cold start |
| `relation "case_reports" does not exist` | Step 2 not run against this database |
| Snapshot load fails on a foreign key | Loading into a non-empty database — truncate first |
