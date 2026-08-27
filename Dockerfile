# One Dockerfile, two targets.
#
#   worker - `gnn` (torch, PyTorch Geometric, xgboost) because scoring and
#            embedding happen there, plus `agent` (LangGraph, the Gemini SDK)
#            because the investigation workflow does too.
#   api    - FastAPI only. No torch, no xgboost, no LangGraph. It reads
#            precomputed rows.
#
# **`api` is deliberately the final stage**, so a bare `docker build .` with no
# `--target` produces the light image rather than several gigabytes of PyTorch.
# Compose names `worker` explicitly when it wants the other one.
#
# PyTorch comes from the CPU-only index (configured in pyproject.toml); the
# default CUDA build is several GB and useless here.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# Dependency manifests only, so the (slow) dependency layer is cached
# independently of application source changes.
COPY backend/pyproject.toml backend/uv.lock ./


# --- Worker -----------------------------------------------------------------
# Built only when explicitly targeted; BuildKit skips these stages otherwise.

FROM base AS worker-deps
RUN uv sync --locked --no-dev --no-install-project --extra gnn --extra agent

FROM worker-deps AS worker
COPY backend/ ./
RUN uv sync --locked --no-dev --extra gnn --extra agent
CMD ["celery", "-A", "argus.jobs.celery_app:celery_app", "worker", "--loglevel=info", "--concurrency=2"]


# --- API --------------------------------------------------------------------
# Last on purpose: this is what a bare `docker build .` produces.

FROM base AS api-deps
RUN uv sync --locked --no-dev --no-install-project

FROM api-deps AS api
COPY backend/ ./
RUN uv sync --locked --no-dev
EXPOSE 8000
CMD ["sh", "-c", "uvicorn argus.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
