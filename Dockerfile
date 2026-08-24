# One Dockerfile, two targets.
#
#   api    - FastAPI only. No torch, no xgboost. This is what gets deployed to
#            the free-tier web service, where it just reads precomputed rows.
#   worker - adds the `gnn` extra (torch, PyTorch Geometric, xgboost) because
#            scoring, embedding and training all happen in the worker.
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


FROM base AS api-deps
RUN uv sync --locked --no-dev --no-install-project

FROM base AS worker-deps
RUN uv sync --locked --no-dev --no-install-project --extra gnn


FROM api-deps AS api
COPY backend/ ./
RUN uv sync --locked --no-dev
EXPOSE 8000
CMD ["uvicorn", "argus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM worker-deps AS worker
COPY backend/ ./
RUN uv sync --locked --no-dev --extra gnn
CMD ["celery", "-A", "argus.jobs.celery_app:celery_app", "worker", "--loglevel=info", "--concurrency=2"]
