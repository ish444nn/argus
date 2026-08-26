"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from argus import __version__
from argus.api.routers import batches, health, overview, queue, tasks
from argus.core.config import get_settings
from argus.core.logging import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title="Argus API",
    version=__version__,
    description="Risk assessment and investigation for financial transaction networks.",
)

# The frontend is a separate origin in development (Vite on :5173) and a static
# site in production, so it always needs CORS. The list comes from
# CORS_ORIGINS -- never a wildcard, and never inferred from the request.
#
# `allow_credentials` is off: the API carries no cookies or session, and
# turning it on would forbid a wildcard origin later without buying anything
# now. When sign-in is added, this and the origin list change together.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(health.router)
app.include_router(tasks.router)
app.include_router(batches.router)
app.include_router(queue.router)
app.include_router(overview.router)
