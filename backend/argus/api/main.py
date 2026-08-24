"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from argus import __version__
from argus.api.routers import health, tasks
from argus.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="Argus API",
    version=__version__,
    description="Risk assessment and investigation for financial transaction networks.",
)

# The frontend is a separate origin in development (Vite on :5173) and a static
# site in production, so it always needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(tasks.router)
