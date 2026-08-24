"""Alembic environment.

The database URL comes from application settings, not alembic.ini, so there is
exactly one place that decides which database we are talking to.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from argus.core.config import get_settings

# Importing models registers every table on Base.metadata. Without this,
# autogenerate and `alembic check` would see an empty schema.
from argus.db import models  # noqa: F401
from argus.db.base import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def render_item(type_: str, obj: object, autogen_context) -> str | bool:  # noqa: ANN001
    """Emit the pgvector import that autogenerate would otherwise omit.

    Without this, a generated migration references pgvector.sqlalchemy.VECTOR
    but never imports it, and fails at runtime.
    """
    if type_ == "type" and obj.__class__.__module__.startswith("pgvector"):
        autogen_context.imports.add("import pgvector.sqlalchemy")
    return False


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        render_item=render_item,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
