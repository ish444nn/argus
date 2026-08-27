-- Runs once, on first initialisation of the Postgres data volume.
-- The migration creates this extension too; doing it here as well means
-- /health is green immediately after `docker compose up`, before migrations.
CREATE EXTENSION IF NOT EXISTS vector;
