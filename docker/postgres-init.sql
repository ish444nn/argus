-- Runs once, on first initialisation of the Postgres data volume.
-- The migration also creates this extension (for Supabase and other
-- environments we do not initialise ourselves); doing it here as well means
-- /health is green immediately after `docker compose up`, before migrations.
CREATE EXTENSION IF NOT EXISTS vector;
