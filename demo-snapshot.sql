-- Argus demo snapshot.
--
-- Apply to a database that has already been migrated:
--   psql "$DATABASE_URL" -f demo-snapshot.sql
--
-- `transactions.features` is NULL throughout: the 166-feature arrays
-- are ~168 MB and only the worker reads them, and the worker is not
-- hosted. Everything the API serves is present.

BEGIN;

-- Advance sequences past the seeded ids.
SELECT setval(pg_get_serial_sequence('users', 'id'), coalesce((SELECT max(id) FROM users), 1));
SELECT setval(pg_get_serial_sequence('batch_runs', 'id'), coalesce((SELECT max(id) FROM batch_runs), 1));
SELECT setval(pg_get_serial_sequence('risk_scores', 'id'), coalesce((SELECT max(id) FROM risk_scores), 1));
SELECT setval(pg_get_serial_sequence('typology_references', 'id'), coalesce((SELECT max(id) FROM typology_references), 1));
SELECT setval(pg_get_serial_sequence('case_reports', 'id'), coalesce((SELECT max(id) FROM case_reports), 1));
SELECT setval(pg_get_serial_sequence('evidence_items', 'id'), coalesce((SELECT max(id) FROM evidence_items), 1));
SELECT setval(pg_get_serial_sequence('reviews', 'id'), coalesce((SELECT max(id) FROM reviews), 1));

COMMIT;