#!/usr/bin/env sh
# Crystal Cache container entrypoint (WS E, E.1 / D5).
#
# Runs database migrations to head, then hands off (exec) to the container's
# command — the API by default, or `python -m crystal_cache.workers` in the
# compose worker container.
#
# Migrations are idempotent (`alembic upgrade head` is a no-op when already
# current). In single-container `docker run` this brings a fresh SQLite volume
# up to schema on first boot. In docker compose a dedicated one-shot `migrate`
# service runs them once and the api/worker services set CC_RUN_MIGRATIONS=false
# (and depend_on it) so they never race each other against Postgres.
#
# CC_RUN_MIGRATIONS is read HERE, by the entrypoint shell — it is not a
# pydantic Settings field (the app ignores unknown CC_* vars).
set -e

if [ "${CC_RUN_MIGRATIONS:-true}" = "true" ]; then
  # SECURITY (2026-07-08): redact the password before echoing — the raw
  # URL was previously printed into Cloud Logging on every migration
  # boot. Host/db stay visible for debugging; credentials never log.
  _redacted_url=$(printf '%s' "${CC_DATABASE_URL:-<unset>}" | sed -E 's#(://[^:/@]+):[^@]*@#\1:***@#')
  echo "[entrypoint] alembic upgrade head (CC_DATABASE_URL=${_redacted_url})"
  alembic upgrade head
else
  echo "[entrypoint] CC_RUN_MIGRATIONS=false — skipping migrations"
fi

echo "[entrypoint] exec: $*"
exec "$@"
