#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Config
# -----------------------
PG_VER="${PG_VER:-14}"
CLUSTER_NAME="${CLUSTER_NAME:-main}"
PGDATA="${PGDATA:-/workspace/pgdata}"

OPTUNA_USER="${OPTUNA_USER:-optuna}"
OPTUNA_PASS="${OPTUNA_PASS:-optuna_pw}"
OPTUNA_DB="${OPTUNA_DB:-optuna_db}"

# -----------------------
# Helpers
# -----------------------
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

need_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: must run as root"
    exit 1
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# -----------------------
# Main
# -----------------------
need_root

log "Installing PostgreSQL ${PG_VER} (if needed)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  "postgresql-${PG_VER}" "postgresql-contrib-${PG_VER}" \
  postgresql-common rsync \
  && rm -rf /var/lib/apt/lists/*

log "Ensuring target data dir exists: ${PGDATA}"
mkdir -p "${PGDATA}"
chown -R postgres:postgres "${PGDATA}"
chmod 700 "${PGDATA}"

# If Postgres is running, stop it.
if pgrep -u postgres -f "postgres.*${PG_VER}.*${CLUSTER_NAME}" >/dev/null 2>&1 || pgrep -u postgres -x postgres >/dev/null 2>&1; then
  log "Stopping existing cluster (if running): ${PG_VER}/${CLUSTER_NAME}"
  pg_ctlcluster "${PG_VER}" "${CLUSTER_NAME}" stop || true
fi

# If default cluster exists, drop it.
if pg_lsclusters | awk '{print $1" "$2}' | grep -q "^${PG_VER} ${CLUSTER_NAME}$"; then
  log "Dropping existing cluster: ${PG_VER}/${CLUSTER_NAME}"
  pg_dropcluster --stop "${PG_VER}" "${CLUSTER_NAME}"
else
  log "No existing cluster ${PG_VER}/${CLUSTER_NAME} found (ok)."
fi

log "Creating new cluster at ${PGDATA}"
pg_createcluster "${PG_VER}" "${CLUSTER_NAME}" --datadir="${PGDATA}"

log "Starting cluster: ${PG_VER}/${CLUSTER_NAME}"
pg_ctlcluster "${PG_VER}" "${CLUSTER_NAME}" start

log "Waiting for Postgres to accept connections..."
su - postgres -c "pg_isready -h 127.0.0.1 -p 5432" || true
for _ in {1..30}; do
  if su - postgres -c "pg_isready -h 127.0.0.1 -p 5432" | grep -q "accepting connections"; then
    break
  fi
  sleep 1
done
su - postgres -c "pg_isready -h 127.0.0.1 -p 5432" | grep -q "accepting connections"

log "Creating role '${OPTUNA_USER}' if not exists..."
su - postgres -c "psql -v ON_ERROR_STOP=1 -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${OPTUNA_USER}'\" | grep -q 1 \
  || psql -v ON_ERROR_STOP=1 -c \"CREATE USER ${OPTUNA_USER} WITH PASSWORD '${OPTUNA_PASS}';\""

log "Creating database '${OPTUNA_DB}' if not exists..."
su - postgres -c "psql -v ON_ERROR_STOP=1 -tAc \"SELECT 1 FROM pg_database WHERE datname='${OPTUNA_DB}'\" | grep -q 1 \
  || psql -v ON_ERROR_STOP=1 -c \"CREATE DATABASE ${OPTUNA_DB} OWNER ${OPTUNA_USER};\""

log "Verifying data_directory..."
su - postgres -c "psql -tAc \"show data_directory;\""

log "Done. Connection string:"
echo "postgresql+psycopg2://${OPTUNA_USER}:${OPTUNA_PASS}@127.0.0.1:5432/${OPTUNA_DB}"