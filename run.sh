#!/usr/bin/env bash
# Launch the REINVENT4 + GNINA web app.
#
# Serves on port 5012 from the `gnina_webapp` conda env (FastAPI + RDKit).
# The app shells out to the reinvent4, gnina-dock and openmmdl envs as needed.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${REINVENT_GNINA_PORT:-5012}"
HOST="${REINVENT_GNINA_HOST:-0.0.0.0}"
ENV_PY="/home/evehom/Programs/miniconda3/envs/gnina_webapp/bin"

exec "${ENV_PY}/uvicorn" reinvent_webapp:app \
    --host "${HOST}" --port "${PORT}" \
    --workers 1 --log-level info
