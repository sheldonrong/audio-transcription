#!/bin/sh
set -eu

python3 - <<'PY' > /app/frontend/dist/config.js
import json
import os

config = {
    "API_BASE_URL": os.getenv("API_BASE_URL", "http://localhost:8000"),
    "WS_URL": os.getenv("WS_URL", "ws://localhost:8000"),
}

v2a_ws_url = os.getenv("V2A_WS_URL", "").strip()
if v2a_ws_url:
    config["V2A_WS_URL"] = v2a_ws_url

print("window.APP_CONFIG = " + json.dumps(config, separators=(",", ":")) + ";")
PY

exec "$@"
