#!/usr/bin/env bash
# ClinNoteGuard Dual-Server Launcher
# Checks if Django (:8000) and WebSocket (:8001) are already running:
#   - If already running, prints connection details (IP & ports) and exits cleanly.
#   - Otherwise starts any missing server(s) and prints connection details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="./venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Error: Virtualenv Python not found at $SCRIPT_DIR/venv/bin/python"
  exit 1
fi

LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' || true)"
if [ -z "$LAN_IP" ]; then
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
LAN_IP="${LAN_IP:-127.0.0.1}"

is_port_listening() {
  local port="$1"
  "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', int(sys.argv[1])))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
" "$port" 2>/dev/null
}

DJANGO_UP=false
WS_UP=false
is_port_listening 8000 && DJANGO_UP=true
is_port_listening 8001 && WS_UP=true

print_connection_info() {
  local dj_status="$1"
  local ws_status="$2"
  echo "============================================================"
  echo " ClinNoteGuard Backend — Connection Details for Mobile App"
  echo "============================================================"
  echo "  Server IP / Host : $LAN_IP"
  echo "  Django Port      : 8000   ($dj_status)"
  echo "  WebSocket Port   : 8001   ($ws_status)"
  echo "------------------------------------------------------------"
  echo "  Django REST API  : http://$LAN_IP:8000"
  echo "  WebSocket Audio  : ws://$LAN_IP:8001"
  echo "============================================================"
}

if $DJANGO_UP && $WS_UP; then
  print_connection_info "ALREADY RUNNING" "ALREADY RUNNING"
  echo "Both servers are already running. Nothing to start."
  exit 0
fi

# Ensure virtualenv cuBLAS libs are on LD_LIBRARY_PATH for CTranslate2 / faster-whisper
NV_CU13="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "$NV_CU13" ]; then
  export LD_LIBRARY_PATH="$NV_CU13:${LD_LIBRARY_PATH:-}"
fi

"$PYTHON" manage.py migrate --noinput

PIDS=()
cleanup() {
  if [ "${#PIDS[@]}" -gt 0 ]; then
    echo ""
    echo "Stopping ClinNoteGuard servers started by this script..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
    wait 2>/dev/null || true
    echo "Stopped."
  fi
}
trap cleanup INT TERM EXIT

DJ_STATUS="ALREADY RUNNING"
WS_STATUS="ALREADY RUNNING"

if ! $DJANGO_UP; then
  "$PYTHON" manage.py runserver 0.0.0.0:8000 &
  PIDS+=($!)
  DJ_STATUS="STARTING"
fi

if ! $WS_UP; then
  "$PYTHON" manage.py run_websocket_server --port 8001 &
  PIDS+=($!)
  WS_STATUS="STARTING"
fi

print_connection_info "$DJ_STATUS" "$WS_STATUS"
echo "Press Ctrl+C to stop."

wait -n "${PIDS[@]}"
