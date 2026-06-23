#!/usr/bin/env bash
# Unified model switch for the Vega rig. The 8 GB cards can't hold two models at once, so this
# runs EXACTLY ONE of:
#   qwen     -> Qwen3.6-35B-A3B (MoE, ~3B active) — FAST daily driver (~25 t/s), port 8089
#   deckard  -> Qwen3.6-40B Deckard-Opus (DENSE 40B) — QUALITY/thinking (~6.8 t/s), port 8090
# It stops whatever is resident (incl. the llama-qwen systemd service), waits for VRAM to free,
# then starts the pick via the existing serve-qwen.sh / serve-deckard.sh launchers.
#
# Usage:
#   serve.sh qwen    [hip|vulkan]     # switch to Qwen
#   serve.sh deckard [hip|vulkan]     # switch to Deckard-40B
#   serve.sh stop                     # stop everything, free all VRAM
#   serve.sh status                   # what's running + VRAM
#   serve.sh restart                  # restart whatever is currently up
# Env passthrough to the launchers: LLAMA_CTX, LLAMA_CACHE_TYPE, LLAMA_PORT, HIP_DEVICES, ...
#   e.g.  LLAMA_CTX=16384 serve.sh deckard
set -euo pipefail

ROOT="/home/botuser/Projects/llama.cpp"
LOGDIR="/home/botuser/Projects/models"
SERVER_PAT="bin/llama-server"            # matches the server binary, NOT this script's cmdline

vram_used_gb() {   # total VRAM used across all cards, integer GB (best-effort)
  timeout 12 rocm-smi --showmeminfo vram 2>/dev/null \
    | awk '/Used Memory/ {sub(/.*: /,""); s+=$1} END {printf "%.0f", s/1e9}'
}

running_model() {   # echo qwen | deckard | none — match by model file, robust to port/host
  local cmd; cmd=$(pgrep -af "$SERVER_PAT" 2>/dev/null || true)
  if   echo "$cmd" | grep -q 'Deck-Opus'; then echo deckard
  elif echo "$cmd" | grep -q '35B-A3B';   then echo qwen
  elif systemctl --user is-active --quiet llama-qwen 2>/dev/null; then echo qwen
  else echo none; fi
}

stop_all() {
  echo ">> stopping any resident model..." >&2
  systemctl --user stop llama-qwen 2>/dev/null || true
  # Kill manual servers (deckard, or a non-service qwen). Pattern matches the binary only,
  # so this script's own cmdline ("bash serve.sh ...") is never a target.
  pkill -f "$SERVER_PAT" 2>/dev/null || true
  # Wait up to ~30s for the processes to die and VRAM to drain.
  for _ in $(seq 1 30); do
    pgrep -f "$SERVER_PAT" >/dev/null 2>&1 || break
    sleep 1
  done
  pkill -9 -f "$SERVER_PAT" 2>/dev/null || true
  for _ in $(seq 1 20); do
    u=$(vram_used_gb); [ "${u:-99}" -le 2 ] && break
    sleep 1
  done
  echo ">> VRAM used after stop: $(vram_used_gb) GB" >&2
}

wait_ready() {   # poll a log for "server is listening"; args: <logfile> <timeout_s>
  local log="$1" timeout="${2:-600}" i
  for ((i=0; i<timeout; i+=5)); do
    grep -q "server is listening" "$log" 2>/dev/null && { echo ">> READY (listening)"; return 0; }
    grep -qiE "VMFaultHandler|Memory access fault|error loading|out of memory|terminate called" "$log" 2>/dev/null \
      && { echo "!! load failed — see $log" >&2; tail -5 "$log" >&2; return 1; }
    pgrep -f "$SERVER_PAT" >/dev/null 2>&1 || { echo "!! server exited — see $log" >&2; tail -5 "$log" >&2; return 1; }
    sleep 5
  done
  echo "!! still loading after ${timeout}s — check $log" >&2; return 1
}

start_qwen() {
  local backend="$1"
  if [ "$backend" = "hip" ]; then
    echo ">> starting Qwen via systemd (llama-qwen)..." >&2
    systemctl --user start llama-qwen
    echo ">> Qwen starting on :8080 — watch: journalctl --user -u llama-qwen -f" >&2
  else
    echo ">> starting Qwen [$backend] (manual)..." >&2
    nohup "$ROOT/serve-qwen.sh" "$backend" > "$LOGDIR/.qwen-serve.log" 2>&1 &
    wait_ready "$LOGDIR/.qwen-serve.log" 300 || true
  fi
}

start_deckard() {
  local backend="$1"
  echo ">> starting Deckard-40B [$backend] on :8090 (dense, ~6.8 t/s, this takes a couple minutes)..." >&2
  nohup "$ROOT/serve-deckard.sh" "$backend" > "$LOGDIR/.deckard-serve.log" 2>&1 &
  wait_ready "$LOGDIR/.deckard-serve.log" 600 || true
}

show_status() {
  local cur; cur=$(running_model)
  echo "resident model : $cur"
  echo "VRAM used      : $(vram_used_gb) GB"
  echo "llama-qwen svc : $(systemctl --user is-active llama-qwen 2>/dev/null) / $(systemctl --user is-enabled llama-qwen 2>/dev/null)"
  pgrep -af "$SERVER_PAT" 2>/dev/null | sed 's/--api-key-file [^ ]*/--api-key-file ***/' | grep -oE "llama-server .*--port [0-9]+" || true
}

MODEL="${1:-status}"
BACKEND="${2:-hip}"

case "$MODEL" in
  qwen)
    [ "$(running_model)" = qwen ] && { echo "Qwen already resident."; show_status; exit 0; }
    stop_all; start_qwen "$BACKEND"; echo; show_status ;;
  deckard)
    [ "$(running_model)" = deckard ] && { echo "Deckard already resident."; show_status; exit 0; }
    stop_all; start_deckard "$BACKEND"; echo; show_status ;;
  stop)
    stop_all; echo; show_status ;;
  restart)
    cur=$(running_model)
    [ "$cur" = none ] && { echo "Nothing running."; exit 0; }
    stop_all; [ "$cur" = deckard ] && start_deckard "$BACKEND" || start_qwen "$BACKEND"; echo; show_status ;;
  status)
    show_status ;;
  *)
    echo "usage: serve.sh <qwen|deckard|stop|restart|status> [hip|vulkan]" >&2; exit 2 ;;
esac
