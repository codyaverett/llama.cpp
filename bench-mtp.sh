#!/usr/bin/env bash
# Benchmark Qwen3.6-35B-A3B: current (non-MTP) vs MTP build with self-speculative decoding.
# MTP is only exercised by llama-server (--spec-type draft-mtp), NOT llama-bench, so we
# drive real generation through the server and read its timings (incl. draft acceptance).
#
# Prereq: the llama-qwen service must be STOPPED first (it occupies ~38/40 GB VRAM).
#   systemctl --user stop llama-qwen
set -uo pipefail

ROOT="/home/botuser/Projects/llama.cpp"
MODELS="/home/botuser/Projects/models"
BIN="$ROOT/build-hip/bin/llama-server"
CUR="$MODELS/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"        # current (no MTP head)
MTP="$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q5_K_M.gguf"    # same quant + MTP head
HOST=127.0.0.1; PORT=8091
CTX=4096; NGEN=256

read -r -d '' P_CODE <<'EOF'
Write a complete, well-commented Python implementation of an LRU cache class using a
doubly linked list and a hash map, supporting get(key) and put(key, value) in O(1).
EOF
read -r -d '' P_PROSE <<'EOF'
Explain, in several clear paragraphs, the main long-term and immediate causes of the
French Revolution, and how they interacted to bring about the collapse of the monarchy.
EOF

# Robustly kill any llama-server on our port (the sg wrapper orphans the real process,
# so we must match llama-server itself), then wait for the port to actually free.
cleanup_server() {
  pkill -f "llama-server .*--port $PORT" 2>/dev/null
  for i in $(seq 1 40); do
    pgrep -f "llama-server .*--port $PORT" >/dev/null || break
    sleep 1
  done
  for i in $(seq 1 40); do
    ss -tln 2>/dev/null | grep -q ":$PORT " || break
    sleep 1
  done
}

run_cfg() {  # $1=label  $2=model  $3=extra server args
  local label="$1" model="$2" extra="$3" log="/tmp/bench-$1.server.log"
  echo "=================================================================="
  echo ">>> $label"
  echo "    model: $(basename "$model")"
  echo "    extra: ${extra:-<none>}"
  cleanup_server   # guarantee a clean slate before starting
  sg render -c "exec '$BIN' -m '$model' -ngl 99 -sm layer -fa on -c $CTX \
    --host $HOST --port $PORT $extra" > "$log" 2>&1 &

  # wait for OUR server to bind + load (health ok). 240s budget for a cold 5-card load.
  local ok=0
  for i in $(seq 1 240); do
    if curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    if grep -q 'HTTP server error' "$log" 2>/dev/null; then
      echo "    !! server failed to bind port (stale server?). tail:"; tail -6 "$log" | sed 's/\x1b\[[0-9;]*m//g'; cleanup_server; return 1
    fi
    sleep 1
  done
  [ "$ok" = 1 ] || { echo "    !! never healthy. tail:"; tail -12 "$log" | sed 's/\x1b\[[0-9;]*m//g'; cleanup_server; return 1; }

  # GUARD: confirm the server we reached actually loaded the expected model file.
  local got; got="$(curl -s "http://$HOST:$PORT/props" 2>/dev/null | jq -r '.model_path // .default_generation_settings.model // empty' 2>/dev/null)"
  echo "    server reports model: $(basename "${got:-?}")"
  if [ -n "$got" ] && [ "$(basename "$got")" != "$(basename "$model")" ]; then
    echo "    !! WRONG MODEL on port (stale). aborting this config."; cleanup_server; return 1
  fi
  # GUARD: if MTP requested, confirm the draft-mtp impl was actually added.
  if echo "$extra" | grep -q 'draft-mtp'; then
    if grep -qiE "speculative implementation 'draft-mtp'|draft-mtp" "$log"; then
      echo "    MTP: draft-mtp implementation ACTIVE"
    else
      echo "    !! MTP requested but draft-mtp NOT active in log — check setup"
    fi
  fi

  for pname in CODE PROSE; do
    local prompt; prompt="$(eval echo "\"\$P_$pname\"")"
    local body; body="$(jq -nc --arg p "$prompt" --argjson n $NGEN \
            '{prompt:$p,n_predict:$n,temperature:0,top_k:1,seed:42,cache_prompt:false}')"
    curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' -d "$body" >/dev/null  # warmup
    local resp; resp="$(curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' -d "$body")"
    echo "    [$pname] $(echo "$resp" | jq -c '{tg_per_s: (.timings.predicted_per_second|.*100|round/100),
        pp_per_s: (.timings.prompt_per_second|.*100|round/100), n_pred: .timings.predicted_n,
        draft_n: (.timings.draft_n // 0), draft_acc: (.timings.draft_n_accepted // 0),
        accept_pct: (if (.timings.draft_n // 0) > 0 then (1000.0*.timings.draft_n_accepted/.timings.draft_n|round/10) else null end)}')"
  done
  cleanup_server
}

echo "### MTP benchmark — $(date)"
run_cfg "A_current_noMTP" "$CUR" ""
run_cfg "B_mtp_specOFF"   "$MTP" ""
run_cfg "C_mtp_specON"    "$MTP" "--spec-type draft-mtp --spec-draft-n-max 4"
echo "### done"
