#!/usr/bin/env bash
# Load GLM-4.5-Air (106B-A12B MoE) across all 10 Vega cards and measure:
#   1) it fits (VRAM headroom per card), 2) generation speed, 3) stability under
#   a sustained long generation (stresses the freshly-expanded power/thermal envelope).
# Prereq: stop llama-qwen first (frees ~27 GB so the ~68 GB model fits in 80 GB).
set -uo pipefail

ROOT="/home/botuser/Projects/llama.cpp"
BIN="$ROOT/build-hip/bin/llama-server"
MODEL="/home/botuser/Projects/models/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"  # split: first part
HOST=127.0.0.1; PORT=8091
CTX=32768

cleanup_server() {
  pkill -f "llama-server .*--port $PORT" 2>/dev/null
  for i in $(seq 1 60); do pgrep -f "llama-server .*--port $PORT" >/dev/null || break; sleep 1; done
  for i in $(seq 1 60); do ss -tln 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
}

echo "### GLM-4.5-Air bench — $(date)"
cleanup_server
LOG=/tmp/glm-air.server.log
sg render -c "exec '$BIN' -m '$MODEL' -ngl 99 -sm layer -fa on -c $CTX --host $HOST --port $PORT" > "$LOG" 2>&1 &

echo ">>> loading across 10 cards (cold load of ~68 GB can take several minutes)..."
ok=0
for i in $(seq 1 600); do
  curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"ok"' && { ok=1; echo "healthy after ${i}s"; break; }
  grep -qiE 'error|failed|out of memory|cuda|hip error' "$LOG" 2>/dev/null && { echo "!! load error:"; grep -iE 'error|failed|out of memory|hip' "$LOG" | tail -8 | sed 's/\x1b\[[0-9;]*m//g'; }
  sleep 1
done
if [ "$ok" != 1 ]; then echo "!! never healthy; tail:"; tail -20 "$LOG" | sed 's/\x1b\[[0-9;]*m//g'; cleanup_server; exit 1; fi

echo ">>> arch + model the server loaded:"
curl -s "http://$HOST:$PORT/props" 2>/dev/null | jq -r '"   model: "+(.model_path // "?")' 2>/dev/null
echo ">>> VRAM per card after load (watch for any card near 8 GB = risky):"
rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used' | awk '{printf "   %s %.2f GiB\n",$1,$NF/1073741824}'

# quick speed: code (high MoE locality) + prose, 256 tokens, greedy
for pname in CODE PROSE; do
  if [ "$pname" = CODE ]; then P="Write a complete, well-commented Python implementation of an LRU cache with O(1) get/put."; else P="Explain the main causes of the French Revolution in several clear paragraphs."; fi
  body="$(jq -nc --arg p "$P" '{prompt:$p,n_predict:256,temperature:0,top_k:1,seed:42,cache_prompt:false}')"
  curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' -d "$body" >/dev/null  # warmup
  resp="$(curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' -d "$body")"
  echo "   [$pname] $(echo "$resp" | jq -c '{tg_per_s:(.timings.predicted_per_second|.*100|round/100), pp_per_s:(.timings.prompt_per_second|.*100|round/100), n:.timings.predicted_n}')"
done

# sustained stress: one long generation; if power/thermals are marginal this is where it dies
echo ">>> sustained-load stress (1024 tokens)..."
body="$(jq -nc '{prompt:"Write a long, detailed technical essay about the history of GPU computing.",n_predict:1024,temperature:0.7,seed:1,cache_prompt:false}')"
resp="$(curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' -d "$body")"
echo "   [STRESS] $(echo "$resp" | jq -c '{tg_per_s:(.timings.predicted_per_second|.*100|round/100), n:.timings.predicted_n, stopped:.stop_type}' 2>/dev/null || echo "NO RESPONSE — possible crash")"
echo ">>> server still alive after stress? $(pgrep -f "llama-server .*--port $PORT" >/dev/null && echo YES || echo NO-CRASHED)"

cleanup_server
echo "### done"
