#!/usr/bin/env bash
# llama-fleet — concurrency scaling benchmark.
#
# Usage: fleet/bench.sh [max_concurrency]      (default: the current shard count)
#
# WHY THIS EXISTS: the fleet's whole premise is that N independent shards beat one model
# pipelined across N cards. On this rig that premise has a ceiling — an 8-shard run on
# 2026-07-28 wedged every llama-server into unkillable D state with the CPU 91% idle, which is
# a GPU-driver level block, not CPU contention. So the shard count must be established by
# measurement, not assumed. This walks concurrency up 1 -> 2 -> 4 -> ... and stops the moment
# a level misbehaves, so a bad level costs one timeout instead of a reboot.
#
# Read the output as: if wall time stays flat as N doubles, the fleet is scaling. If wall time
# grows in proportion to N, you have hit the ceiling and the extra shards are buying nothing.
set -euo pipefail

ROOT="/home/botuser/Projects/llama.cpp"
STATE="${LLAMA_FLEET_STATE:-$HOME/.local/state/llama-fleet}"
# shellcheck disable=SC1091
. "$ROOT/fleet/cards.conf"
# shellcheck disable=SC1091
. "$STATE/plan.env"

MAXN="${1:-$SHARD_COUNT}"
KEY=$(head -1 "$ROOT/.qwen-api-key" 2>/dev/null || echo "")
BODY='{"model":"'"$(basename "$STATE" >/dev/null; echo ornith)"'","messages":[{"role":"user","content":"Count from 1 to 40, one number per line."}],"max_tokens":200,"temperature":0}'
# Per-request ceiling. Deliberately short: a healthy 200-token completion takes <15s on one
# shard, so 120s means something is wrong and we want to know fast rather than hang for 10min.
TIMEOUT=120

req() {
  curl -s -m "$TIMEOUT" "http://127.0.0.1:$1/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$BODY" -o /dev/null -w '%{http_code}'
}

echo "profile=$PROFILE shards=$SHARD_COUNT  (per-request timeout ${TIMEOUT}s)"
echo

# Warm every shard first: a cold first-touch would otherwise be charged to N=1.
echo "warming all $SHARD_COUNT shards..."
for i in $(seq 0 $((SHARD_COUNT - 1))); do req $((SHARD_PORT_BASE + i)) >/dev/null & done
wait
echo

printf '%-6s %-10s %-10s %-10s %s\n' N WALL THRUPUT VS-N=1 RESULT
base=""
n=1
while [ "$n" -le "$MAXN" ]; do
  s=$(date +%s.%N)
  rc=$(mktemp)
  for i in $(seq 0 $((n - 1))); do req $((SHARD_PORT_BASE + i)) >> "$rc" & done
  wait
  e=$(date +%s.%N)
  wall=$(echo "$e - $s" | bc)
  # `|| true` is load-bearing: grep exits 1 when it finds nothing, and under `set -e -o pipefail`
  # that would abort the benchmark exactly when every request SUCCEEDED.
  bad=$(tr -d '\n' < "$rc" | grep -o '000\|5[0-9][0-9]' | wc -l || true)
  rm -f "$rc"
  [ -z "$base" ] && base="$wall"

  # Flat ratio => real concurrency. Ratio tracking N => serialized, no benefit from more shards.
  ratio=$(echo "scale=2; $wall / $base" | bc)
  thru=$(echo "scale=2; $n / $wall" | bc)
  verdict=ok
  [ "$bad" -gt 0 ] && verdict="FAILED ($bad req)"
  printf '%-6s %-10s %-10s %-10s %s\n' "$n" \
    "$(printf '%.2fs' "$wall")" "${thru}/s" "${ratio}x" "$verdict"

  if [ "$bad" -gt 0 ]; then
    echo
    echo "STOP: concurrency $n produced failures. Do not raise MAX_SHARDS past $((n / 2))."
    echo "Check for the wedge signature:  ps -eo stat | grep -c '^D'   (should be 0)"
    exit 1
  fi
  n=$((n * 2))
  sleep 5
done

echo
echo "All levels up to $MAXN healthy."
echo "If WALL stayed roughly flat, shards are scaling; if it grew with N, the ceiling is lower."
echo "Confirm no wedge before trusting the run:  ps -eo stat --no-headers | grep -c '^D'"
