#!/usr/bin/env bash
# llama-fleet — launch ONE shard of the active profile.
#
# Usage: serve-shard.sh <shard-index>
#
# This is the model-agnostic replacement for the seven bespoke serve-*.sh scripts. Everything
# model-specific lives in fleet/profiles/<name>.conf; everything rig-specific lives here.
#
# The shard does NOT decide which cards it gets. `fleet use` / `fleet probe` resolve the whole
# shard->card map up front and write it to $STATE/plan.env; this script just looks up its own
# row. That way the health probe and the launcher can never disagree about placement.
#
# Runs the server inside the 'render' group via sg, so it can reach /dev/kfd and the DRI render
# nodes even when the caller (the systemd --user manager) lacks that group.
set -euo pipefail

IDX="${1:?usage: serve-shard.sh <shard-index>}"
ROOT="/home/botuser/Projects/llama.cpp"
MODELS_DIR="/home/botuser/Projects/models"
STATE="${LLAMA_FLEET_STATE:-$HOME/.local/state/llama-fleet}"

[ -f "$STATE/plan.env" ] || { echo "ERROR: no resolved plan at $STATE/plan.env — run 'fleet use <profile>' first" >&2; exit 1; }
# shellcheck disable=SC1091
. "$STATE/plan.env"

# Pull this shard's row out of the plan. Written by fleet.sh as SHARD_<n>_CARDS / _PORT.
eval "CARDS=\${SHARD_${IDX}_CARDS:-}"
eval "PORT=\${SHARD_${IDX}_PORT:-}"
[ -n "$CARDS" ] || { echo "ERROR: shard $IDX is not in the current plan (SHARD_COUNT=${SHARD_COUNT:-0})" >&2; exit 1; }

PROFILE_FILE="$ROOT/fleet/profiles/${PROFILE}.conf"
[ -f "$PROFILE_FILE" ] || { echo "ERROR: profile '$PROFILE' not found at $PROFILE_FILE" >&2; exit 1; }

# Profile defaults, then the profile overrides them.
ALIAS=""; CTX=16384; PARALLEL=1; CACHE_TYPE=""; MMAP=off; EXTRA=""; MMPROJ=""; BACKEND=hip
THREADS=""; THREADS_HTTP=""
# shellcheck disable=SC1090
. "$PROFILE_FILE"

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
ALIAS="${ALIAS:-$PROFILE}"

case "$BACKEND" in
  hip)    BIN="$ROOT/build-hip/bin/llama-server" ;;
  vulkan) BIN="$ROOT/build-vulkan/bin/llama-server" ;;
  *) echo "unknown BACKEND '$BACKEND' (use: hip | vulkan)" >&2; exit 2 ;;
esac
[ -x "$BIN" ] || { echo "ERROR: llama-server not built at $BIN" >&2; exit 1; }

# HIP_VISIBLE_DEVICES remaps: exposing "3" makes that card device 0 inside this process, so a
# single-card shard is always -mg 0. Multi-card shards keep the layer split across what they see.
NCARDS=$(echo "$CARDS" | tr ',' ' ' | wc -w)
if [ "$NCARDS" -eq 1 ]; then
  SPLIT="-sm none -mg 0"
else
  SPLIT="-sm layer"
fi

# API-key auth: if the key file exists, require a Bearer token. Passed via --api-key-file
# (NOT --api-key) so the secret never appears in the process list on this shared box.
KEYFILE="${LLAMA_API_KEY_FILE:-$ROOT/.qwen-api-key}"
AUTH=""
if [ -f "$KEYFILE" ]; then
  AUTH="--api-key-file '$KEYFILE'"
fi

# Flash Attention keeps the attention compute buffer flat as context grows, so VRAM is
# dominated by the KV cache alone. Also required for KV-cache quantization.
CACHE=""
if [ -n "$CACHE_TYPE" ]; then
  CACHE="--cache-type-k $CACHE_TYPE --cache-type-v $CACHE_TYPE"
fi

# mmap is a per-model decision, which is why it is a profile knob rather than a fixed flag:
#   MMAP=off — model is larger than free host RAM (31 GB total on this box). With mmap the
#              kernel thrashes weight pages off the slow SATA SSD and the port binds but the
#              load never finishes. --no-mmap streams tensors straight to VRAM instead.
#   MMAP=on  — model comfortably fits RAM. Now mmap is strictly BETTER for a fleet: all shards
#              share one page-cache copy of the file instead of N processes each reading their
#              own, which is the difference between one disk read and nine concurrent ones.
# --no-warmup skips the full-weight warmup decode that would re-touch every tensor.
MMAP_FLAGS="--no-warmup"
if [ "$MMAP" = "off" ]; then
  MMAP_FLAGS="--no-mmap --no-warmup"
fi

# -fit off: the auto "fitting params to device memory" probe triggers a GPU memory access fault
# (VMFaultHandler) on this 10x gfx900 rig — see the load log, "fitting params to device memory"
# lands right before the crash-restart loop that returns 503 "Loading model". Disabling it makes
# the loader trust the explicit -ngl 99 + split placement instead of probing.
MM=""
if [ -n "$MMPROJ" ]; then
  [ -f "$MMPROJ" ] || { echo "ERROR: mmproj not found: $MMPROJ" >&2; exit 1; }
  MM="--mmproj '$MMPROJ'"
fi

# THE CPU IS THE FLEET'S REAL CEILING, not the GPUs. This box is a 4-core i3-9100, and
# llama-server defaults to 4 generation threads + 6 HTTP threads PER PROCESS. Eight shards at
# those defaults put ~80 threads on 4 cores; measured result was 100% user CPU, an 8-deep run
# queue, and requests that never returned — far WORSE than serializing on one model. ROCm's
# synchronization also busy-waits, so a GPU-bound shard still burns a core.
#
# With -ngl 99 every layer is on the GPU, so generation needs almost no CPU: 1 thread is right.
# Leave THREADS empty to accept llama.cpp's own default (correct for a 1-shard profile).
THREAD_FLAGS=""
[ -n "$THREADS" ]      && THREAD_FLAGS="$THREAD_FLAGS -t $THREADS -tb $THREADS"
[ -n "$THREADS_HTTP" ] && THREAD_FLAGS="$THREAD_FLAGS --threads-http $THREADS_HTTP"

echo "shard $IDX: profile=$PROFILE alias=$ALIAS cards=[$CARDS] port=$PORT ctx=$CTX parallel=$PARALLEL mmap=$MMAP${CACHE_TYPE:+ kv=$CACHE_TYPE}${THREADS:+ threads=$THREADS}" >&2

exec sg render -c "exec env HIP_VISIBLE_DEVICES=$CARDS '$BIN' \
  -m '$MODEL' --alias '$ALIAS' \
  -ngl 99 $SPLIT -fit off -fa on $MMAP_FLAGS \
  -c $CTX --parallel $PARALLEL $THREAD_FLAGS \
  --host 127.0.0.1 --port $PORT \
  $AUTH $CACHE $MM $EXTRA"
