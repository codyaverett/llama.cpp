#!/usr/bin/env bash
# Launch llama-server for Qwen3.6-35B-A3B split across the 5 Vega cards.
# Usage: serve-qwen.sh [hip|vulkan]      (default: hip — best for interactive chat)
# Env overrides: LLAMA_HOST (default 127.0.0.1), LLAMA_PORT (8080), LLAMA_CTX (16384)
#
# Runs the server inside the 'render' group via sg, so it can reach /dev/kfd and the DRI
# render nodes even when the caller (e.g. the systemd --user manager) lacks that group.
set -euo pipefail

BACKEND="${1:-hip}"
ROOT="/home/botuser/Projects/llama.cpp"
MODEL="/home/botuser/Projects/models/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8080}"
CTX="${LLAMA_CTX:-16384}"
# API-key auth: if this file exists, require a Bearer token. Passed via --api-key-file
# (NOT --api-key) so the secret never appears in the process list on this shared box.
KEYFILE="${LLAMA_API_KEY_FILE:-$ROOT/.qwen-api-key}"

# GPU[6] (PCIe 0000:16:00.0, HSA node-7) reliably triggers a GPU memory access fault
# (VMFaultHandler) the moment a large per-card buffer lands on it — reproducible across
# -fit on/off and across model loads, always node-7. Exclude it; run on the other 9.
# Override with HIP_DEVICES / GGML_VK_VISIBLE_DEVICES if the card is repaired/replaced.
HIP_DEVS="${HIP_DEVICES:-0,1,2,3,4,5,7,8,9}"

case "$BACKEND" in
  hip)
    BIN="$ROOT/build-hip/bin/llama-server"
    PREFIX="HIP_VISIBLE_DEVICES=$HIP_DEVS"
    ;;
  vulkan)
    BIN="$ROOT/build-vulkan/bin/llama-server"
    PREFIX="GGML_VK_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9,10"   # skip Intel iGPU (Vulkan0) + llvmpipe (Vulkan11)
    ;;
  *)
    echo "unknown backend '$BACKEND' (use: hip | vulkan)" >&2
    exit 2
    ;;
esac

AUTH=""
if [ -f "$KEYFILE" ]; then
  AUTH="--api-key-file '$KEYFILE'"
  echo "API-key auth: ENABLED (keys from $KEYFILE)" >&2
else
  echo "API-key auth: DISABLED (no key file at $KEYFILE)" >&2
fi

# Flash Attention keeps the attention compute buffer flat as context grows (so VRAM is
# dominated by the KV cache only). Required for KV-cache quantization below.
# Optional: set LLAMA_CACHE_TYPE=q8_0 to halve KV size (near-lossless) for 256k context.
CACHE=""
if [ -n "${LLAMA_CACHE_TYPE:-}" ]; then
  CACHE="--cache-type-k $LLAMA_CACHE_TYPE --cache-type-v $LLAMA_CACHE_TYPE"
fi

# -fit off: the auto "fitting params to device memory" probe triggers a GPU memory access
# fault (VMFaultHandler) on this 10x gfx900 rig (see the load log: "fitting params to device
# memory ..." right before the crash-restart loop that returns 503 "Loading model"). Disabling
# it makes the loader trust the explicit -ngl 99 -sm layer placement instead of probing.
# --no-mmap + --no-warmup: this box has 31 GB RAM and the model is 25 GB, so with mmap the
# kernel thrashes weight pages off the slow SATA SSD (port binds but never finishes loading).
# --no-mmap streams tensors straight to VRAM (all layers offloaded) so host RAM never holds the
# whole file; --no-warmup skips the full-weight warmup decode that would re-touch everything.
# --parallel: llama.cpp splits -c across N slots, so N=4 gives each request only CTX/4 tokens.
# Default to 4 to preserve the prior behaviour (no --parallel == 4); set LLAMA_PARALLEL to change.
NP="${LLAMA_PARALLEL:-4}"
echo "Starting Qwen3.6 llama-server [$BACKEND] on $HOST:$PORT (ctx=$CTX, parallel=$NP, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), 9x Vega layer-split, no-mmap" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' -ngl 99 -sm layer -fit off -fa on --no-mmap --no-warmup -c $CTX --parallel $NP --host $HOST --port $PORT $AUTH $CACHE"
