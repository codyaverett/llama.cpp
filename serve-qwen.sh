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

case "$BACKEND" in
  hip)
    BIN="$ROOT/build-hip/bin/llama-server"
    PREFIX=""
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

echo "Starting Qwen3.6 llama-server [$BACKEND] on $HOST:$PORT (ctx=$CTX, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), 5x Vega layer-split" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' -ngl 99 -sm layer -fa on -c $CTX --host $HOST --port $PORT $AUTH $CACHE"
