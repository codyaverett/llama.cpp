#!/usr/bin/env bash
# Launch llama-server for GLM-4.5-Air (106B-A12B MoE, arch glm4moe) across the 10 Vega cards.
# Usage: serve-glm.sh [hip|vulkan]       (default: hip — best for interactive chat)
# Env overrides: LLAMA_HOST (default 127.0.0.1), LLAMA_PORT (8089), LLAMA_CTX (8192),
#                LLAMA_CACHE_TYPE (e.g. q8_0 to halve KV and buy more context).
#
# Drop-in replacement for serve-qwen.sh on the SAME port (8089): only one model fits in
# 80 GB at a time, so llama-glm and llama-qwen must not run simultaneously.
#
# Runs inside the 'render' group via sg so it can reach /dev/kfd even when the systemd
# --user manager lacks that group.
set -euo pipefail

BACKEND="${1:-hip}"
ROOT="/home/botuser/Projects/llama.cpp"
# Split GGUF: point -m at the FIRST shard; llama.cpp auto-loads 00002-of-00002.
MODEL="/home/botuser/Projects/models/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8089}"
CTX="${LLAMA_CTX:-8192}"
# API-key auth (same key file as Qwen): passed via --api-key-file so the secret never
# appears in the process list on this shared box.
KEYFILE="${LLAMA_API_KEY_FILE:-$ROOT/.qwen-api-key}"

case "$BACKEND" in
  hip)
    BIN="$ROOT/build-hip/bin/llama-server"
    PREFIX=""
    ;;
  vulkan)
    BIN="$ROOT/build-vulkan/bin/llama-server"
    PREFIX="GGML_VK_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9,10"   # skip Intel iGPU (Vulkan0) + llvmpipe
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

# Flash Attention keeps the attention compute buffer flat as context grows (VRAM is then
# dominated by the KV cache only). Required for KV-cache quantization below.
# GLM-4.5-Air weights (~68 GiB UD-Q4_K_XL) leave little per-card headroom on 8 GiB cards,
# so context is modest unless LLAMA_CACHE_TYPE=q8_0 is set to halve the KV footprint.
CACHE=""
if [ -n "${LLAMA_CACHE_TYPE:-}" ]; then
  CACHE="--cache-type-k $LLAMA_CACHE_TYPE --cache-type-v $LLAMA_CACHE_TYPE"
fi

echo "Starting GLM-4.5-Air llama-server [$BACKEND] on $HOST:$PORT (ctx=$CTX, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), 10x Vega layer-split" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' -ngl 99 -sm layer -fa on -c $CTX --host $HOST --port $PORT $AUTH $CACHE"
