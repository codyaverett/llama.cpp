#!/usr/bin/env bash
# Launch llama-server for Ornith-1.0-35B (Qwen3.5 MoE, ~3B active/token; agentic coding)
# split across the 9 usable Vega cards.
# Usage: serve-ornith.sh [hip|vulkan] [base|aeon]   (defaults: hip, base — best for interactive chat)
# Env overrides: LLAMA_HOST (default 127.0.0.1), LLAMA_PORT (8092), LLAMA_CTX (16384),
#                LLAMA_ALIAS (per-variant default), LLAMA_VARIANT (base|aeon; or pass as 2nd arg)
#
# VARIANTS (select with LLAMA_VARIANT=... or the 2nd positional arg):
#   base  deepreinforce-ai/Ornith-1.0-35B-GGUF (Q5_K_M, 24.7 GB) — the stock agentic coder.
#   aeon  vcruz305/Ornith-1.0-35B-AEON-Ultimate-Uncensored-GGUF (Q4_K_M, 21.2 GB) — uncensored
#         finetune; the AEON repo's own weights are NVFP4 (Blackwell-only), so we use this
#         community GGUF re-quant. Q4_K_M is the best PLAIN quant the repo ships (no Q5_K_M).
#         The MTP quants in that repo need newer multi-token-prediction support — not used here.
# Both are the same qwen3_5_moe arch, so they layer-split across the cards identically.
#
# Runs the server inside the 'render' group via sg, so it can reach /dev/kfd and the DRI
# render nodes even when the caller (e.g. the systemd --user manager) lacks that group.
set -euo pipefail

BACKEND="${1:-hip}"
ROOT="/home/botuser/Projects/llama.cpp"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8092}"   # 8089 Qwen/GLM, 8090 Deckard, 8091 GLM-air are taken
CTX="${LLAMA_CTX:-16384}"   # Ornith supports 256K; raise with LLAMA_CACHE_TYPE=q8_0 KV for big windows

# Variant select: 2nd positional arg wins, else LLAMA_VARIANT, else "base". Sets MODEL + the
# default served-model id (ALIAS), so base and aeon expose distinct ids on /v1/models.
VARIANT="${2:-${LLAMA_VARIANT:-base}}"
case "$VARIANT" in
  base) MODEL="/home/botuser/Projects/models/ornith-1.0-35b-Q5_K_M.gguf";    DEF_ALIAS="ornith" ;;
  aeon) MODEL="/home/botuser/Projects/models/ornith-aeon-35b-Q4_K_M.gguf";   DEF_ALIAS="ornith-aeon" ;;
  *)    echo "unknown variant '$VARIANT' (use: base | aeon)" >&2; exit 2 ;;
esac
# Served model id exposed via /v1/models. Cline/OpenAI clients use this as the "Model ID".
ALIAS="${LLAMA_ALIAS:-$DEF_ALIAS}"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model for variant '$VARIANT' not found: $MODEL" >&2
  exit 1
fi
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
# Default to 4 for this multi-client MoE backend; set LLAMA_PARALLEL=1 for full ctx in one slot.
NP="${LLAMA_PARALLEL:-4}"
echo "Starting Ornith-1.0-35B [$VARIANT] llama-server [$BACKEND] on $HOST:$PORT (alias=$ALIAS, ctx=$CTX, parallel=$NP, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), 9x Vega layer-split, no-mmap" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' --alias '$ALIAS' -ngl 99 -sm layer -fit off -fa on --no-mmap --no-warmup -c $CTX --parallel $NP --host $HOST --port $PORT $AUTH $CACHE"
