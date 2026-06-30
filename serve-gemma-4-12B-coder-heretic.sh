#!/usr/bin/env bash
# Launch llama-server for <gemma-4-12B-coder-fable5-composer2.5-v1-uncensored-heretic-Q8_0> split across the Vega cards.
# Usage: serve-newmodel.sh [hip|vulkan]      (default: hip — best for interactive chat)
# Env overrides: LLAMA_HOST (default 127.0.0.1), LLAMA_PORT (8091), LLAMA_CTX (16384)
#
# Scaffolded from serve-qwen.sh — carries the 10x gfx900 rig workarounds (-fit off, bad-card
# exclusion, --no-mmap). Fill in the TODOs after downloading the GGUF, then rename this file.
#
# Runs the server inside the 'render' group via sg, so it can reach /dev/kfd and the DRI
# render nodes even when the caller (e.g. the systemd --user manager) lacks that group.
set -euo pipefail

BACKEND="${1:-hip}"
ROOT="/home/botuser/Projects/llama.cpp"
MODELS_DIR="/home/botuser/Projects/models"
#
# ── Downloading a GGUF from Hugging Face ───────────────────────────────────────────────────
# llama.cpp needs GGUF files. Pick a quant that fits VRAM (8 GB/card x 9-10 cards ~= 72-80 GB
# total for weights + KV). Unsloth "UD" (Dynamic) quants are what this rig already runs.
# Check disk first:  df -h "$MODELS_DIR"     (these files are 25-68 GB)
#
#  A) huggingface-cli — puts a real file exactly where you want it (recommended):
#       pip install -U "huggingface_hub[cli]"          # once
#       huggingface-cli download <user/repo> <FILE.gguf> --local-dir "$MODELS_DIR"
#     For a SPLIT gguf, list EVERY shard (00001-of-000NN ... last) in the same command:
#       huggingface-cli download <user/repo> \
#         <model>-00001-of-00002.gguf <model>-00002-of-00002.gguf --local-dir "$MODELS_DIR"
#     Gated/private repos: run `huggingface-cli login` first (or pass --token).
#
#  B) wget/curl — no tools needed; -c resumes partial pulls (important on the slow SATA SSD):
#       wget -c -P "$MODELS_DIR" \
#         https://huggingface.co/<user/repo>/resolve/main/<FILE.gguf>
#
#  C) Built-in downloader — try-before-you-commit; caches under ~/.cache/llama.cpp (no -m path):
#       "$BIN" -hf <user/repo>:<QUANT>   # e.g. unsloth/GLM-4.5-Air-GGUF:Q4_K_XL
#
# TODO: set this to the downloaded GGUF. For a SPLIT gguf, point at shard 00001-of-000NN;
#       llama.cpp auto-loads the remaining shards.
MODEL="$MODELS_DIR/gemma-4-12B-coder-fable5-composer2.5-v1-uncensored-heretic-Q8_0.gguf"
HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8091}"   # TODO: confirm a free port (8089 Qwen/GLM, 8090 Deckard are taken)
CTX="${LLAMA_CTX:-262144}"
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
# Optional: set LLAMA_CACHE_TYPE=q8_0 to halve KV size (near-lossless) for big context.
CACHE=""
if [ -n "${LLAMA_CACHE_TYPE:-}" ]; then
  CACHE="--cache-type-k $LLAMA_CACHE_TYPE --cache-type-v $LLAMA_CACHE_TYPE"
fi

# -fit off: the auto "fitting params to device memory" probe triggers a GPU memory access
# fault (VMFaultHandler) on this 10x gfx900 rig (see the load log: "fitting params to device
# memory ..." right before the crash-restart loop that returns 503 "Loading model"). Disabling
# it makes the loader trust the explicit -ngl 99 -sm layer placement instead of probing.
# --no-mmap + --no-warmup: this box has 31 GB RAM; if the model is larger than free RAM, mmap
# makes the kernel thrash weight pages off the slow SATA SSD (port binds but never finishes
# loading). --no-mmap streams tensors straight to VRAM (all layers offloaded) so host RAM never
# holds the whole file; --no-warmup skips the full-weight warmup decode that re-touches everything.
# Safe to keep on regardless of model size; drop --no-mmap only if the model comfortably fits RAM.
#
# --parallel: llama.cpp splits -c across N slots, so N=4 gives each request only CTX/4 tokens.
#   MoE (few active params/token): default 4 is fine for a multi-client backend.
#   DENSE model or single-agent backend: set LLAMA_PARALLEL=1 so one request gets the FULL ctx.
NP="${LLAMA_PARALLEL:-4}"
echo "Starting <gemma-4-12B-coder-fable5-composer2.5-v1-uncensored-heretic-Q8_0> llama-server [$BACKEND] on $HOST:$PORT (ctx=$CTX, parallel=$NP, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), 9x Vega layer-split, no-mmap" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' -ngl 99 -sm layer -fit off -fa on --no-mmap --no-warmup -c $CTX --parallel $NP --host $HOST --port $PORT $AUTH $CACHE"
