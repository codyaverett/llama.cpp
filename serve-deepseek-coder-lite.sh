#!/usr/bin/env bash
# Launch llama-server for DeepSeek-Coder-V2-Lite-Instruct — a 16B-total / 2.4B-active MoE
# code model (mid-2024), pinned to TWO Vega cards as a low-footprint coder / benchmark target
# to put head-to-head against the Qwen3.6-35B-A3B agent backend (serve-qwen.sh, port 8089).
# Usage: serve-deepseek-coder-lite.sh [hip|vulkan]   (default: hip — best for interactive chat)
# Env overrides: LLAMA_HOST (default 127.0.0.1), LLAMA_PORT (8094), LLAMA_CTX (16384),
#                LLAMA_ALIAS (deepseek-coder-lite), LLAMA_PARALLEL (1), HIP_DEVICES (0,1)
#
# Source GGUF: not present yet. Fetch e.g. a Q5_K_M (~11 GB, the comparable quant to Qwen's
# Q5_K_M) into $MODEL below, for example:
#   huggingface-cli download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF \
#     DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf \
#     --local-dir /home/botuser/Projects/models
# (Drop to Q4_K_M ~10 GB if you want extra VRAM margin; this model holds up well at Q4.)
#
# WHY 2 CARDS (and the coexistence caveat — READ THIS):
# At Q5_K_M the weights are ~11 GB, which fits comfortably on 2 idle Vegas (2x8=16 GB) with
# room for the KV cache. BUT serve-qwen.sh layer-splits across ALL 9 good cards and parks
# ~5 GB on each (leaving ~3 GB free/card), so there are NO fully-free good cards while Qwen
# is up. ~5.5 GB/card for a 2-card DeepSeek split will NOT fit into 3 GB of leftover headroom.
# Therefore, to run a CLEAN head-to-head benchmark, do ONE of:
#   (a) Stop Qwen first (sequential benchmark — recommended; avoids both models fighting for
#       the same cards and skewing latency numbers), then run this on the default 2 cards; or
#   (b) Restrict Qwen to fewer cards (e.g. HIP_DEVICES=2,3,4,5,7,8,9 serve-qwen.sh) to free
#       idx 0,1 for this script; or
#   (c) For genuine SIMULTANEOUS coexistence, thin-split this across all 9 good cards like the
#       rust-coder backend does:  HIP_DEVICES=0,1,2,3,4,5,7,8,9 serve-deepseek-coder-lite.sh
#       (~1.3 GB/card — fits the leftover headroom alongside Qwen, at the cost of 9-way PCIe
#       layer-split overhead, so use this for "does it coexist" not for clean latency numbers).
#
# CARD CHOICE: default is idx 0,1 (two GOOD cards). idx 6 / node-7 (the "bad" card) faults on
# load (VMFaultHandler) even for tiny models and is excluded from every default here. Override
# with HIP_DEVICES. See the bad-vega-card note. We keep `-fit off` because the auto memory-
# fitting probe faults on this gfx900 rig; the loader then trusts the explicit -ngl 99 split.
#
# Runs the server inside the 'render' group via sg, so it can reach /dev/kfd and the DRI
# render nodes even when the caller (e.g. the systemd --user manager) lacks that group.
set -euo pipefail

BACKEND="${1:-hip}"
ROOT="/home/botuser/Projects/llama.cpp"
MODEL="/home/botuser/Projects/models/DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8094}"   # 8089 Qwen, 8090 Deckard, 8091 GLM-air/gemma-heretic, 8092 Ornith, 8093 rust-coder
CTX="${LLAMA_CTX:-16384}"    # model supports 163840; raise with LLAMA_CACHE_TYPE=q8_0 KV for big windows
# Served model id exposed via /v1/models. Cline/OpenAI clients use this as the "Model ID".
ALIAS="${LLAMA_ALIAS:-deepseek-coder-lite}"
# API-key auth: if this file exists, require a Bearer token. Passed via --api-key-file
# (NOT --api-key) so the secret never appears in the process list on this shared box.
KEYFILE="${LLAMA_API_KEY_FILE:-$ROOT/.qwen-api-key}"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found: $MODEL" >&2
  echo "       fetch a Q5_K_M (or Q4_K_M) GGUF first — see the header comment for the command." >&2
  exit 1
fi

# Default: pin to TWO good cards (idx 0,1). Override with HIP_DEVICES — e.g. the 9-card
# thin-split (0,1,2,3,4,5,7,8,9) for simultaneous coexistence with Qwen. idx 6 is excluded.
HIP_DEVS="${HIP_DEVICES:-0,1}"

case "$BACKEND" in
  hip)
    BIN="$ROOT/build-hip/bin/llama-server"
    PREFIX="HIP_VISIBLE_DEVICES=$HIP_DEVS"
    ;;
  vulkan)
    BIN="$ROOT/build-vulkan/bin/llama-server"
    PREFIX="GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-1,2}"   # two cards; skip Intel iGPU (Vulkan0)
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

# Optional KV-cache quantization (needs FA, which is on). LLAMA_CACHE_TYPE=q8_0 ~halves KV for
# big context windows; near-lossless.
CACHE=""
if [ -n "${LLAMA_CACHE_TYPE:-}" ]; then
  CACHE="--cache-type-k $LLAMA_CACHE_TYPE --cache-type-v $LLAMA_CACHE_TYPE"
fi

# -fit off: the auto "fitting params to device memory" probe faults (VMFaultHandler) on this
# gfx900 rig; disabling it makes the loader trust the explicit -ngl 99 -sm layer placement.
# --no-mmap + --no-warmup: this box has only ~10 GB RAM free (20/31 GB used by the other
# backends) and the model is ~11 GB, so mmap would thrash weight pages off the slow SATA SSD.
# --no-mmap streams tensors straight to VRAM; --no-warmup skips the full-weight warmup decode.
# --parallel 1: default to a SINGLE slot so one benchmark request gets the FULL ctx and latency
# numbers are clean (raise LLAMA_PARALLEL for concurrent clients sharing CTX/N each).
NP="${LLAMA_PARALLEL:-1}"
echo "Starting DeepSeek-Coder-V2-Lite llama-server [$BACKEND] on $HOST:$PORT (ctx=$CTX, parallel=$NP, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), HIP_VISIBLE_DEVICES=$HIP_DEVS" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' --alias '$ALIAS' -ngl 99 -sm layer -fit off -fa on --no-mmap --no-warmup -c $CTX --parallel $NP --host $HOST --port $PORT $AUTH $CACHE"
