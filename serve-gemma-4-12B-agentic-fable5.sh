#!/usr/bin/env bash
# Launch llama-server for gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2 — a Gemma-4 12B
# DENSE model finetuned for agentic / tool-calling work (tau2-bench tuned).
# Usage: serve-gemma-4-12B-agentic-fable5.sh [hip|vulkan]   (default: hip — best for interactive chat)
# Env overrides: LLAMA_HOST (default 127.0.0.1), LLAMA_PORT (8095), LLAMA_CTX (16384),
#                LLAMA_ALIAS (gemma-agentic-fable5), LLAMA_PARALLEL (1)
#
# Source: yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF (Q8_0, ~12.7 GB).
# Repo also ships Q3_K_M/Q4_K_M/Q6_K + an MTP build; Q8_0 picked for best fidelity (small model,
# fits easily). Sibling of serve-gemma-4-12B-coder-heretic.sh (the v1 uncensored composer).
#
# DENSE 12B (not MoE): every param is active per token, so default --parallel 1 gives one request
# the FULL context. It's ~12.7 GB at Q8_0, so it COEXISTS in the leftover VRAM headroom alongside
# a big MoE backend (like the rust-coder), or runs solo with room to spare on the 9-card split.
#
# Runs the server inside the 'render' group via sg, so it can reach /dev/kfd and the DRI
# render nodes even when the caller (e.g. the systemd --user manager) lacks that group.
set -euo pipefail

BACKEND="${1:-hip}"
ROOT="/home/botuser/Projects/llama.cpp"
MODEL="/home/botuser/Projects/models/gemma-4-12B-agentic-fable5-composer2.5-v2-tau2-Q8_0.gguf"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8095}"   # 8089 Qwen, 8090 Deckard, 8091 gemma-heretic, 8092 Ornith, 8093 rust-coder, 8094 deepseek-coder-lite
CTX="${LLAMA_CTX:-16384}"    # Gemma-4 supports 131072; raise with LLAMA_CACHE_TYPE=q8_0 KV for big windows
# Served model id exposed via /v1/models. Cline/OpenAI clients use this as the "Model ID".
ALIAS="${LLAMA_ALIAS:-gemma-agentic-fable5}"
# API-key auth: if this file exists, require a Bearer token. Passed via --api-key-file
# (NOT --api-key) so the secret never appears in the process list on this shared box.
KEYFILE="${LLAMA_API_KEY_FILE:-$ROOT/.qwen-api-key}"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found: $MODEL" >&2
  echo "       still downloading? check .gemma-v2-agentic-dl.log in the models dir." >&2
  exit 1
fi

# Default: split across the 9 GOOD cards; idx 6 / node-7 (the "bad" card) is excluded because it
# faults on load. This fits the leftover headroom even when a big MoE backend is loaded. Override
# with HIP_DEVICES=0 (single card) when the rig is idle — optimal for this small dense model.
HIP_DEVS="${HIP_DEVICES:-0,1,2,3,4,5,7,8,9}"

case "$BACKEND" in
  hip)
    BIN="$ROOT/build-hip/bin/llama-server"
    PREFIX="HIP_VISIBLE_DEVICES=$HIP_DEVS"
    ;;
  vulkan)
    BIN="$ROOT/build-vulkan/bin/llama-server"
    PREFIX="GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-1}"   # one card; skip Intel iGPU (Vulkan0)
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
# --jinja: Gemma-4 ships a Jinja chat template (tool calls + thinking) in the GGUF metadata;
# without --jinja llama.cpp falls back to a built-in template and tool-call/format handling breaks.
# This model is ~12.7 GB and the box has ~10 GB RAM free, so --no-mmap streams straight to VRAM
# (avoids paging weights off the slow SATA SSD); --no-warmup skips the full-weight warmup decode.
# DENSE model -> default --parallel 1 so a single agent request gets the FULL ctx (raise
# LLAMA_PARALLEL only for concurrent clients sharing CTX/N each).
NP="${LLAMA_PARALLEL:-1}"
echo "Starting gemma-4-12B-agentic-fable5 llama-server [$BACKEND] on $HOST:$PORT (alias=$ALIAS, ctx=$CTX, parallel=$NP, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), HIP_VISIBLE_DEVICES=$HIP_DEVS" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' --alias '$ALIAS' --jinja -ngl 99 -sm layer -fit off -fa on --no-mmap --no-warmup -c $CTX --parallel $NP --host $HOST --port $PORT $AUTH $CACHE"
