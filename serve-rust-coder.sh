#!/usr/bin/env bash
# Launch llama-server for Gemma-4-Rust-Coder (MassivDash/Gemma-4-Rust-Coder) — a Gemma-4
# E-series (~5B) DENSE, multimodal, REASONING instruction model fine-tuned for Rust.
# Usage: serve-rust-coder.sh [hip|vulkan]   (default: hip — best for interactive chat)
# Env overrides: LLAMA_HOST (default 127.0.0.1), LLAMA_PORT (8093), LLAMA_CTX (16384),
#                LLAMA_ALIAS (rust-coder), LLAMA_PARALLEL (1), LLAMA_MMPROJ (auto)
#
# Source: https://huggingface.co/MassivDash/Gemma-4-Rust-Coder  (Gemma4ForConditionalGeneration,
# 131072 native ctx, Q8_0 GGUF ~5 GB + a BF16 vision projector).
#
# REASONING MODEL: despite the HF card calling it "standard instruction-tuned", the Gemma-4
# jinja template emits a thinking phase -> the chain-of-thought lands in `reasoning_content`
# and the final answer in `content` (same shape as Ornith). Give it generous max_tokens
# (>=1024) or `content` comes back empty with finish_reason=length while it's still thinking.
#
# WHY THIS SCRIPT DIVERGES FROM THE SIBLINGS (serve-qwen/ornith/deckard.sh):
# Those serve 25-68 GB MoE models, so they need --no-mmap (stream straight to VRAM, no page
# thrashing off the SSD). This model is ~5 GB at Q8_0 and mmap is fine since 5 GB << 31 GB RAM,
# so --no-mmap/--no-warmup are dropped. It also fits the leftover VRAM headroom on the good
# cards, so it COEXISTS with a big MoE backend instead of conflicting with it (verified running
# alongside Ornith: ~0.5 GB/card layer-split into the ~3 GB free on each of the 9 good cards).
#
# CARD CHOICE: we tried pinning to the otherwise-idle "bad" card (idx 6 / node-7) — it faults
# (VMFaultHandler) on load even for this tiny model, so it is unusable for compute, period.
# Default is therefore an -sm layer split across the 9 GOOD cards (idx 6 excluded), which works
# whether or not a big MoE backend is up. We keep `-fit off` (the auto memory-fitting probe
# faults on this gfx900 rig). Override with HIP_DEVICES (e.g. =0 for a single card if its VRAM
# is free — optimal for this small model when the rig is otherwise idle). See bad-vega-card note.
#
# Runs the server inside the 'render' group via sg, so it can reach /dev/kfd and the DRI
# render nodes even when the caller (e.g. the systemd --user manager) lacks that group.
set -euo pipefail

BACKEND="${1:-hip}"
ROOT="/home/botuser/Projects/llama.cpp"
MODEL="/home/botuser/Projects/models/gemma-4-rust-coder-Q8_0.gguf"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8093}"   # 8089 Qwen/GLM, 8090 Deckard, 8091 GLM-air/gemma-heretic, 8092 Ornith
CTX="${LLAMA_CTX:-16384}"    # model supports 131072; raise with LLAMA_CACHE_TYPE=q8_0 KV for big windows
# Served model id exposed via /v1/models. Cline/OpenAI clients use this as the "Model ID".
ALIAS="${LLAMA_ALIAS:-rust-coder}"
# API-key auth: if this file exists, require a Bearer token. Passed via --api-key-file
# (NOT --api-key) so the secret never appears in the process list on this shared box.
KEYFILE="${LLAMA_API_KEY_FILE:-$ROOT/.qwen-api-key}"

# Vision projector: enables image input (architecture diagrams / UI mockups -> Rust). Auto-
# detected if present; set LLAMA_MMPROJ= (empty) to force text-only, or to another path to override.
MMPROJ_DEFAULT="/home/botuser/Projects/models/gemma-4-rust-coder-mmproj-BF16.gguf"
MMPROJ="${LLAMA_MMPROJ-$MMPROJ_DEFAULT}"

# Default: split across the 9 GOOD cards; idx 6 / node-7 (the "bad" card) is excluded because
# it faults on load. This fits in the leftover headroom even when a big MoE backend is loaded.
# Override with HIP_DEVICES=0 (single card) when the rig is idle — optimal for this small model.
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

MM=""
if [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ]; then
  MM="--mmproj '$MMPROJ'"
  echo "Vision: ENABLED (mmproj $MMPROJ)" >&2
else
  echo "Vision: DISABLED (no mmproj)" >&2
fi

# Optional KV-cache quantization (needs FA, which is on). LLAMA_CACHE_TYPE=q8_0 ~halves KV for
# big context windows; near-lossless.
CACHE=""
if [ -n "${LLAMA_CACHE_TYPE:-}" ]; then
  CACHE="--cache-type-k $LLAMA_CACHE_TYPE --cache-type-v $LLAMA_CACHE_TYPE"
fi

# -fit off: the auto "fitting params to device memory" probe faults (VMFaultHandler) on this
# gfx900 rig; disabling it makes the loader trust the explicit -ngl 99 placement.
# --jinja: Gemma-4 ships a Jinja chat template (tool calls + thinking + multimodal) in the GGUF
# metadata; without --jinja llama.cpp falls back to a built-in template and formatting breaks.
# DENSE model -> default --parallel 1 so a single request gets the FULL ctx (set LLAMA_PARALLEL>1
# only if you want concurrent clients sharing CTX/N each).
NP="${LLAMA_PARALLEL:-1}"
echo "Starting Gemma-4-Rust-Coder llama-server [$BACKEND] on $HOST:$PORT (ctx=$CTX, parallel=$NP, FA on${LLAMA_CACHE_TYPE:+, kv=$LLAMA_CACHE_TYPE}), HIP_VISIBLE_DEVICES=$HIP_DEVS" >&2
exec sg render -c "exec env $PREFIX '$BIN' -m '$MODEL' --alias '$ALIAS' --jinja -ngl 99 -sm layer -fit off -fa on -c $CTX --parallel $NP --host $HOST --port $PORT $AUTH $MM $CACHE"
