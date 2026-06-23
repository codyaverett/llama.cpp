#!/usr/bin/env bash
# llama.cpp HIP/ROCm build for AMD Vega (gfx900) — all 5 Vega cards visible as ROCm0..4.
# Library paths are baked in via RPATH (Ubuntu ROCm 5.7/6.0, not the old /opt/rocm-5.3.0).
# Requires the running user to be in the 'render' group (for /dev/kfd).
#
# Usage:
#   ./run-hip.sh -m model.gguf -p "Hello" -ngl 99          # llama-cli (default)
#   LLAMA_BIN=llama-bench ./run-hip.sh -m model.gguf -ngl 99
#   LLAMA_BIN=llama-server ./run-hip.sh -m model.gguf -ngl 99 --host 0.0.0.0
#   HIP_VISIBLE_DEVICES=0 ./run-hip.sh ...                  # pin to a single card
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/build-hip/bin/${LLAMA_BIN:-llama-cli}" "$@"
