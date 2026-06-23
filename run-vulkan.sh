#!/usr/bin/env bash
# llama.cpp Vulkan build (RADV) for AMD Vega. The Intel iGPU is Vulkan device 0 and a
# software llvmpipe rasterizer is the last device, so by default we restrict to the 10
# Vega cards (Vulkan1..10). Override GGML_VK_VISIBLE_DEVICES to change selection.
# Requires the running user to be in the 'render' group.
#
# Usage:
#   ./run-vulkan.sh -m model.gguf -p "Hello" -ngl 99        # llama-cli (default)
#   LLAMA_BIN=llama-bench ./run-vulkan.sh -m model.gguf -ngl 99
#   GGML_VK_VISIBLE_DEVICES=1 ./run-vulkan.sh ...           # pin to a single Vega
export GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-1,2,3,4,5,6,7,8,9,10}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/build-vulkan/bin/${LLAMA_BIN:-llama-cli}" "$@"
