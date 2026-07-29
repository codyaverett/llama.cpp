# llama.cpp on AMD Vega 56/64 (gfx900) — build notes

Built on Ubuntu 24.04, **10× Vega 10 cards** (mix of Vega 64/56), all `gfx900`,
~80 GiB VRAM total (8 GiB HBM2 each), every card on PCIe 3.0 x16 via the board's
PLX switches. Source: https://github.com/ggml-org/llama.cpp @ `4b48a53`.

> **History:** the rig started as 5 cards (~40 GiB). The other 5 were physically
> installed but on a second PSU that only powers on during a **cold boot** — a warm
> reboot left them invisible (no PCIe link training; `rocm-smi` stuck at 5). A full
> power-cycle brought all 10 up. If a card count looks wrong after maintenance, cold-boot.

> **Single-stream speed does NOT scale with card count.** `-sm layer` is a sequential
> pipeline: each token still traverses every layer in order, so adding cards spreads the
> weights thinner but leaves per-chat tokens/sec ~flat (a touch lower with more pipeline
> hops). More cards buy **capacity** (bigger models / longer context) and **concurrent
> throughput** (more `--parallel` slots), not faster single replies.

## Backends built
- **Vulkan** (RADV): `build-vulkan/bin/` — recommended daily driver (faster prompt processing here, simplest, no ROCm version fragility).
- **HIP/ROCm** (gfx900): `build-hip/bin/` — uses Ubuntu's ROCm (HIP 5.7, rocBLAS/hipBLAS 5.5.1, which still ship gfx900 kernels).

Single-card benchmark, llama 7B Q4_0:

| Backend | pp64 (prompt) | tg32 (gen) |
|---------|--------------:|-----------:|
| Vulkan  | ~415 t/s      | ~52 t/s    |
| HIP     | ~201 t/s      | ~53 t/s    |

Multi-GPU benchmark, Qwen3.6-35B-A3B Q5_K_M (24.63 GiB, MoE 35B total / 3B active),
split across all 5 Vega cards (`-ngl 99 -sm layer`, 5 reps, llama-bench) — *these are
the original 5-card numbers; single-stream gen is ~the same on 10 cards (pipeline, see above)*:

| Backend | pp512 (prompt) | tg128 (gen) |
|---------|---------------:|------------:|
| Vulkan  | ~519 t/s       | ~15.3 t/s   |
| HIP     | ~143 t/s       | ~28.0 t/s   |

Takeaway: the two backends are mirror images for this MoE model. **Vulkan** wins prompt
processing (~3.6x) — best for long prompts / RAG / batch. **HIP** wins token generation
(~1.8x) — best for interactive chat. Rule of thumb: long prompt + short answer -> Vulkan;
short prompt + long answer -> HIP. The systemd service (below) defaults to HIP for chat.

Model file: `/home/botuser/Projects/models/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` (arch `qwen35moe`).

## systemd user service (Qwen server)
- Launcher: `serve-qwen.sh [hip|vulkan]` — wraps `llama-server` in `sg render` so it gets
  `/dev/kfd` access even though the lingering user manager lacks the `render` group.
- Units: `~/.config/systemd/user/llama-qwen.service` (HIP, enabled) and
  `llama-qwen-vulkan.service` (Vulkan, on demand). Only run ONE at a time (VRAM).
- Manage: `systemctl --user start|stop|status llama-qwen`, logs via `journalctl --user -u llama-qwen -f`.
- Endpoint: OpenAI-compatible API at `http://127.0.0.1:8089` (8080 was already taken on this box;
  override with `LLAMA_HOST`/`LLAMA_PORT`/`LLAMA_CTX`, or the `Environment=LLAMA_PORT=` line in the unit).
- Qwen3.6 is a **reasoning model**: by default it emits chain-of-thought into the API's
  `reasoning_content` field and the final answer into `content`. For direct answers, add
  `"chat_template_kwargs":{"enable_thinking":false}` to the request (or give it a large
  `max_tokens` so it can finish thinking). Verified working on port 8089.

## Context window sizing (this model on 10x Vega = ~80 GiB VRAM)
> Sizing table below was measured on the original 5-card / ~40 GiB rig and is still the
> conservative floor. With 10 cards (~80 GiB) there is far more headroom — the 24.63 GiB
> weights leave ~50+ GiB for KV+compute, so 128k–256k context is comfortable, and the
> 80 GiB total is what now makes 70B-class / GLM-4.5-Air models fit (see model notes).
KV cache is cheap for this model: only 2 KV heads (heavy GQA), 40 layers, head_dim 256 ->
**80 KiB/token at f16** (~0.08 GiB per 1k tokens, total across all 5 cards). Weights take
24.63 GiB; the rest is KV + compute. Always run with **Flash Attention (`-fa on`)** so the
attention compute buffer stays flat as context grows. Measured (HIP, `-sm layer`, FA on):

| Context | KV (f16) | Tightest card free | Total free | Verdict |
|--------:|---------:|-------------------:|-----------:|---------|
| 16k     | ~1.3 GiB | ~1.5 GiB           | ~9.4 GiB   | trivial |
| **64k** | ~5 GiB   | **~1.7 GiB**       | ~10.8 GiB  | **recommended default** |
| 128k    | ~10 GiB  | ~1.16 GiB          | ~8.2 GiB   | fits; OK for dedicated use |
| 256k    | ~20 GiB  | n/a (f16 too big)  | -          | needs KV quant (below) |

- **Recommended: 65536 (64k)** — generous for real workloads and leaves ~10+ GiB free for
  the other GPU users on this shared box. Set via `Environment=LLAMA_CTX=` in the unit (now 64k).
- **128k**: set `LLAMA_CTX=131072` — validated to fit, but GPU0 gets tight (~1.16 GiB free);
  best when nothing else is using the GPUs. **(Currently active.)**
- **Full 256k**: set `LLAMA_CTX=262144` **and** `Environment=LLAMA_CACHE_TYPE=q8_0` (halves KV,
  near-lossless; FA makes this work). Tight — for occasional long-doc use, not 24/7.
- Server default is `--parallel 4` with a unified KV pool, so `LLAMA_CTX` is the per-conversation
  max (slots share the pool). For guaranteed full context to a single conversation, also pass
  `--parallel 1` (add to the launcher's exec line).

## GLM-4.5-Air (106B-A12B) — attempted, does NOT run usefully here (2026-06-22)
Tried `unsloth/GLM-4.5-Air-GGUF` UD-Q4_K_XL (~68 GiB, arch `glm4moe`) across all 10 cards via
`serve-glm.sh` + `llama-glm.service` (both kept on disk, service left **disabled**). Outcome:
- **Weights DO load** into VRAM (~63 GiB across 10 cards; tightest card GPU8 ~7.1/7.98 GiB —
  only ~0.9 GiB free, so context would be tiny anyway).
- **But the server never becomes healthy.** After upload it gets stuck in a single-threaded,
  CPU-bound phase (graph alloc / warmup): one core pinned at 100%, **zero disk I/O**, no
  progress for 20+ min. A warmup pass should take seconds. Effectively hung.
- Load is also pathologically slow first: ~6.5 min of mmap reading because **31 GiB system RAM
  << 68 GiB model** (RES capped ~23 GiB, VIRT ~167 GiB — heavy page-fault churn).
- **Root causes:** (1) a CPU-bound post-load stall (likely a llama.cpp pathology with
  glm4moe + FA + 10-way layer split on this ROCm 5.7 build), compounded by (2) far too little
  system RAM for a 68 GiB model. **80 GiB of VRAM is not enough on its own — the load path
  goes through system RAM, and 31 GiB is the bottleneck.**
- **Not retried to success.** Worth trying later: `--no-warmup` (if the stall is the warmup
  pass), a smaller quant (IQ4_XS ~60 GiB), and/or a **system RAM upgrade (64–128 GiB)**.
- **Verdict: stay on Qwen3.6-35B-A3B** — it loads in ~2 min, serves fast (~28 t/s), and the
  MoE-3B-active design is a far better fit for these 8 GiB cards than a 68 GiB dense-ish MoE.

## API-key auth + network exposure
- Bound to `0.0.0.0:8089` (units set `Environment=LLAMA_HOST=0.0.0.0`). Reachable on the LAN
  at `http://192.168.86.50:8089` and over Tailscale at `http://100.88.229.68:8089`.
- Auth required: key file `~/Projects/llama.cpp/.qwen-api-key` (chmod 600), passed via
  `--api-key-file` so the secret never shows up in `ps`. Add more keys = one per line, then
  restart the service. Requests need `Authorization: Bearer <key>`; no/wrong key -> HTTP 401.
- Example:
  ```
  KEY=$(cat ~/Projects/llama.cpp/.qwen-api-key)
  curl http://192.168.86.50:8089/v1/chat/completions -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}}'
  ```
- Streaming: supported per-request (no server flag) — add `"stream": true` to get SSE token
  chunks (`delta.content`), and `"stream_options":{"include_usage":true}` for token counts in
  the final chunk. Continuous batching is on by default so multiple streams run concurrently.
- Security notes: the key is the only thing protecting this endpoint, and traffic is plain HTTP
  (no TLS) — prefer the Tailscale address over the LAN one. Host firewall (ufw) was not
  verifiable without sudo; confirm whether port 8089 is reachable from where you need it.

## Serving models: use llama-fleet (2026-07-28)
**`fleet/` is now the way to serve any model here** — see [fleet/README.md](fleet/README.md) for
the full model catalog, profile knobs and recovery steps. One generic launcher plus a small
profile per model replaces the seven bespoke `serve-*.sh` scripts (which still work, untouched,
as a fallback). Swap models with `fleet/fleet.sh use <profile> && fleet/fleet.sh restart`; the
front door stays on **:8092** so client configs never change.

Two findings from building it that apply to *anything* run on this rig:

1. **A second bad card: GPU[2] / HSA node-3.** Same defect as GPU[6] below — enumerates fine,
   reports full free VRAM, loads a 5.4 GB model to completion, then faults on the **first
   inference** (`Memory access fault ... Page not present`). Reproduced across three consecutive
   restarts. It had previously been written off as a transient "wedged at 100% busy" card; on
   2026-07-28 it was idle at 0% busy and still faulted. **Usable cards are 0,1,3,4,5,7,8,9.**
2. **Concurrency ceiling is CPU cores, not GPUs.** This i3-9100 has 4 cores and each
   `llama-server` busy-waits on GPU sync, so a serving process costs ~1 full core. Four
   concurrent shards scale flat (7.58s → 7.88s from 1 to 4 concurrent — 4x throughput for +3%
   latency). **Eight collapse**, and the collapse poisons ROCm driver state so that every
   `llama-server` started afterwards ends in unkillable **D state** until a reboot. Do not run
   more concurrent server processes than there are CPU cores; `fleet.sh` now enforces this.

## Running
Use the wrappers (`./run-vulkan.sh`, `./run-hip.sh`) or the binaries directly.
The running user must be in the **`render`** group (for `/dev/kfd` and the DRI render nodes).
`botuser` was added to `render`/`video`; a fresh login session picks this up, or use `sg render -c "..."`.
- HIP needs no env vars — correct ROCm 5.7/6.0 libs are baked in via RPATH.
- Vulkan: device 0 is the Intel iGPU and the last device is a software `llvmpipe` rasterizer;
  the wrapper sets `GGML_VK_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9,10` to use only the 10 Vega cards.

## Environment gotchas (already handled)
1. **GPU group access** — user needed to be in `render`/`video`.
2. **ROCm library conflict** — old ROCm 5.3 `libhsa-runtime`/`libamd_comgr` under `/opt/rocm-5.3.0`
   are forced ahead by `/etc/ld.so.conf.d`, breaking the Ubuntu HIP 5.7 runtime. Fixed per-process
   by baking `-rpath /usr/lib/x86_64-linux-gnu` (with `--disable-new-dtags` so it is transitive)
   into the HIP binaries — no system-wide change, so the other users' 5.3 OpenCL stack is untouched.

## Source/build patches applied (re-apply after `git pull`)
Current llama.cpp requires ROCm ≥ 6.1, but ROCm 6.x dropped gfx900 (Vega) support, so we build
against HIP 5.7 with these small back-compat fixes:
1. `ggml/src/ggml-hip/CMakeLists.txt` — relaxed the `VERSION_LESS 6.1` guard to `5.5`.
2. `ggml/src/ggml-cuda/vendors/hip.h` — replaced `#define cudaStreamWaitEvent hipStreamWaitEvent`
   with an inline wrapper defaulting `flags = 0` (HIP 5.7 lacks the 2-arg form).
3. `ggml/src/ggml-cuda/solve_tri.cu` — `const_cast<float* const*>(A_ptrs_dev)` (hipBLAS 5.5
   `hipblasStrsmBatched` wants a non-const array pointer).
4. Link flag `-Wl,--allow-multiple-definition` — works around an ODR bug in ROCm 5.7's
   `amd_hip_bf16.h` (`__low2float` emitted non-inline in every TU).

## Exact configure commands
Vulkan (needs `glslc`; SPIRV-Headers installed locally to `../deps`):
```
cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DCMAKE_PREFIX_PATH=/home/botuser/Projects/deps \
  -DCMAKE_CXX_FLAGS="-I/home/botuser/Projects/deps/include"
cmake --build build-vulkan -j4
```
HIP (gfx900):
```
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu cmake -B build-hip -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx900 -DGPU_TARGETS=gfx900 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DCMAKE_HIP_COMPILER=/usr/bin/clang++-17 \
  -DCMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu/cmake;/home/botuser/Projects/deps" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,/usr/lib/x86_64-linux-gnu -Wl,--allow-multiple-definition" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,/usr/lib/x86_64-linux-gnu -Wl,--allow-multiple-definition"
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu cmake --build build-hip -j4
```

## BAD CARD: GPU[6] / HSA node-7 / PCIe 0000:16:00.0 (exclude it)
> **See also the serving section at the top: GPU[2] / HSA node-3 is a SECOND bad card with the
> same signature, confirmed 2026-07-28. Usable cards are 0,1,3,4,5,7,8,9 — not 9 cards but 8.**

One Vega (rocm-smi index **6**, PCIe bus **0x16**, KFD/HSA **node-7**, location_id 5632) reliably
triggers a `Memory access fault by GPU node-7 ... Page not present` → `VMFaultHandler` assertion
the moment a large (~3 GB) per-card weight buffer lands on it. Reproducible across `-fit on/off`
and multiple model loads; always node-7. The smaller MoE Qwen (Q5, ~2.4 GB/card) squeaks by, but
any larger per-card allocation faults — and a fault here can wedge the whole ROCm runtime until a
cold power-cycle. **Exclude it:** `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,7,8,9` (HIP idx 6 == this card).
Map check: `rocm-smi --showbus` (idx→bus) and `/sys/class/kfd/kfd/topology/nodes/*/properties`
(location_id 5632 = bus 0x16). Suspect the card or its PCIe riser/link; swap to confirm.

## Running Qwen3.6-40B Deckard-Opus (DENSE 40B) — serve-deckard.sh, port 8090
DavidAU/Qwen3.6-40B-...-NEO-CODE-...-MAX-GGUF, Q6_K (32 GB), 96 layers, 256K ctx, vision+thinking.
DENSE (not MoE): every token reads all 40 B params → bandwidth-bound. Measured **~6.8 t/s gen**,
~12 t/s prompt on 9 Vega (vs ~25 t/s for the MoE Qwen). Quality > speed.
Three walls hit and the flags that fixed each:
- **`-fit off`** — the new "fitting params to device memory" auto-probe faults on this 10-GPU rig.
- **`HIP_VISIBLE_DEVICES` excluding GPU[6]** — see bad-card note above (was the real fault source).
- **`--no-mmap --no-warmup`** — 31 GB RAM < 32 GB model: with mmap the kernel thrashes weight pages
  off the SATA SSD (171 MB/s, port binds but never serves). `--no-mmap` streams tensors straight
  to VRAM (server RES only ~3.6 GB); `--no-warmup` skips the full-weight warmup re-touch.
Footprint when up: ~4.3 GB on each of 9 cards (GPU9 ~5 GB holds embed/output), GPU6 idle.
Headroom to raise `-c` well above 8192. The 32 GB Q6_K is at the practical ceiling for 31 GB RAM —
a RAM upgrade (≥64 GB) would remove the --no-mmap constraint and allow bigger/Q8 models.
