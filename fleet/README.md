# llama-fleet

Run one model as N independent `llama-server` shards across the Vega rig, behind a single
stable front door on **:8092**.

```sh
fleet/fleet.sh list                 # every profile, * marks the active one
fleet/fleet.sh plan ornith-9b       # preview shard->card map + VRAM budget (touches nothing)
fleet/fleet.sh use ornith-9b        # make it active (re-resolves plan, rewires systemd)
fleet/fleet.sh restart && fleet/fleet.sh wait
fleet/fleet.sh status               # per-shard systemd state + /health
fleet/bench.sh                      # walk concurrency up, stop at first failure
```

Swapping models is `fleet use <profile> && fleet restart`. The pool port never changes, so
client configs (the rustyclaw "rusty" agent, Cline, …) never need editing.

## Why shards

`-sm layer` across N cards is a **pipeline**: card 0 runs layers 0..k, then card 1, and so on,
so for any single token only one card is doing math. A model that fits on one card should
instead run as N independent servers that all compute at once. A model too big for one card
falls back to exactly the old behaviour — one shard, layer-split, pool as pass-through.

## The ceiling is CPU cores, not GPUs

Measured 2026-07-28 on this 4-core i3-9100. Each shard busy-waits on GPU synchronisation, so an
actively-serving shard costs roughly **one full core**.

| Concurrency | Wall time | Verdict |
|---|---|---|
| 1 | 7.58s | baseline |
| 2 | 7.61s | flat |
| 4 | 7.88s | flat — **4x throughput for +3% latency** |
| 8 | timeout | collapses, and poisons ROCm driver state |

At 8 shards, requests time out and shards afterwards sit spinning at 80–91% CPU each (266% of
400%) with nothing in flight. Worse, the collapse leaves the driver in a state where *every*
subsequently started `llama-server` — including a fresh 4-shard load — ends in unkillable **D
state**, recoverable only by reboot. Two reboots were spent learning this.

`fleet.sh` therefore enforces a hard cap of `shards <= nproc` in `resolve_plan()`, so a profile
typo cannot cost a reboot. With 4 shards on 4 cards, the remaining good cards are better spent
on a **second model profile** than on more shards of the same one.

## Cards

Source of truth is `cards.conf`. **Usable: 0, 1, 3, 4, 5, 7, 8, 9.**

Two cards are permanently denylisted, both with the same defect — they enumerate fine, report
full free VRAM, load a multi-GB model to completion, then fault on the **first inference** with
`Memory access fault ... Page not present or supervisor privilege` → `VMFaultHandler`:

- **card 6** (HSA node-7, PCIe `0000:16:00.0`) — long known.
- **card 2** (HSA node-3) — confirmed 2026-07-28. Previously mistaken for a transient "wedged at
  100% busy" card; it was idle at 0% and still faulted. *A card looking healthy is not evidence
  that it is.*

Neither is caught by the fast boot probe, because both pass enumeration. `fleet probe --deep`
load-tests each card with a real decode and does catch them; the fast probe still earns its keep
against the other failure mode (card wedged at 100% busy, model load hangs forever).

## Model catalog

`CARDS/SH` is cards per shard; `SHARDS` is what that resolves to on 8 good cards *after* the
`<= nproc` cap. Sizes are the on-disk GGUF.

| Profile | Model | Size | Type | CARDS/SH | SHARDS | CTX | Status |
|---|---|---|---|---|---|---|---|
| `ornith-9b` | Ornith-1.0-9B Q4_K_M | 5436 MiB | dense hybrid `qwen35` | 1 | 4 | 32768 | **verified** |
| `rust-coder` | Gemma-4-Rust-Coder Q8_0 (+941 MiB mmproj) | 4737 MiB | ~5B dense + vision | 1 | 4 | 16384 | untested as fleet |
| `deepseek-coder-lite` | DeepSeek-Coder-V2-Lite Q5_K_M | 11302 MiB | 16B/2.4B MoE | 2 | 4 | 16384 | untested as fleet |
| `gemma-agentic-fable5` | Gemma-4 12B agentic tau2 Q8_0 | 12082 MiB | 12B dense, tool-calling | 2 | 4 | 16384 | untested as fleet |
| `gemma-coder-heretic` | Gemma-4 12B uncensored Q8_0 | 12082 MiB | 12B dense | 2 | 4 | 32768 | untested as fleet |
| `ornith-35b` | Ornith-1.0-35B Q5_K_M | 23583 MiB | `qwen35moe`, ~3B active | 8 | 1 | 131072 | as before |
| `ornith-aeon-35b` | Ornith-1.0-35B AEON uncensored Q4_K_M | 20186 MiB | `qwen35moe`, ~3B active | 8 | 1 | 131072 | as before |
| `qwen-35b` | Qwen3.6-35B-A3B UD-Q5_K_M | 25230 MiB | MoE, ~3B active | 8 | 1 | 16384 | as before |
| `deckard-40b` | Qwen3.6-40B Deck-Opus NEO Q6_K | 30890 MiB | **dense** 40B | 8 | 1 | 32768 | as before |

**Status means:** *verified* = benchmarked as a fleet on this rig. *as before* = single-shard,
transcribed flag-for-flag from the original `serve-*.sh`, so it reproduces the known-good
configuration. *untested as fleet* = the profile is written and sized, but multi-shard operation
has not been benchmarked — run `fleet/bench.sh` after switching to one.

Ornith 9B is worth a note on sizing: 32 layers but only 8 are full-attention
(`full_attention_interval: 4`); the other 24 are linear/recurrent with a fixed per-slot state
independent of context length. Full-attention KV is 4 kv-heads × 256 head-dim × 8 layers ≈
32 KB/token at f16, ~17 KB at q8_0 — so 32K context costs only ~557 MiB. Actual measured
footprint is **6.11 GB of 8176 MiB** per card, leaving ~1.9 GB spare.

## Profile knobs

```sh
MODEL="$MODELS_DIR/....gguf"
MMPROJ="..."            # optional, vision models
ALIAS=ornith            # served-model id on /v1/models
CARDS_PER_SHARD=1       # 1 => one shard per card; 8 => one shard across the rig
CTX=32768               # per shard, split across PARALLEL slots
PARALLEL=2              # server slots per shard
CACHE_TYPE=q8_0         # empty => f16 KV
MMAP=off                # see below
ROUTING=ip_hash         # ip_hash | least_conn
THREADS=1               # see below
THREADS_HTTP=2
MAX_SHARDS=4            # capped again by nproc regardless
EXTRA="--jinja"
```

**`MMAP`** is genuinely per-model, not a constant. `off` streams tensors straight to VRAM and is
required whenever the model approaches free host RAM (31 GB box) — with mmap the kernel thrashes
weight pages off the SATA SSD and the port binds but never serves. `on` would let shards share
one page-cache copy, which sounds strictly better for a fleet and is how `ornith-9b` was first
written — but an 8-shard `MMAP=on` run wedged every process into D state, the signature of
blocking on page faults inside the GPU driver. **Every profile here uses `off`.** Do not flip it
without testing at 2 shards first.

**`THREADS`** — `llama-server` defaults to 4 generation + 6 HTTP threads *per process*. Four
shards at those defaults is ~40 threads on 4 cores. With `-ngl 99` all compute is on the GPU, so
`THREADS=1` is right for any multi-shard profile. Leave empty for single-shard profiles.

**`ROUTING`** — `ip_hash` pins each client to one shard so llama.cpp's prompt cache stays warm
across an agent's turns, which for agentic coding is worth far more than perfect balance
(otherwise every turn resends a growing conversation and reprocesses it from scratch).
`least_conn` balances better; use it for batch work. Caveat: stickiness keys on client IP, so
several agents on *localhost* all hash to one shard — route on a header instead if that bites.

## systemd

| Unit | Role |
|---|---|
| `llama-fleet.target` | the enabled-at-boot entry point; `Conflicts=` the legacy per-model units |
| `llama-fleet-probe.service` | oneshot; probes cards, rewrites the plan, ordered `Before=` all shards |
| `llama-shard@N.service` | template; `%i` is the shard index, staggered `%i * 8`s to avoid a thundering herd |
| `llama-pool.service` | rootless nginx on :8092 |

`fleet use` rewrites `llama-fleet.target.wants/` so the shard count survives a reboot with no
unit editing. Shards never compute their own placement — they read the resolved
`~/.local/state/llama-fleet/plan.env`, so probe and launcher cannot disagree.

The pool is a **second, rootless nginx**, not the system one: the system nginx runs as root and
would need sudo to reload. This one has a private prefix under `~/.local/state/llama-fleet/nginx`
and a port >1024, so `fleet reload` needs no privileges.

## Fallback

The original `serve-*.sh` scripts are untouched and still work. To drop back to the old
single-model setup:

```sh
systemctl --user start llama-ornith-aeon   # fleet stops itself via Conflicts=
systemctl --user disable llama-fleet.target
```

## Recovering a wedged rig

Symptom: `llama-server` processes in **D state**, unkillable by SIGKILL, VRAM still pinned, load
average high while the CPU is mostly idle.

```sh
ps -eo stat,comm --no-headers | awk '$1 ~ /^D/ && $2 ~ /llama/' | wc -l   # 0 == healthy
```

There is no userspace recovery — `pkill -9` does not work, because the processes are blocked in
the kernel. Reboot.
