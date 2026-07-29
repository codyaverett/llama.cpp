#!/usr/bin/env bash
# llama-fleet — run one model as N independent llama-server shards across the Vega rig,
# behind a single stable front-door port.
#
#   fleet list              show profiles, mark the active one
#   fleet use <profile>     make <profile> active (re-resolves the plan, rewires systemd)
#   fleet plan              print the resolved shard -> card map + VRAM budget (touches nothing)
#   fleet probe [--deep]    health-check the cards, refresh good-cards, re-resolve the plan
#                           --deep additionally load-tests each card with the active model,
#                           which is the only way to catch a card that faults under compute
#   fleet up | down | restart
#   fleet wait [secs]       block until every shard answers /health
#   fleet status            active profile + per-shard health
#   fleet logs [shard]      journal for one shard, or all of them
#   fleet reload            regenerate + reload the pool config only (e.g. after ROUTING change)
#
# WHY SHARDS: `-sm layer` across N cards is a pipeline — card 0 runs layers 0..k, then card 1,
# and so on, so for any single token only ONE card is doing math. A model small enough to fit on
# one card should instead run as N independent servers, one per card, all computing at once.
# That is roughly a 10-15x aggregate throughput difference on this rig. A model too big to fit
# on one card falls back to exactly the old behaviour: one shard, layer-split, pool is a
# pass-through. Same commands either way.
set -euo pipefail

ROOT="/home/botuser/Projects/llama.cpp"
FLEET="$ROOT/fleet"
MODELS_DIR="/home/botuser/Projects/models"
STATE="${LLAMA_FLEET_STATE:-$HOME/.local/state/llama-fleet}"
UNITS="$HOME/.config/systemd/user"
CARD_MIB=8176   # usable VRAM per Vega, from llama-server --list-devices

# shellcheck disable=SC1091
. "$FLEET/cards.conf"

mkdir -p "$STATE" "$STATE/nginx/logs" "$STATE/nginx/tmp"

die() { echo "fleet: $*" >&2; exit 1; }

active_profile() { cat "$STATE/profile" 2>/dev/null || echo ""; }

# ── card discovery ────────────────────────────────────────────────────────────────────────────

# Cards eligible for probing: everything present, minus the permanent hardware denylist.
candidate_cards() {
  local c keep=""
  for c in $ALL_CARDS; do
    case " $DENY_CARDS " in *" $c "*) continue ;; esac
    keep="$keep $c"
  done
  echo "$keep" | xargs
}

# Healthy cards. Prefers the probe's result; falls back to "everything not denied" so the fleet
# still starts if the probe has never run (e.g. first install, or state dir wiped).
good_cards() {
  if [ -s "$STATE/good-cards" ]; then
    tr '\n' ' ' < "$STATE/good-cards" | xargs
  else
    candidate_cards
  fi
}

# This rig has TWO distinct card failure modes and they need two distinct tests:
#
#   wedged   — card pinned at 100% busy with no KFD processes. It never finishes enumerating,
#              so any model load hangs forever. Caught by the fast probe (a timeout).
#   faulting — card enumerates, reports full free VRAM, loads a multi-GB model successfully,
#              then dies with "Memory access fault ... Page not present or supervisor
#              privilege" on the first real decode. Cards 6 and 2 both do this. The fast probe
#              CANNOT see it; only running actual compute can. That is --deep.
#
# Boot runs the fast probe (seconds). --deep is for after a hardware change or when a card is
# suspected — it costs a model load per card, so it is deliberately not on the boot path.
cmd_probe() {
  local deep=0
  [ "${1:-}" = "--deep" ] && deep=1
  local bin="$ROOT/build-hip/bin/llama-server" c ok=""
  [ -x "$bin" ] || die "llama-server not built at $bin"

  local model=""
  if [ "$deep" = 1 ]; then
    local p; p=$(active_profile)
    [ -n "$p" ] || die "--deep needs an active profile (it load-tests with that model)"
    local MODEL=""
    # shellcheck disable=SC1090
    . "$FLEET/profiles/${p}.conf"
    model="$MODEL"
    [ -f "$model" ] || die "model not found for deep probe: $model"
    echo "deep-probing cards with $(basename "$model") (up to ${DEEP_PROBE_TIMEOUT}s each; denylisted: ${DENY_CARDS:-none})..."
  else
    echo "probing cards (timeout ${PROBE_TIMEOUT}s each; denylisted: ${DENY_CARDS:-none})..."
  fi

  for c in $(candidate_cards); do
    if [ "$deep" = 1 ]; then
      # llama-bench allocates the full weight buffers and runs real prompt+decode passes on the
      # card — the exact workload that triggers the fault. Exit status is the verdict.
      if timeout "$DEEP_PROBE_TIMEOUT" sg render -c \
           "HIP_VISIBLE_DEVICES=$c '$ROOT/build-hip/bin/llama-bench' -m '$model' \
            -ngl 99 -sm none -mg 0 -fa 1 -p 32 -n 8 -r 1 -o csv" >/dev/null 2>&1; then
        printf '  card %-2s ok (loaded + decoded)\n' "$c"
        ok="$ok $c"
      else
        printf '  card %-2s FAULTED under load — excluded\n' "$c"
      fi
    else
      if timeout "$PROBE_TIMEOUT" sg render -c \
           "HIP_VISIBLE_DEVICES=$c '$bin' --list-devices" 2>/dev/null | grep -q 'ROCm0:'; then
        printf '  card %-2s ok\n' "$c"
        ok="$ok $c"
      else
        printf '  card %-2s UNHEALTHY (no enumeration within %ss) — excluded\n' "$c" "$PROBE_TIMEOUT"
      fi
    fi
  done

  echo "$ok" | xargs -n1 > "$STATE/good-cards"
  echo "good cards: $(good_cards)"
  # Card count may have changed, so the plan has to be rebuilt.
  [ -n "$(active_profile)" ] && resolve_plan "$(active_profile)"
  return 0
}

# ── plan resolution ───────────────────────────────────────────────────────────────────────────

# Writes the authoritative shard -> card mapping. Everything downstream (serve-shard.sh, the
# nginx upstream, the systemd wants) reads from this one file.
#
# Second arg redirects the output elsewhere, which is how `fleet plan <profile>` inspects a
# profile that is NOT active without clobbering the running fleet's plan. Only a write to the
# real plan.env regenerates the pool config.
resolve_plan() {
  local profile="$1"
  local out="${2:-$STATE/plan.env}"
  local pf="$FLEET/profiles/${profile}.conf"
  [ -f "$pf" ] || die "no such profile: $profile"

  local CARDS_PER_SHARD=1 MAX_SHARDS="" MODEL="" ALIAS="" CTX=16384 PARALLEL=1 ROUTING=ip_hash
  # shellcheck disable=SC1090
  . "$pf"

  local cards; cards=$(good_cards)
  local ncards; ncards=$(echo "$cards" | wc -w)
  [ "$ncards" -gt 0 ] || die "no healthy cards available"

  # A model needing more cards than exist still runs — on everything there is, as one shard.
  if [ "$CARDS_PER_SHARD" -gt "$ncards" ]; then
    echo "fleet: warning: profile wants $CARDS_PER_SHARD cards/shard but only $ncards are healthy;" >&2
    echo "       running a single shard across all $ncards." >&2
    CARDS_PER_SHARD=$ncards
  fi

  local nshards=$(( ncards / CARDS_PER_SHARD ))
  [ "$nshards" -ge 1 ] || nshards=1
  if [ -n "$MAX_SHARDS" ] && [ "$nshards" -gt "$MAX_SHARDS" ]; then
    nshards=$MAX_SHARDS
  fi

  # HARD SAFETY CAP: never run more shards than the host has CPU cores.
  #
  # This is not a tuning preference, it is damage control. Each shard busy-waits on GPU
  # synchronisation, so a serving shard costs ~1 full core. Measured on this 4-core box:
  # 4 shards scale flat (7.58s -> 7.88s from 1 to 4 concurrent), while 8 shards collapse AND
  # poison the ROCm driver state — every llama-server, including ones started afterwards, ends
  # in unkillable D state and only a reboot recovers. A profile typo should not be able to cost
  # a reboot, so the cap is enforced here rather than trusted to each profile.
  local cores; cores=$(nproc)
  if [ "$nshards" -gt "$cores" ]; then
    echo "fleet: capping $nshards shards to $cores (one per CPU core; more oversubscribes the" >&2
    echo "       host and wedges ROCm — see MAX_SHARDS note in profiles/ornith-9b.conf)." >&2
    nshards=$cores
  fi

  {
    echo "# generated by fleet.sh — do not edit; run 'fleet use' or 'fleet probe' to regenerate"
    echo "PROFILE=$profile"
    echo "SHARD_COUNT=$nshards"
    echo "CARDS_PER_SHARD=$CARDS_PER_SHARD"
    echo "ROUTING=$ROUTING"
    local i n set
    set=$(echo "$cards" | tr ' ' '\n')
    for i in $(seq 0 $((nshards - 1))); do
      # Shard i takes the i-th consecutive block of cards.
      n=$(echo "$set" | sed -n "$((i * CARDS_PER_SHARD + 1)),$(((i + 1) * CARDS_PER_SHARD))p" | paste -sd, -)
      echo "SHARD_${i}_CARDS=$n"
      echo "SHARD_${i}_PORT=$((SHARD_PORT_BASE + i))"
    done
  } > "$out"

  [ "$out" = "$STATE/plan.env" ] && gen_pool_conf
  return 0
}

# ── the pool (rootless nginx) ─────────────────────────────────────────────────────────────────

# The system nginx runs as root and would need sudo to reload, so the fleet runs its OWN nginx
# as this user with a private prefix. Nothing here needs privileges: the port is >1024 and every
# path nginx wants to write lives under $STATE.
gen_pool_conf() {
  # shellcheck disable=SC1091
  . "$STATE/plan.env"
  local i upstream=""
  for i in $(seq 0 $((SHARD_COUNT - 1))); do
    # max_fails/fail_timeout matter on this rig: a card can fault mid-flight and take its shard
    # down (see cards.conf on GPUs 2 and 6). With passive checks disabled nginx would keep
    # hashing 1/N of all clients into a dead port indefinitely. Instead, eject a shard after 3
    # failures and re-try it 30s later, so one bad card degrades the fleet instead of breaking it.
    upstream="$upstream        server 127.0.0.1:$((SHARD_PORT_BASE + i)) max_fails=3 fail_timeout=30s;
"
  done

  # ip_hash keeps each client pinned to one shard so llama.cpp's prompt cache stays warm across
  # an agent's turns — worth far more than perfect balance for agentic coding, where every turn
  # resends a growing conversation. least_conn balances better but forces a full prompt
  # reprocess on every turn. `consistent` means adding/removing a shard only remaps 1/N of
  # clients rather than reshuffling all of them.
  local balance
  case "${ROUTING:-ip_hash}" in
    ip_hash)    balance="        hash \$remote_addr consistent;" ;;
    least_conn) balance="        least_conn;" ;;
    *) die "unknown ROUTING '$ROUTING' (use: ip_hash | least_conn)" ;;
  esac

  cat > "$STATE/nginx/nginx.conf" <<EOF
# generated by fleet.sh — do not edit
daemon off;
pid $STATE/nginx/nginx.pid;
error_log $STATE/nginx/logs/error.log warn;
events { worker_connections 1024; }
http {
    include /etc/nginx/mime.types;
    access_log $STATE/nginx/logs/access.log;
    client_body_temp_path $STATE/nginx/tmp/client;
    proxy_temp_path       $STATE/nginx/tmp/proxy;
    fastcgi_temp_path     $STATE/nginx/tmp/fastcgi;
    uwsgi_temp_path       $STATE/nginx/tmp/uwsgi;
    scgi_temp_path        $STATE/nginx/tmp/scgi;

    upstream fleet {
$balance
$upstream    }

    server {
        listen $POOL_HOST:$POOL_PORT;
        # Generation is long and streamed; a 60s default would cut off long completions.
        client_max_body_size 512m;
        proxy_read_timeout   3600s;
        proxy_send_timeout   3600s;

        location / {
            proxy_pass http://fleet;
            proxy_http_version 1.1;
            # If a shard is down, fail over to a live one rather than returning 502. Safe for
            # these POSTs: nginx only retries when the request was never delivered (connection
            # refused / shard still loading), which is exactly the dead-card case.
            proxy_next_upstream error timeout http_502 http_503;
            proxy_next_upstream_tries 3;
            # REQUIRED for streaming: with buffering on, nginx holds tokens until its buffer
            # fills, so the client sees nothing and then a burst instead of a live stream.
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header Connection "";
        }
    }
}
EOF
  nginx -t -p "$STATE/nginx" -c "$STATE/nginx/nginx.conf" 2>&1 | sed 's/^/  nginx: /'
}

# ── systemd wiring ────────────────────────────────────────────────────────────────────────────

# The target's .wants directory is the fleet's shard count made durable: `fleet use` rewrites it
# so that a plain `systemctl --user start llama-fleet.target` (including at boot) brings up
# exactly the right number of shards for the active profile.
sync_units() {
  # shellcheck disable=SC1091
  . "$STATE/plan.env"
  mkdir -p "$UNITS/llama-fleet.target.wants"
  rm -f "$UNITS/llama-fleet.target.wants"/llama-shard@*.service
  local i
  for i in $(seq 0 $((SHARD_COUNT - 1))); do
    ln -sf "$UNITS/llama-shard@.service" "$UNITS/llama-fleet.target.wants/llama-shard@${i}.service"
  done
  ln -sf "$UNITS/llama-pool.service"        "$UNITS/llama-fleet.target.wants/llama-pool.service"
  ln -sf "$UNITS/llama-fleet-probe.service" "$UNITS/llama-fleet.target.wants/llama-fleet-probe.service"
  systemctl --user daemon-reload
}

# Shards that are running but no longer in the plan must be stopped explicitly — systemd will
# not stop a template instance just because its .wants symlink vanished.
stop_stale_shards() {
  local keep="${1:-0}" u i
  for u in $(systemctl --user list-units --all --plain --no-legend 'llama-shard@*.service' 2>/dev/null | awk '{print $1}'); do
    i=$(echo "$u" | sed 's/.*@\([0-9]*\)\.service/\1/')
    if [ "$i" -ge "$keep" ] 2>/dev/null; then
      systemctl --user stop "$u" 2>/dev/null || true
      systemctl --user reset-failed "$u" 2>/dev/null || true
    fi
  done
}

# ── commands ──────────────────────────────────────────────────────────────────────────────────

cmd_list() {
  local a; a=$(active_profile)
  printf '%-22s %-9s %-8s %-7s %s\n' PROFILE CARDS/SH CTX PAR MODEL
  local f p CARDS_PER_SHARD CTX PARALLEL MODEL
  for f in "$FLEET"/profiles/*.conf; do
    p=$(basename "$f" .conf)
    CARDS_PER_SHARD=1; CTX=""; PARALLEL=""; MODEL=""
    # shellcheck disable=SC1090
    ( . "$f"; printf '%-22s %-9s %-8s %-7s %s\n' \
        "$p$([ "$p" = "$a" ] && echo ' *')" "$CARDS_PER_SHARD" "$CTX" "$PARALLEL" "$(basename "$MODEL")" )
  done
  echo
  echo "* = active. Pool listens on $POOL_HOST:$POOL_PORT"
}

cmd_plan() {
  local p="${1:-$(active_profile)}"
  [ -n "$p" ] || die "no active profile; run 'fleet use <profile>'"
  # Resolve into a scratch file: inspecting a profile must never disturb a running fleet.
  local tmp; tmp=$(mktemp "$STATE/plan.preview.XXXXXX")
  # shellcheck disable=SC2064
  trap "rm -f '$tmp'" RETURN
  resolve_plan "$p" "$tmp" >/dev/null
  # shellcheck disable=SC1090
  . "$tmp"
  local pf="$FLEET/profiles/${p}.conf"
  local MODEL="" CTX="" PARALLEL="" CACHE_TYPE="" MMAP=""
  # shellcheck disable=SC1090
  . "$pf"

  local sz_mib; sz_mib=$(( $(stat -c%s "$MODEL" 2>/dev/null || echo 0) / 1048576 ))
  local per_card=$(( sz_mib / CARDS_PER_SHARD ))

  echo "profile:      $p"
  echo "model:        $(basename "$MODEL")  (${sz_mib} MiB)"
  echo "healthy cards: $(good_cards)   [denied: ${DENY_CARDS:-none}]"
  echo "shards:       $SHARD_COUNT x ${CARDS_PER_SHARD} card(s)   routing=$ROUTING"
  echo "per shard:    ctx=$CTX parallel=$PARALLEL mmap=$MMAP kv=${CACHE_TYPE:-f16}"
  echo
  printf '  %-7s %-14s %-7s %-12s %s\n' SHARD CARDS PORT WEIGHTS/CARD 'HEADROOM FOR KV+BUFS'
  local i c pt
  for i in $(seq 0 $((SHARD_COUNT - 1))); do
    eval "c=\$SHARD_${i}_CARDS"; eval "pt=\$SHARD_${i}_PORT"
    printf '  %-7s %-14s %-7s %-12s %s MiB\n' "$i" "$c" "$pt" "${per_card} MiB" "$((CARD_MIB - per_card))"
  done
  echo
  if [ "$per_card" -ge "$CARD_MIB" ]; then
    echo "  !! weights alone exceed ${CARD_MIB} MiB/card — this will NOT load. Raise CARDS_PER_SHARD."
  elif [ "$((CARD_MIB - per_card))" -lt 900 ]; then
    echo "  !  under 900 MiB headroom/card — tight. Lower CTX, set CACHE_TYPE=q8_0, or raise CARDS_PER_SHARD."
  fi
}

cmd_use() {
  local p="${1:?usage: fleet use <profile>}"
  [ -f "$FLEET/profiles/${p}.conf" ] || die "no such profile: $p (try 'fleet list')"
  local old_count=0
  [ -f "$STATE/plan.env" ] && old_count=$(grep -oP '(?<=^SHARD_COUNT=)\d+' "$STATE/plan.env" || echo 0)
  echo "$p" > "$STATE/profile"
  resolve_plan "$p"
  sync_units
  # shellcheck disable=SC1091
  . "$STATE/plan.env"
  echo "active profile: $p  ($SHARD_COUNT shard(s), routing=$ROUTING)"
  if systemctl --user is-active --quiet llama-fleet.target 2>/dev/null; then
    echo "fleet is running — 'fleet restart' to apply."
  fi
}

cmd_up() {
  [ -f "$STATE/plan.env" ] || die "no plan; run 'fleet use <profile>' first"
  # shellcheck disable=SC1091
  . "$STATE/plan.env"
  stop_stale_shards "$SHARD_COUNT"
  # --no-block: the probe alone takes ~30s and each shard then streams GB to VRAM, so a
  # blocking start looks like a hang for minutes. Start detached and let the user poll.
  systemctl --user start --no-block llama-fleet.target
  echo "starting: $SHARD_COUNT shard(s) of '$PROFILE' + pool on $POOL_HOST:$POOL_PORT"
  echo "  (probe runs first, then shards load; watch with 'fleet status' or 'fleet wait')"
}

# Block until every shard answers /health, so scripts and benchmarks have something to wait on.
cmd_wait() {
  # shellcheck disable=SC1091
  . "$STATE/plan.env"
  local deadline=$((SECONDS + ${1:-600})) i pt ready
  while [ "$SECONDS" -lt "$deadline" ]; do
    ready=0
    for i in $(seq 0 $((SHARD_COUNT - 1))); do
      eval "pt=\$SHARD_${i}_PORT"
      [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$pt/health" 2>/dev/null)" = "200" ] && ready=$((ready + 1))
    done
    [ "$ready" -eq "$SHARD_COUNT" ] && { echo "all $SHARD_COUNT shards ready"; return 0; }
    sleep 5
  done
  echo "timed out with $ready/$SHARD_COUNT shards ready" >&2
  return 1
}

cmd_down() {
  systemctl --user stop llama-fleet.target 2>/dev/null || true
  stop_stale_shards 0
  systemctl --user stop llama-pool.service 2>/dev/null || true
  echo "fleet stopped"
}

cmd_restart() { cmd_down; cmd_up; }

cmd_reload() {
  gen_pool_conf
  systemctl --user reload-or-restart llama-pool.service
  echo "pool reloaded"
}

cmd_status() {
  local p; p=$(active_profile)
  [ -n "$p" ] || die "no active profile"
  # shellcheck disable=SC1091
  . "$STATE/plan.env"
  echo "profile: $p   shards: $SHARD_COUNT   routing: $ROUTING   pool: $POOL_HOST:$POOL_PORT"
  echo
  printf '  %-7s %-14s %-7s %-12s %s\n' SHARD CARDS PORT SYSTEMD HEALTH
  local i c pt st hl
  for i in $(seq 0 $((SHARD_COUNT - 1))); do
    eval "c=\$SHARD_${i}_CARDS"; eval "pt=\$SHARD_${i}_PORT"
    st=$(systemctl --user is-active "llama-shard@${i}.service" 2>/dev/null || true)
    hl=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$pt/health" 2>/dev/null || echo '---')
    printf '  %-7s %-14s %-7s %-12s %s\n' "$i" "$c" "$pt" "$st" "$hl"
  done
  echo
  printf '  pool     %-14s %-7s %-12s %s\n' '-' "$POOL_PORT" \
    "$(systemctl --user is-active llama-pool.service 2>/dev/null || true)" \
    "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$POOL_PORT/health" 2>/dev/null || echo '---')"
}

cmd_logs() {
  if [ -n "${1:-}" ]; then
    journalctl --user -u "llama-shard@${1}.service" -n 100 -f
  else
    journalctl --user -u 'llama-shard@*' -u llama-pool.service -u llama-fleet-probe.service -n 100 -f
  fi
}

case "${1:-}" in
  list)    shift; cmd_list "$@" ;;
  use)     shift; cmd_use "$@" ;;
  plan)    shift; cmd_plan "$@" ;;
  probe)   shift; cmd_probe "$@" ;;
  up)      shift; cmd_up "$@" ;;
  wait)    shift; cmd_wait "$@" ;;
  down)    shift; cmd_down "$@" ;;
  restart) shift; cmd_restart "$@" ;;
  reload)  shift; cmd_reload "$@" ;;
  status)  shift; cmd_status "$@" ;;
  logs)    shift; cmd_logs "$@" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 1 ;;
esac
