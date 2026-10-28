#!/bin/bash
# ONE phase of ONE seeded run under the adaptability protocol (ASS-007 12.2, 12.5, 14.6).
# Generalises tools/policy_learning_arms.sh: isolated root per run, a frozen overlay,
# a fixed seed, a stop at the phase budget, a snapshot between phases, logs collected.
# Called by tools/eval_runner.py, which owns ordering, resume and the crash rule.
# Runs ON THE TRAINING HOST. NOT RUN at build time (GOV-011 entry WP-14).
#
#   bash tools/eval_run.sh <phase> <run_id> <seed> <overlay.yaml|-> <games,csv> <minutes>
#
#   phase  source | target       source builds the root fresh and trains on <games>;
#                                target continues THE SAME root (the trained brain),
#                                rewrites games.enabled to <games> and relaunches.
#                                A fresh-control run has only a target phase: the
#                                root is built fresh if it does not exist.
# Exit: 0 phase completed to its budget; 3 crash (supervisor crash history or
# CRASHLOOP seen, or the agent never came up); 2 usage.
#
# Environment: EVAL_EXP (default ~/eval_exp), EVAL_VENV (default /workspace/devai/venv_mc),
# EVAL_CODE (a staged copy of the code under test, default $EVAL_EXP/code). Never edits
# /workspace/devai. The overlay is deep-merged into the root's configs/universal.yaml
# and is the ONLY difference between arms (changed arm = baseline + its switch).
set -u
[ $# -ge 6 ] || { echo "usage: eval_run.sh <source|target> <run_id> <seed> <overlay|-> <games,csv> <minutes>"; exit 2; }
PHASE=$1; RID=$2; SEED=$3; OVERLAY=$4; GAMES=$5; MIN=$6
EXP=${EVAL_EXP:-$HOME/eval_exp}; V=${EVAL_VENV:-/workspace/devai/venv_mc}; CODE=${EVAL_CODE:-$EXP/code}
ROOT=$EXP/runs/$RID; M=$ROOT/devai; RES=$ROOT/results/$PHASE
mkdir -p "$ROOT" "$RES"

alive() { ps -eo args | grep -q "([r]un_minecraft|scripts/[r]un)[.]py --config runlogs/run_config.yaml"; }
hms()   { date +%H:%M:%S; }

build_root() {
  rm -rf "$M"; mkdir -p "$M/runlogs"; cp -a "$CODE/." "$M/"
  ln -s "$V" "$M/venv_mc"; cp -a /workspace/devai/roms "$M/" 2>/dev/null || true
  # Evaluation-only Doom partition; normal run configs keep the regular map pool.
  (cd "$M" && "$V/bin/python" - <<'PY'
import yaml
p = "configs/games/doom.yaml"
c = yaml.safe_load(open(p))
c["game"]["held_out_maps"] = ["MAP29", "MAP30", "MAP31", "MAP32"]
# Evaluator line every 256 decisions per stream (about 50 s). The default of 5000 gives ONE
# line in a 20-minute target phase, so time-to-checkpoint could not be resolved
# (ASS-003 Amendment 1 set 512 for the same reason; this generalised script dropped it).
c["game"]["log_every"] = 256
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
PY
  ) || { echo "could not configure held-out Doom maps"; exit 3; }
  if [ "$OVERLAY" != "-" ]; then
    (cd "$M" && "$V/bin/python" - "$OVERLAY" <<'PY'
import sys, yaml
def merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
ov = yaml.safe_load(open(sys.argv[1])) or {}
# GOV-014 item C: a reserved top-level key merges into a GAME's YAML, after the held-out maps were
# set above (so configure_stream still refuses a leak), then is removed from the universal merge.
#   _games: {doom: {game: {map_assignment: {...}}}}
for gname, patch in (ov.pop("_games", None) or {}).items():
    gp = f"configs/games/{gname}.yaml"
    gc = yaml.safe_load(open(gp))
    merge(gc, patch)
    yaml.safe_dump(gc, open(gp, "w"), sort_keys=False)
c = yaml.safe_load(open("configs/universal.yaml"))
merge(c, ov)
yaml.safe_dump(c, open("configs/universal.yaml", "w"), sort_keys=False)
PY
    ) || { echo "overlay merge failed"; exit 3; }
  fi
}
write_games() { printf '%s\n' $(echo "$GAMES" | tr ',' ' ') > "$M/runlogs/games.enabled"; }
start() {
  ( export SKYBOT_ROOT=$M SKYBOT_SEED=$SEED SKYBOT_EVAL_PHASE=$PHASE; cd "$M"; rm -f runlogs/STOP
    setsid nohup bash scripts/supervise_skybot.sh 4000000 > runlogs/supervisor.log 2>&1 < /dev/null & )
  for i in $(seq 1 24); do sleep 5; alive && return 0; done
  echo "agent never came up"; return 1
}
stop() {
  touch "$M/runlogs/STOP"
  for i in $(seq 1 240); do alive || break; sleep 5; done
  sleep 8
}
crashed() { [ -f "$M/runlogs/CRASHLOOP" ] || [ -s "$M/runlogs/crash_history" ]; }

if [ "$PHASE" = source ] || [ ! -d "$M" ]; then
  [ "$PHASE" = source ] || echo "no source phase for this run: building a fresh root (fresh control)"
  build_root
fi
write_games
T0_EPOCH=$(date +%s); T0_HMS=$(hms)
start || { echo "{\"phase\":\"$PHASE\",\"crash\":\"never came up\"}" > "$ROOT/phase_$PHASE.json"; exit 3; }
END=$(( T0_EPOCH + ${MIN%.*} * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  sleep 30
  if crashed; then break; fi
done
CR=0; crashed && CR=1
stop
cp "$M"/runlogs/minecraft_skybot_run.log* "$M/runlogs/learning.jsonl" "$M/runlogs/supervisor.log" "$RES/" 2>/dev/null
[ -d "$M/runlogs" ] && cp "$M"/runlogs/crash_history "$M"/runlogs/CRASHLOOP "$RES/" 2>/dev/null
cp "$M/runlogs/run_config.yaml" "$RES/run_config.yaml" 2>/dev/null || true
# snapshot BETWEEN phases: the trained brain at source end (metrics/logs and the
# final checkpoint are kept; replay buffers are the 5 GB item to delete after analysis)
[ "$PHASE" = source ] && rsync -a "$M/brain_universal/" "$ROOT/snap_source_end/" 2>/dev/null
printf '{"phase":"%s","seed":%s,"games":"%s","minutes":%s,"start_epoch":%s,"start_hms":"%s","crash":%s}\n' \
  "$PHASE" "$SEED" "$GAMES" "$MIN" "$T0_EPOCH" "$T0_HMS" "$CR" > "$ROOT/phase_$PHASE.json"
[ "$CR" = 0 ] && exit 0 || exit 3
