#!/bin/bash
# ASS-003 live comparison on main. OFF = every new switch off; ON = all on,
# one stop/relaunch at the half. Isolated roots under ~/pl_exp; never touches /workspace/devai.
set -u
EXP=$HOME/pl_exp; V=/workspace/devai/venv_mc; MIN=${MINUTES:-120}
mkdir -p $EXP/results
OVERLAY_OFF='
c["realtime_credit"]={"enabled":False}
c.setdefault("policy",{}).update({"logit_bound_valid_only":False,"kl_check":"minibatch","hold_intrinsic_without_extrinsic":False,"scouts_same_reward":False})
c.setdefault("dream_training",{}).update({"distill_min_weight":0.0,"distill_masked":False})
c.setdefault("infra",{})["farm_key"]="engine"
'
wait_stop() { # until the agent is gone
  for i in $(seq 1 200); do pgrep -f "$1/runlogs/run_config" >/dev/null || pgrep -f "SKYBOT_ROOT=$1" >/dev/null || break
    ps -eo args | grep -q "([r]un_minecraft|scripts/[r]un)[.]py --config runlogs/run_config.yaml" || break; sleep 5; done; }
start() { M=$1
  ( export SKYBOT_ROOT=$M; cd $M; rm -f runlogs/STOP
    setsid nohup bash scripts/supervise_skybot.sh 4000000 > runlogs/supervisor.log 2>&1 < /dev/null & )
  sleep 40; }
stop() { M=$1; touch $M/runlogs/STOP
  for i in $(seq 1 120); do ps -eo args | grep -q "([r]un_minecraft|scripts/[r]un)[.]py --config runlogs/run_config.yaml" || break; sleep 5; done
  sleep 8; }
snap() { A=$1; M=$EXP/$A/devai; D=$EXP/results/$A; mkdir -p $D
  cp $M/runlogs/minecraft_skybot_run.log* $M/runlogs/learning.jsonl $M/runlogs/supervisor.log $EXP/$A/samples.log $D/ 2>/dev/null
  cp $M/brain_universal/infra/autoscale.json $D/ 2>/dev/null
  $V/bin/python $M/tools/replay_lag_probe.py $M/brain_universal/logs/checkpoints/replay_buffer --rows 6000 > $D/lag_probe.txt 2>&1; }
run_arm() { A=$1; KIND=${2:-$1}; export SKYBOT_SEED=${3:-0}; RELAUNCH=${4:-0}
  M=$EXP/$A/devai; rm -rf $EXP/$A; mkdir -p $M/runlogs; cp -a $EXP/code/. $M/
  ln -s $V $M/venv_mc; cp -a /workspace/devai/roms $M/; cp /workspace/devai/runlogs/games.enabled $M/runlogs/
  (cd $M && $V/bin/python - "$KIND" "$OVERLAY_OFF" <<'PY'
import sys, yaml
arm, off = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open("configs/universal.yaml"))
if arm == "OFF":
    exec(off)
yaml.safe_dump(c, open("configs/universal.yaml", "w"), sort_keys=False)
g = yaml.safe_load(open("configs/games/doom.yaml")); g["game"]["log_every"] = 512
yaml.safe_dump(g, open("configs/games/doom.yaml", "w"), sort_keys=False)
PY
  )
  ( cd $EXP/$A; while true; do P=$(cat devai/runlogs/skybot_run.pid 2>/dev/null); R=$(ps -o rss= -p "$P" 2>/dev/null | awk '{printf "%d",$1/1024}')
      echo "$(date +%s) rss=${R:-0} avail=$(free -m | awk '/^Mem:/{print $7}') gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"; sleep 30; done ) > $EXP/$A/samples.log 2>&1 &
  SP=$!
  echo "$(date +%T) $A start" >> $EXP/progress.txt
  start $M
  if [ "$RELAUNCH" = 1 ]; then
    sleep $(( MIN * 30 )); stop $M; echo "$(date +%T) ON half-time stop; relaunching" >> $EXP/progress.txt
    cp $M/runlogs/minecraft_skybot_run.log $M/runlogs/run_first_half.log
    start $M; sleep $(( MIN * 30 ))
  else sleep $(( MIN * 60 )); fi
  stop $M; kill $SP 2>/dev/null; snap $A
  [ "$RELAUNCH" = 1 ] && cp $M/runlogs/run_first_half.log $EXP/results/$A/
  echo "$(date +%T) $A done" >> $EXP/progress.txt; }
# name:kind:seed:relaunch-at-half. Default order: the fixed ON rerun, then a
# second seed of each arm. (A quoted default inside ${ARMS:-"..."} is easy to
# get wrong, so the list is assigned on its own line.)
ARMS=${ARMS:-"ON_fix:ON:0:1 OFF_s1:OFF:1:0 ON_s1:ON:1:0"}
for spec in $ARMS; do
  IFS=: read -r n k sd rl <<< "$spec"
  echo "$(date +%T) spec $spec" >> $EXP/progress.txt
  run_arm "$n" "$k" "$sd" "$rl" < /dev/null
done
touch $EXP/ALL_DONE
