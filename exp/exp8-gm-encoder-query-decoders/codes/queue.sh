#!/usr/bin/env bash
# H200 only. exp8: three variants x three seeds on the exp7 dataset. First seed of every variant runs first;
# the queue stops if any of those fails. Finished runs are skipped, so the script can be started again.
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
cd "$EXP"
source ~/seismic-exp7/envrc.sh
mkdir -p runs
LOG=runs/queue.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "queue started at $(git rev-parse --short HEAD)"
run() {
  local v=$1 s=$2
  [ -f "results/exp8/$v/seed$s/report.json" ] && return 0
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 70000 ]; do log "waiting for GPU"; sleep 600; done
  log "START $v seed $s"
  python codes/run.py --variant "$v" --seed "$s" > "runs/${v}_$s.log" 2>&1
  local rc=$?
  log "END $v seed $s exit=$rc"
  return $rc
}
for v in gm_mlp gm_fno control_film_mlp; do run "$v" 20261024 || { log "first $v run failed; stopping"; exit 1; }; done
python codes/summarize.py >> "$LOG" 2>&1
for s in 20261025 20261026; do for v in gm_mlp gm_fno control_film_mlp; do run "$v" "$s"; done; done
python codes/summarize.py >> "$LOG" 2>&1
log "QUEUE_DONE"
