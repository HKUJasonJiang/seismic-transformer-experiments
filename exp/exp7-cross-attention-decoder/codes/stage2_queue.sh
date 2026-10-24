#!/usr/bin/env bash
# H200 only. Waits for dataset generation, audits it, then runs stage 2 in the fixed order of exp7.md 7.3:
# main model (cross_attention, exp5_decoder control) for three seeds, then the record-count scaling curve.
# Stops if generation or the audit fails, or if either first stage-2 run fails. Finished runs are skipped,
# so the script can simply be started again. Progress: runs/stage2_queue.log
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
cd "$EXP"
source ~/seismic-exp7/envrc.sh
export EXP7_KNET_DIR="$HOME/seismic-exp7/All_GMs/KNET(3474)" EXP7_PEER_DIR="$HOME/seismic-exp7/All_GMs/PEER(65)"
LOG=runs/stage2_queue.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "queue started at $(git rev-parse --short HEAD)"

until grep -q '^EXIT=' runs/data/generate.log 2>/dev/null; do sleep 120; done
if ! grep -q '^EXIT=0' runs/data/generate.log; then log "generation failed; stopping"; exit 1; fi
log "generation finished"
if ! python codes/datagen/audit.py > runs/data/audit.log 2>&1; then log "audit FAILED (runs/data/audit.log); stopping"; exit 1; fi
log "audit PASS"

# Shared machine: never stop other jobs; wait until enough GPU memory is free.
wait_gpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge 70000 ] && return
    log "waiting for GPU memory (free ${free} MiB)"
    sleep 600
  done
}

run() {  # variant seed [records]
  local v=$1 s=$2 n=${3:-} name=$1 extra=()
  if [ -n "$n" ]; then name="$v/records$n"; extra=(--records "$n"); fi
  [ -f "results/stage2/$name/seed$s/report.json" ] && return 0
  wait_gpu
  log "START $name seed $s"
  python codes/run.py --config configs/stage2.json --variant "$v" --seed "$s" "${extra[@]}" > "runs/stage2_${v}_${n:-all}_$s.log" 2>&1
  local rc=$?
  log "END $name seed $s exit=$rc"
  return $rc
}

run cross_attention 20261024 || { log "first cross_attention run failed; stopping"; exit 1; }
run exp5_decoder 20261024 || { log "first exp5_decoder run failed; stopping"; exit 1; }
python codes/summarize.py --config configs/stage2.json >> "$LOG" 2>&1
for s in 20261025 20261026; do run cross_attention $s; run exp5_decoder $s; done
python codes/summarize.py --config configs/stage2.json >> "$LOG" 2>&1
for n in 7 30 100 300 1000; do
  for v in cross_attention exp5_decoder; do
    for s in 20261024 20261025 20261026; do run "$v" "$s" "$n"; done
  done
done
python codes/summarize.py --config configs/stage2.json >> "$LOG" 2>&1
log "QUEUE_DONE"
