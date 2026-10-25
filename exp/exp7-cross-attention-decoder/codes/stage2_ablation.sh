#!/usr/bin/env bash
# H200 only. Separates the three changes bundled in cross_attention on the full stage-2 data (all training records):
# ground_residual (MLP + relative acceleration), local_snippet (+ local ground snippet),
# local_snippet_absolute (MLP + snippet, absolute acceleration). Three seeds each; finished runs are skipped.
set -u
EXP=$(cd "$(dirname "$0")/.." && pwd)
cd "$EXP"
source ~/seismic-exp7/envrc.sh
LOG=runs/stage2_ablation.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "ablation started at $(git rev-parse --short HEAD)"
for v in ground_residual local_snippet local_snippet_absolute; do
  for s in 20261024 20261025 20261026; do
    [ -f "results/stage2/$v/seed$s/report.json" ] && continue
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 70000 ]; do log "waiting for GPU"; sleep 600; done
    log "START $v seed $s"
    python codes/run.py --config configs/stage2.json --variant "$v" --seed "$s" > "runs/stage2_${v}_all_$s.log" 2>&1
    log "END $v seed $s exit=$?"
  done
done
python codes/summarize.py --config configs/stage2.json >> "$LOG" 2>&1
log "ABLATION_DONE"
